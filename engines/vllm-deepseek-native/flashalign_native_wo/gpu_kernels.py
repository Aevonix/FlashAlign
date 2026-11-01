# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fixed-shape native WO masking/scatter; requires real CUDA qualification."""
import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _mask_tmp(Source, Dest, Owners, Indices, ROWS: tl.constexpr,
              RANK: tl.constexpr, GROUPS: tl.constexpr,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
              CAPACITY: tl.constexpr, HAS_INDICES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source_row = tl.load(Indices + row) if HAS_INDICES else row
    valid = (source_row >= 0) & (source_row < CAPACITY)
    owner = tl.load(Owners + source_row, valid, other=0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = row * S0 + (col % RANK) * S1 + (col // RANK) * S2
    # No real off-row intermediate is loaded. Zeros fill every inactive slot.
    if owner != 0:
        value = tl.load(Source + offset, col < RANK * GROUPS, other=0)
    else:
        value = tl.full((BLOCK,), 0, tl.bfloat16)
    tl.store(Dest + offset, value, col < RANK * GROUPS)


@triton.jit
def _scatter(Output, Edited, Owners, Indices, WIDTH: tl.constexpr,
             O0: tl.constexpr, O1: tl.constexpr, E0: tl.constexpr, E1: tl.constexpr,
             CAPACITY: tl.constexpr, HAS_INDICES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source_row = tl.load(Indices + row) if HAS_INDICES else row
    valid = (source_row >= 0) & (source_row < CAPACITY)
    owner = tl.load(Owners + source_row, valid, other=0)
    if owner != 0:
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Edited + row * E0 + col * E1, col < WIDTH, other=0)
        tl.store(Output + row * O0 + col * O1, value, col < WIDTH)


def _validate_rows(rows, tensor, indices):
    if (rows.ndim != 1 or rows.dtype != torch.int32 or rows.device != tensor.device
            or not rows.is_contiguous()):
        raise ValueError("Native owner rows must be contiguous device int32")
    if indices is not None:
        if (indices.ndim != 1 or indices.numel() != tensor.shape[0]
                or indices.dtype not in (torch.int32, torch.int64)
                or indices.device != tensor.device or not indices.is_contiguous()):
            raise ValueError("Native row map differs from attention rows")
    elif tensor.shape[0] > rows.numel():
        raise ValueError("Native attention rows exceed owner capacity")


def mask_tmp(source, dest, owners, indices):
    _validate_rows(owners, source, indices)
    if (source.ndim != 3 or source.dtype != torch.bfloat16
            or dest.shape != source.shape or dest.stride() != source.stride()
            or dest.dtype != source.dtype or dest.device != source.device
            or dest.data_ptr() == source.data_ptr()):
        raise ValueError("Native masked WO-A storage must be independent and shape matched")
    if source.shape[0]:
        _mask_tmp[(source.shape[0], triton.cdiv(source.shape[1] * source.shape[2], 256))](
            source, dest, owners, indices if indices is not None else owners,
            source.shape[0], source.shape[1], source.shape[2], *source.stride(),
            owners.numel(), indices is not None, 256)


def scatter_owner(output, edited, owners, indices):
    _validate_rows(owners, output, indices)
    if (output.ndim != 3 or output.shape[2] != 1 or output.shape != edited.shape
            or output.dtype != torch.bfloat16 or output.dtype != edited.dtype
            or output.device != edited.device or output.data_ptr() == edited.data_ptr()):
        raise ValueError("Native edited output must be independent and shape matched")
    if output.shape[0]:
        _scatter[(output.shape[0], triton.cdiv(output.shape[1], 256))](
            output, edited, owners, indices if indices is not None else owners,
            output.shape[1], output.stride(0), output.stride(1),
            edited.stride(0), edited.stride(1), owners.numel(), indices is not None, 256)
