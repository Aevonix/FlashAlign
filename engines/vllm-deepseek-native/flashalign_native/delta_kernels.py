# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Row-gated low-rank additions; rows that selected no profile are never loaded or stored.

Both functions run inside the engine's opaque attention and MoE operations, so
the compiler never sees their writes. The kernels need real CUDA qualification.
The CPU reference is delta_profile.reference_delta.
"""
import torch
from vllm.triton_utils import triton, tl


@triton.jit(do_not_specialize=["ROWS"])
def _stage(RowMap, Indices, ROWS, HAS_INDICES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < ROWS
    if HAS_INDICES:
        source = tl.load(Indices + row, live, other=-1).to(tl.int32)
    else:
        source = row
    tl.store(RowMap + row, source, live)


@triton.jit(do_not_specialize=["S_GROUP"])
def _delta(X, Source, B, Q, Owners, Strengths, RowMap, S_GROUP, SITE_SCALE,
           WIDTH: tl.constexpr, X_ROW: tl.constexpr, X_COL: tl.constexpr,
           INNER: tl.constexpr, GROUPS: tl.constexpr,
           S_ROW: tl.constexpr, S_INNER: tl.constexpr,
           RANK: tl.constexpr, CAPACITY: tl.constexpr,
           IN_BLOCK: tl.constexpr, OUT_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source_row = tl.load(RowMap + row)
    valid = (source_row >= 0) & (source_row < CAPACITY)
    owner = tl.load(Owners + source_row, valid, other=0)
    # This is a branch, not zero times a delta. Rows that did not select the
    # profile can contain NaN and retain their bits; nothing is loaded from
    # or stored to them.
    if owner != 0:
        inner = tl.arange(0, IN_BLOCK)
        col = tl.arange(0, OUT_BLOCK)
        total = tl.zeros((OUT_BLOCK,), dtype=tl.float32)
        for k in range(RANK):
            products = tl.zeros((IN_BLOCK,), dtype=tl.float32)
            for group in range(GROUPS):
                value = tl.load(Source + row * S_ROW + group * S_GROUP + inner * S_INNER,
                                inner < INNER, other=0).to(tl.float32)
                weight = tl.load(B + k * (INNER * GROUPS) + group * INNER + inner,
                                 inner < INNER, other=0)
                products += value * weight
            direction = tl.load(Q + k * WIDTH + col, col < WIDTH, other=0)
            total += tl.sum(products, 0) * direction
        alpha = tl.load(Strengths + source_row) * SITE_SCALE
        value = tl.load(X + row * X_ROW + col * X_COL, col < WIDTH, other=0).to(tl.float32)
        # One final store: the sum over every direction is rounded once.
        tl.store(X + row * X_ROW + col * X_COL, value + alpha * total, col < WIDTH)


def stage_rows(row_map: torch.Tensor, x: torch.Tensor, indices: torch.Tensor | None) -> None:
    """Record which request row each row of x came from; identity without a row map."""
    rows = x.shape[0]
    if row_map.ndim != 1 or row_map.dtype != torch.int32 or not row_map.is_contiguous() or rows > row_map.numel():
        raise ValueError("Rows exceed the staged row-map capacity")
    if indices is not None and (indices.ndim != 1 or indices.numel() != rows or not indices.is_contiguous()
                                or indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Row map differs from model rows")
    if rows:
        _stage[(triton.cdiv(rows, 128),)](row_map, indices if indices is not None else row_map,
                                          rows, indices is not None, 128)


def delta_inplace(x, source, b, q, owners, strengths, row_map, site_scale):
    """x [rows, hidden] += strength * scale * (source B^T) Q on the rows that selected the profile.

    source is this rank's writer input: [rows, input] or the grouped [rows, inner, groups]
    view, whose column order is group-major.
    """
    if (x.ndim != 2 or q.ndim != 2 or q.shape[1] != x.shape[1] or q.dtype != torch.float32
            or not q.is_contiguous() or b.ndim != 2 or b.dtype != torch.float32 or not b.is_contiguous()
            or b.shape[0] != q.shape[0]):
        raise ValueError("Invalid low-rank operator shape")
    if source.ndim == 2:
        (rows, inner), groups = source.shape, 1
        (s_row, s_inner), s_group = source.stride(), 0
    elif source.ndim == 3:
        rows, inner, groups = source.shape
        s_row, s_inner, s_group = source.stride()
    else:
        raise ValueError("Unsupported writer input layout")
    if rows != x.shape[0] or b.shape[1] != inner * groups or source.device != x.device:
        raise ValueError("Writer input differs from the low-rank operator")
    if (owners.ndim != 1 or owners.dtype != torch.int32 or strengths.ndim != 1
            or strengths.dtype != torch.float32 or strengths.numel() != owners.numel()
            or not owners.is_contiguous() or not strengths.is_contiguous()
            or row_map.ndim != 1 or row_map.dtype != torch.int32 or rows > row_map.numel()):
        raise ValueError("Invalid row state for a low-rank profile")
    if rows:
        # Row count, group stride and site scale are launch values: one compiled
        # kernel serves every batch size and every site of the same shape.
        _delta[(rows,)](x, source, b, q, owners, strengths, row_map, s_group, float(site_scale),
                        x.shape[1], x.stride(0), x.stride(1), inner, groups, s_row, s_inner,
                        q.shape[0], owners.numel(),
                        triton.next_power_of_2(inner), triton.next_power_of_2(x.shape[1]),
                        num_warps=8)
