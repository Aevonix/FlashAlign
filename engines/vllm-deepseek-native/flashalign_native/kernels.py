# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CUDA graph compatible projection; the stock branch performs no transform."""
import torch
from vllm.triton_utils import triton, tl
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _rows(Starts, Owners, Strengths, RowOwners, RowStrengths,
          NREQ: tl.constexpr, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    owner = tl.full((BLOCK,), 0, tl.int32)
    strength = tl.full((BLOCK,), 0, tl.float32)
    for request in range(NREQ):
        start = tl.load(Starts + request)
        end = tl.load(Starts + request + 1)
        present = (row >= start) & (row < end)
        owner = tl.where(present, tl.load(Owners + request), owner)
        strength = tl.where(present, tl.load(Strengths + request), strength)
    tl.store(RowOwners + row, owner, row < CAPACITY)
    tl.store(RowStrengths + row, strength, row < CAPACITY)


@triton.jit
def _project(X, Q, Owners, Strengths, Indices, Excluded,
             ROWS: tl.constexpr, WIDTH: tl.constexpr,
             STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr,
             CAPACITY: tl.constexpr, HAS_INDICES: tl.constexpr,
             SITE_SCALE: tl.constexpr, HAS_EXCLUDED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source_row = tl.load(Indices + row) if HAS_INDICES else row
    valid = (source_row >= 0) & (source_row < CAPACITY)
    owner = tl.load(Owners + source_row, valid, other=0)
    if HAS_EXCLUDED:
        owner = tl.where(tl.load(Excluded + row), 0, owner)
    # This is a branch, not zero times a transform. Off rows can contain NaN
    # and retain their bits; no x/q load, reduction or store occurs for them.
    if owner != 0:
        col = tl.arange(0, BLOCK)
        value = tl.load(X + row * STRIDE_ROW + col * STRIDE_COL,
                        col < WIDTH, other=0).to(tl.float32)
        q = tl.load(Q + col, col < WIDTH, other=0)
        dot = tl.sum(value * q, 0)
        alpha = tl.load(Strengths + source_row)
        shifted = value - (alpha * SITE_SCALE) * dot * q
        tl.store(X + row * STRIDE_ROW + col * STRIDE_COL, shifted, col < WIDTH)


def prepare_rows(starts, owners, strengths, row_owners, row_strengths, nreq):
    _rows[(triton.cdiv(row_owners.numel(), 128),)](
        starts, owners, strengths, row_owners, row_strengths,
        nreq, row_owners.numel(), 128)


def project_inplace(x: torch.Tensor, q: torch.Tensor, owners: torch.Tensor,
                    strengths: torch.Tensor, indices: torch.Tensor | None,
                    site_scale: float, excluded: torch.Tensor | None) -> None:
    if x.ndim != 2 or q.ndim != 1 or x.shape[1] != q.numel():
        raise ValueError("Invalid projection shape")
    if indices is not None and indices.numel() != x.shape[0]:
        raise ValueError("CED row map differs from projection rows")
    if indices is None and x.shape[0] > owners.numel():
        raise ValueError("Projection exceeds row-state capacity")
    if excluded is not None and excluded.numel() != x.shape[0]:
        raise ValueError("Embedding exclusion mask differs from projection rows")
    if x.numel():
        _project[(x.shape[0],)](x, q, owners, strengths,
            indices if indices is not None else owners,
            excluded if excluded is not None else owners,
            x.shape[0], x.shape[1], x.stride(0), x.stride(1), owners.numel(),
            indices is not None, site_scale, excluded is not None, triton.next_power_of_2(x.shape[1]),
            num_warps=8)


def _fake(x, q, owners, strengths, indices, site_scale, excluded):
    return None


direct_register_custom_op("flashalign_project_inplace", project_inplace,
                         mutates_args=["x"], fake_impl=_fake)


def apply(x, q, owners, strengths, indices, scale, excluded=None):
    torch.ops.vllm.flashalign_project_inplace(x, q, owners, strengths, indices, scale, excluded)
    return x


@triton.jit
def _project_rank2(X, Q, Owners, Strengths, Indices, Excluded,
                   ROWS: tl.constexpr, WIDTH: tl.constexpr,
                   STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr,
                   CAPACITY: tl.constexpr, HAS_INDICES: tl.constexpr,
                   SITE_SCALE: tl.constexpr, HAS_EXCLUDED: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source_row = tl.load(Indices + row) if HAS_INDICES else row
    valid = (source_row >= 0) & (source_row < CAPACITY)
    owner = tl.load(Owners + source_row, valid, other=0)
    if HAS_EXCLUDED:
        owner = tl.where(tl.load(Excluded + row), 0, owner)
    if owner != 0:
        col = tl.arange(0, BLOCK)
        value = tl.load(X + row * STRIDE_ROW + col * STRIDE_COL,
                        col < WIDTH, other=0).to(tl.float32)
        q0 = tl.load(Q + col, col < WIDTH, other=0)
        q1 = tl.load(Q + WIDTH + col, col < WIDTH, other=0)
        # Both components use the original row. A single final store avoids
        # intermediate BF16 rounding from two sequential rank-one edits.
        dot0 = tl.sum(value * q0, 0)
        dot1 = tl.sum(value * q1, 0)
        alpha = tl.load(Strengths + source_row)
        shifted = value - (alpha * SITE_SCALE) * (dot0 * q0 + dot1 * q1)
        tl.store(X + row * STRIDE_ROW + col * STRIDE_COL, shifted, col < WIDTH)


def project_rank2_inplace(x: torch.Tensor, q: torch.Tensor, owners: torch.Tensor,
                          strengths: torch.Tensor, indices: torch.Tensor | None,
                          site_scale: float, excluded: torch.Tensor | None) -> None:
    if (x.ndim != 2 or q.ndim != 2 or tuple(q.shape) != (2, x.shape[1])
            or q.dtype != torch.float32 or not q.is_contiguous()):
        raise ValueError("Invalid rank-two projection shape or basis")
    if indices is not None and indices.numel() != x.shape[0]:
        raise ValueError("CED row map differs from projection rows")
    if indices is None and x.shape[0] > owners.numel():
        raise ValueError("Projection exceeds row-state capacity")
    if excluded is not None and excluded.numel() != x.shape[0]:
        raise ValueError("Embedding exclusion mask differs from projection rows")
    if x.numel():
        _project_rank2[(x.shape[0],)](x, q, owners, strengths,
            indices if indices is not None else owners,
            excluded if excluded is not None else owners,
            x.shape[0], x.shape[1], x.stride(0), x.stride(1), owners.numel(),
            indices is not None, site_scale, excluded is not None,
            triton.next_power_of_2(x.shape[1]), num_warps=8)


direct_register_custom_op("flashalign_project_rank2_inplace", project_rank2_inplace,
                         mutates_args=["x"], fake_impl=_fake)


def apply_rank2(x, q, owners, strengths, indices, scale, excluded=None):
    torch.ops.vllm.flashalign_project_rank2_inplace(x, q, owners, strengths,
                                                 indices, scale, excluded)
    return x
