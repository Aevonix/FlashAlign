# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Independent FP32 rank-one projection with an exact off-row bypass."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rank_one_writer_projection(
    Y, SLOTS, ALPHAS, Q, SCALES, FIRED,
    HIDDEN: tl.constexpr, STRIDE_ROW: tl.constexpr, STRIDE_COL: tl.constexpr,
    Q_ROW: tl.constexpr, BLOCK: tl.constexpr, ATTEST: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row)
    # Runtime GPU branch, deliberately outside every Y/Q load and operation.
    # The branch itself is captured; later replays may change row ownership.
    if slot != 0:
        coefficient = tl.load(ALPHAS + row) * tl.load(SCALES + slot)
        if coefficient != 0.:
            columns = tl.arange(0, BLOCK)
            q = tl.load(Q + slot * Q_ROW + columns, columns < HIDDEN, 0.)
            pointers = Y + row * STRIDE_ROW + columns * STRIDE_COL
            value = tl.load(pointers, columns < HIDDEN, 0.).to(tl.float32)
            component = tl.sum(value * q, 0)
            updated = value - (coefficient * component) * q
            tl.store(pointers, updated, columns < HIDDEN)
            if ATTEST:
                tl.atomic_add(FIRED, 1, sem="relaxed")


def project_inplace(values, state, q_bank, site_scales, fired=None, compiled_sink=None):
    """In-place at a named matrix output, before its caller adds residuals."""
    if values.ndim != 2 or values.shape[0] > state.capacity:
        raise ValueError("Unsupported writer-output shape")
    if values.stride(1) <= 0 or values.stride(0) <= (values.shape[1] - 1) * values.stride(1):
        raise ValueError("Overlapping output strides are not supported")
    if values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Unqualified activation dtype")
    if values.device != state.slots.device or values.device != q_bank.device:
        raise ValueError("Cross-device row state")
    if q_bank.dtype != torch.float32 or site_scales.dtype != torch.float32:
        raise ValueError("Profile parameters must remain FP32")
    if q_bank.ndim != 2 or q_bank.shape[1] != values.shape[1]:
        raise ValueError("Rank-one profile bank geometry mismatch")
    if q_bank.shape[0] != site_scales.numel() or q_bank.stride(1) != 1:
        raise ValueError("Invalid profile bank layout")
    if state.profile_count + 1 != q_bank.shape[0]:
        raise ValueError("Profile bank does not match the immutable slot registry")
    if values.shape[0] == 0:
        return values
    compiled = _rank_one_writer_projection[(values.shape[0],)](
        values, state.slots, state.strengths, q_bank, site_scales,
        fired if fired is not None else state.unused_counter,
        HIDDEN=values.shape[1], STRIDE_ROW=values.stride(0), STRIDE_COL=values.stride(1),
        Q_ROW=q_bank.stride(0), BLOCK=triton.next_power_of_2(values.shape[1]),
        ATTEST=fired is not None, num_warps=8 if values.shape[1] > 4096 else 4,
        enable_fp_fusion=False,
    )
    if compiled_sink is not None:
        compiled_sink.append(compiled)
    return values
