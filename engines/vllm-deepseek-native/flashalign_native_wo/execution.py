# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Exact native WO-B execution for the pinned prepared engine.

No installation side effects. The loader/graph integration must opt in.
CPU orchestration tests do not qualify these GPU kernels or their graph use.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib

from .planning import ExecutionClass


def install_native_weights(attention, profile, payload, rank):
    """Create a second immutable packed binding without modifying stock storage."""
    import torch
    from b12x.gemm._shared.wo_mxfp8 import pack_fp8_block_scaled_weight_mxfp8

    layer = getattr(attention, "layer_id", None)
    if type(layer) is not int or layer < 0:
        raise ValueError("Native WO-B attention has no valid layer identity")
    if layer not in profile.layers:
        return None
    if hasattr(attention, "_flashalign_native_weights"):
        installed = getattr(attention, "_flashalign_native_install", None)
        if (installed is None or installed[0] != profile.digest
                or installed[1] is not attention._wo_projection_weights):
            raise ValueError("Native installation changed its profile or stock binding")
        return installed[2]
    if (attention.hidden_size != profile.hidden_size
            or attention.n_local_groups * attention.o_lora_rank
            != profile.input_size // profile.tp_world_size):
        raise ValueError("Native WO-B shape differs from profile")
    stock = attention._wo_projection_weights
    if stock is None:
        raise ValueError("Stock WO projection must be packed first")
    inputs = []
    tensor_receipts = []
    original_tensors = []
    for suffix, value, dtype in (
            ("weight", attention.wo_b.weight, torch.float8_e4m3fn),
            ("scale", attention.wo_b.weight_scale_inv, torch.float8_e8m0fnu)):
        name = f"layers.{layer}.attn.wo_b.{suffix}"
        t = profile.tensor(name)
        shape = (t.shape[0], t.shape[1] // profile.tp_world_size)
        if value.dtype != dtype or tuple(value.shape) != shape:
            raise ValueError("Native serving tensor representation differs from profile")
        # Setup only: this CPU copy validates actual serving stock bytes. It is
        # outside requests and graph capture, and never writes stock storage.
        before = value.detach().view(torch.uint8).cpu().contiguous().numpy().tobytes()
        payload.verify_stock_partition(name, rank, before)
        raw = payload.partition(name, rank)
        native = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(shape)
        native = native.to(device=value.device, copy=True)
        if native.data_ptr() == value.data_ptr():
            raise RuntimeError("Alternate native storage aliases stock")
        device_bytes = native.view(torch.uint8).cpu().contiguous().numpy().tobytes()
        if hashlib.sha256(device_bytes).hexdigest() != t.edited_partitions[rank]:
            raise RuntimeError("Copied native device bytes differ from source")
        inputs.append(native)
        original_tensors.append((value, value.data_ptr(), before))
        tensor_receipts.append({"tensor": name, "stock_sha256": hashlib.sha256(before).hexdigest(),
                                "edited_sha256": hashlib.sha256(raw).hexdigest(),
                                "device_sha256": hashlib.sha256(device_bytes).hexdigest(),
                                "bytes": len(raw), "independent_storage": True})
    packed = pack_fp8_block_scaled_weight_mxfp8(
        inputs[0], inputs[1], m=profile.hidden_size,
        k=profile.input_size // profile.tp_world_size, num_groups=1, block_size=(32, 32))
    alternate = replace(stock, wo_b=packed)
    if alternate.wo_a is not stock.wo_a:
        raise RuntimeError("Alternate WO binding must share unchanged WO-A")
    packed_bytes = packed.values.view(torch.uint8).cpu().contiguous().numpy().tobytes()
    if hashlib.sha256(packed_bytes).hexdigest() != tensor_receipts[0]["edited_sha256"]:
        raise RuntimeError("Packed WO-B values differ from exact source bytes")
    expected_scales = inputs[1].view(torch.uint8).repeat_interleave(32, dim=0).unsqueeze(0)
    if not torch.equal(packed.scale_rows.view(torch.uint8), expected_scales):
        raise RuntimeError("Packed WO-B row scales differ from exact source scales")
    logical_mma_scales = packed.scale_mma.view(torch.uint8).permute(5, 2, 1, 0, 4, 3).contiguous()
    logical_mma_scales = logical_mma_scales.reshape(expected_scales.shape)
    if not torch.equal(logical_mma_scales, expected_scales):
        raise RuntimeError("Packed WO-B MMA scales differ from exact source scales")
    for value, pointer, before in original_tensors:
        after = value.detach().view(torch.uint8).cpu().contiguous().numpy().tobytes()
        if value.data_ptr() != pointer or after != before:
            raise RuntimeError("Native installation changed stock storage")
    if attention._wo_projection_weights is not stock:
        raise RuntimeError("Native installation changed the stock packed binding")
    attention._flashalign_native_weights = alternate
    receipt = {"layer": layer, "tp_rank": rank, "profile_digest": profile.digest,
               "shared_stock_wo_a": True, "packed_values_exact": True,
               "packed_row_and_mma_scales_exact": True, "stock_storage_unchanged": True,
               "tensors": tensor_receipts}
    attention._flashalign_native_install = (profile.digest, stock, receipt)
    return receipt


def run_native_wo(binding, state, alternate, mode, owner_rows, row_indices, stream,
                  *, operations=None):
    """Execute one local WO result, before the caller's original TP reduction.

    The OFF branch is retained for independent orchestration tests. The engine
    patch calls its unchanged original public function directly for OFF.
    """
    if type(mode) not in (int, ExecutionClass) or mode not in (0, 1, 2):
        raise ValueError("Unknown native execution class")
    if mode == ExecutionClass.OFF:
        return state.run_inv_rope(binding, stream=stream)
    if alternate is None:
        raise ValueError("Missing immutable alternate native weights")
    if alternate.wo_a is not binding.weights.wo_a:
        raise ValueError("Native WO-A must remain shared with stock")
    if mode == ExecutionClass.OWNER:
        return state.run_inv_rope(replace(binding, weights=alternate), stream=stream)
    if owner_rows is None:
        raise ValueError("Mixed native execution requires device ownership rows")
    operations = operations or _TorchOperations()
    result = state.run_inv_rope(binding, stream=stream)
    # The stock call leaves its common WO-A result in binding.tmp. Separate
    # storage prevents corrupting that scratch or changing off-row bytes.
    masked = operations.mask_tmp(binding.tmp, owner_rows, row_indices)
    edited_output = operations.run_b(state, binding, alternate, masked, stream)
    operations.scatter_owner(binding.output, edited_output, owner_rows, row_indices)
    return result


class _TorchOperations:
    def mask_tmp(self, source, owner_rows, row_indices):
        import torch
        from .gpu_kernels import mask_tmp
        if source.ndim != 3 or source.dtype != torch.bfloat16:
            raise ValueError("Unexpected common WO-A intermediate")
        masked = torch.empty_strided(source.shape, source.stride(),
                                     dtype=source.dtype, device=source.device)
        mask_tmp(source, masked, owner_rows, row_indices)
        return masked

    def run_b(self, state, binding, alternate, masked, stream):
        import torch
        rows = masked.shape[0]
        output = torch.empty_strided(binding.output.shape, binding.output.stride(),
                                    dtype=binding.output.dtype, device=masked.device)
        weights = alternate.wo_b
        if state.fused_b is not None:
            source = masked if alternate.groups != 1 else masked.as_strided(
                (rows, alternate.rank), (alternate.rank, 1))
            state.fused_b.run(source, weights.values.reshape(alternate.hidden, -1, 1),
                              weights.scale_mma, out=output, stream=stream,
                              rhs_values_tiled=(weights.values_tiled if state.query.wo_b_tiled else None))
        else:
            # Same prepared quantizer and native GEMM as the stock call. Input
            # quantization is per row, so owner values are unchanged by zeros in
            # other rows. This statement still needs real GPU parity evidence.
            quantized = state.quantize_b(masked)
            state.ordinary_b.run((quantized.values.unsqueeze(-1), quantized.scale_mma),
                                 (weights.values.unsqueeze(-1), weights.scale_mma),
                                 out=output, alpha=None, stream=stream)
        return output

    def scatter_owner(self, output, edited, owner_rows, row_indices):
        from .gpu_kernels import scatter_owner
        scatter_owner(output, edited, owner_rows, row_indices)
