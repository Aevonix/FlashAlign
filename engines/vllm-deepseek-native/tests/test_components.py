# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU contract checks only. No engine import, CUDA, or runtime qualification."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import os

import pytest

from flashalign_native_wo.loader import FORMAT, REQUIRES, NativePayload, load_spec
from flashalign_native_wo.planning import (ExecutionClass, Selection, classify_requests,
                               expand_capture_descriptors, request_owners,
                               require_native_runtime_qualified)
from flashalign_native_wo.rows import (compact_owner_reference, expand_final_starts,
                            map_layer_rows, merge_selected_reference)
from flashalign_native_wo.execution import run_native_wo
from flashalign_native_wo.stock_source import StockRanges

DIGEST = "a" * 64
OWNER = Selection("synthetic-native", DIGEST, 1.0)


def classify(scheduled, owners):
    return classify_requests(scheduled, owners, profile_name="synthetic-native", profile_digest=DIGEST)


def test_host_classification_no_device_token_count():
    assert classify({}, {}) is ExecutionClass.OFF
    assert classify({"a": 10}, {}) is ExecutionClass.OFF
    assert classify({"a": 10}, {"a": OWNER}) is ExecutionClass.OWNER
    assert classify({"a": 10, "b": 40}, {"a": OWNER}) is ExecutionClass.MIXED
    assert classify({"a": 0, "b": 40}, {"a": OWNER}) is ExecutionClass.OFF
    assert classify({"a": 4, "b": 0}, {"a": OWNER}) is ExecutionClass.OWNER


@pytest.mark.parametrize("selection", [Selection("other", DIGEST, 1),
    Selection("synthetic-native", "b" * 64, 1), Selection("synthetic-native", DIGEST, 1.1),
    Selection("synthetic-native", DIGEST, True), Selection("synthetic-native", DIGEST, float("nan"))])
def test_wrong_selected_identity_or_strength_fails(selection):
    with pytest.raises(ValueError):
        classify({"a": 4}, {"a": selection})


def test_reordered_batch_and_final_adaptive_boundaries():
    owners = request_owners(["b", "a", "c"], {"a": OWNER},
                           profile_name="synthetic-native", profile_digest=DIGEST)
    assert owners == (False, True, False)
    # CPU scheduler allowed 8/8/8; final adaptive GPU starts gave 1/3/0.
    rows = expand_final_starts((0, 1, 4, 4), owners, 8)
    assert rows == (False, True, True, True, False, False, False, False)
    assert map_layer_rows(rows, 6, (3, 0, -1, 2, 99, 1)) == (
        True, False, False, True, False, True)


def test_removed_and_reused_request_id_has_no_implicit_owner():
    assert classify({"old": 1}, {"old": OWNER}) is ExecutionClass.OWNER
    assert classify({"new": 1}, {"old": OWNER}) is ExecutionClass.OFF
    with pytest.raises(ValueError):
        request_owners(["a", "a"], {"a": OWNER}, profile_name="synthetic-native", profile_digest=DIGEST)


def test_scatter_retains_off_bytes_and_ignores_padding():
    stock = (b"nan-payload-1", b"stock-1", b"-0-bits", b"stock-3")
    compact = compact_owner_reference((False, True, False, True), 4)
    assert merge_selected_reference(stock, (b"keys-1", b"keys-3", b"bad", b"bad"), compact) == (
        stock[0], b"keys-1", stock[2], b"keys-3")
    zero = compact_owner_reference((False,) * 4, 4)
    assert merge_selected_reference(stock, (b"bad",) * 4, zero) == stock


@dataclass(frozen=True)
class Descriptor:
    tokens: int
    num_active_loras: int = 0
    alignment_case: int = 0


def test_capture_classes_have_distinct_keys_and_preserve_lora_field():
    bases = [Descriptor(8), Descriptor(16)]
    result = expand_capture_descriptors(bases, enabled=True)
    assert len(set(result)) == 6
    assert {row.alignment_case for row in result} == {0, 1, 2}
    assert all(row.num_active_loras == 0 for row in result)
    assert expand_capture_descriptors(bases, enabled=False) == bases


def test_preparation_never_implies_runtime_qualification():
    with pytest.raises(RuntimeError, match="not qualified"):
        require_native_runtime_qualified(native_kernels=False, graph_variants=True,
                                         row_mapping=True, tp_receipts=False)


def make_fixture(directory):
    tensors, contents, offset = {}, {}, 0
    for layer in (10, 11):
        for suffix, dtype, shape in (("weight", "F8_E4M3", [64, 128]),
                                     ("scale", "F8_E8M0", [2, 4])):
            name = f"layers.{layer}.attn.wo_b.{suffix}"
            raw = bytes((i + layer) % 64 for i in range(shape[0] * shape[1]))
            parts = [b"".join(raw[row*shape[1]+rank*(shape[1]//2):
                                 row*shape[1]+(rank+1)*(shape[1]//2)]
                              for row in range(shape[0])) for rank in range(2)]
            tensors[name] = {"dtype": dtype, "shape": shape,
                "data_offsets": [offset, offset + len(raw)],
                "edited_sha256": hashlib.sha256(raw).hexdigest(),
                "stock_sha256": hashlib.sha256(raw).hexdigest(),
                "edited_partitions": [hashlib.sha256(p).hexdigest() for p in parts],
                "stock_partitions": [hashlib.sha256(p).hexdigest() for p in parts]}
            contents[name] = raw
            offset += len(raw)
    header = json.dumps({k: {q: v[q] for q in ("dtype", "shape", "data_offsets")}
                         for k, v in tensors.items()}, separators=(",", ":")).encode()
    payload = struct.pack("<Q", len(header)) + header + b"".join(contents.values())
    (directory / "payload.safetensors").write_bytes(payload)
    spec = {"format": FORMAT, "name": "synthetic-native", "model": {"id": "stock", "revision": "a"*40},
        "source": {"id": "edited", "revision": "b"*40}, "hidden_size": 64, "input_size": 128,
        "layers": [10, 11], "tp_world_size": 2, "transform": "native_matrix_override_v1",
        "requires": list(REQUIRES), "strengths": [1.0], "tensors": tensors,
        "payload": {"file": "payload.safetensors", "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "header_size_bytes": len(header), "header_sha256": hashlib.sha256(header).hexdigest()}}
    return spec, contents


def write_spec(directory, spec):
    raw = json.dumps(spec, separators=(",", ":")).encode()
    path = directory / "profile.json"
    path.write_bytes(raw)
    return load_spec(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_streaming_payload_and_all_tp_partitions(tmp_path):
    spec, contents = make_fixture(tmp_path)
    loaded = write_spec(tmp_path, spec)
    with NativePayload(tmp_path, loaded) as payload:
        for tensor in loaded.tensors:
            parts = [payload.partition(tensor.name, rank) for rank in range(2)]
            rows, columns = tensor.shape
            width = columns // 2
            joined = b"".join(parts[rank][row*width:(row+1)*width]
                               for row in range(rows) for rank in range(2))
            assert joined == contents[tensor.name]
            for rank, part in enumerate(parts):
                payload.verify_stock_partition(tensor.name, rank, part)


@pytest.mark.parametrize("change", ["unknown", "strength", "pair", "shape", "gap", "sha", "path"])
def test_malformed_spec_rejected(tmp_path, change):
    spec, _ = make_fixture(tmp_path)
    first = next(iter(spec["tensors"].values()))
    if change == "unknown": spec["extra"] = 1
    if change == "strength": spec["strengths"] = [True]
    if change == "pair": spec["tensors"].pop("layers.10.attn.wo_b.scale")
    if change == "shape": first["shape"] = [64, 64]
    if change == "gap": first["data_offsets"] = [1, 8193]
    if change == "sha": spec["payload"]["sha256"] = "NOT-A-HASH"
    if change == "path": spec["payload"]["file"] = "../payload.safetensors"
    with pytest.raises(ValueError):
        write_spec(tmp_path, spec)


def test_payload_mutation_and_wrong_partition_fail(tmp_path):
    spec, _ = make_fixture(tmp_path)
    loaded = write_spec(tmp_path, spec)
    with NativePayload(tmp_path, loaded) as payload:
        with pytest.raises(ValueError): payload.verify_stock_partition(loaded.tensors[0].name, 0, b"bad")
        path = tmp_path / "payload.safetensors"
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with pytest.raises(ValueError): payload.partition(loaded.tensors[0].name, 0)
    with pytest.raises(ValueError): NativePayload(tmp_path, loaded)


def test_payload_symlink_rejected(tmp_path):
    spec, _ = make_fixture(tmp_path)
    loaded = write_spec(tmp_path, spec)
    source = tmp_path / "payload.safetensors"
    original = tmp_path / "original"
    source.rename(original); source.symlink_to(original)
    with pytest.raises(OSError): NativePayload(tmp_path, loaded)


@dataclass(frozen=True)
class FakeWeights:
    wo_a: object
    wo_b: object


@dataclass(frozen=True)
class FakeBinding:
    weights: FakeWeights
    tmp: object
    output: list


def test_native_execution_branches_use_one_wo_a_and_only_selected_wo_b():
    shared = object()
    stock = FakeWeights(shared, "stock")
    edited = FakeWeights(shared, "synthetic-native")
    calls = []
    binding = FakeBinding(stock, "original-A", [b"stock-off", b"stock-owner"])

    class State:
        def run_inv_rope(self, bound, *, stream):
            calls.append(("one-A-then-B", bound.weights.wo_b, stream))
            return bound.output

    class Operations:
        def mask_tmp(self, source, owners, indices):
            calls.append(("mask", source, owners, indices))
            return "only-owner-A"
        def run_b(self, state, bound, alt, masked, stream):
            calls.append(("B-only", alt.wo_b, masked))
            return [b"zero-padding", b"keys-owner"]
        def scatter_owner(self, output, alt, owners, indices):
            calls.append(("scatter",))
            output[1] = alt[1]

    state, ops = State(), Operations()
    assert run_native_wo(binding, state, edited, 0, None, None, "stream", operations=ops) is binding.output
    assert calls == [("one-A-then-B", "stock", "stream")]
    calls.clear()
    run_native_wo(binding, state, edited, 1, None, None, "stream", operations=ops)
    assert calls == [("one-A-then-B", "synthetic-native", "stream")]
    assert binding.weights is stock
    calls.clear()
    out = run_native_wo(binding, state, edited, 2, (False, True), None, "stream", operations=ops)
    assert out == [b"stock-off", b"keys-owner"]
    assert [item[0] for item in calls] == ["one-A-then-B", "mask", "B-only", "scatter"]


def test_native_execution_rejects_unshared_a_and_missing_mixed_owner_rows():
    stock = FakeWeights(object(), "stock")
    binding = FakeBinding(stock, None, [])
    with pytest.raises(ValueError, match="shared"):
        run_native_wo(binding, object(), FakeWeights(object(), "synthetic-native"), 1, None, None, None)
    with pytest.raises(ValueError, match="ownership"):
        run_native_wo(binding, object(), FakeWeights(stock.wo_a, "synthetic-native"), 2, None, None, None)


def test_stock_native_ranges_are_bound_to_full_file_receipt(tmp_path):
    name = "layers.10.attn.wo_a.weight"
    raw = bytes(range(64)) * 32
    header = json.dumps({name: {"dtype":"F8_E4M3","shape":[64,32],
                               "data_offsets":[0,len(raw)]}}).encode()
    whole = struct.pack("<Q",len(header)) + header + raw
    shard = "model-1.safetensors"
    path = tmp_path/shard; path.write_bytes(whole)
    index = json.dumps({"weight_map":{name:shard}}).encode()
    (tmp_path/"model.safetensors.index.json").write_bytes(index)
    manifest = {"stock":{"index_sha256":hashlib.sha256(index).hexdigest(),"shards":[
        {"file":shard,"size_bytes":len(whole),"header_sha256":hashlib.sha256(header).hexdigest()}]}}
    manifest_raw = json.dumps(manifest).encode(); manifest_path=tmp_path/"manifest.json"
    manifest_path.write_bytes(manifest_raw)
    info = path.stat()
    receipt = {"pass":True,"sha256_matches_hf_lfs":True,"file_identity_unchanged":True,
        "full_payload_sha256_computed":True,"sha256":hashlib.sha256(whole).hexdigest(),
        "expected_sha256":hashlib.sha256(whole).hexdigest(),"after_fstat":{
            "device":info.st_dev,"inode":info.st_ino,"size":info.st_size,
            "mtime_ns":info.st_mtime_ns,"ctime_ns":info.st_ctime_ns}}
    (tmp_path/(shard+".json")).write_text(json.dumps(receipt))
    source = StockRanges(tmp_path,manifest_path,hashlib.sha256(manifest_raw).hexdigest(),tmp_path)
    try:
        rows,shape,dtype=source.partition(name,1,2,0)
        assert rows==raw[len(raw)//2:] and shape==(32,32)
        columns,shape,dtype=source.partition(name,1,2,1)
        assert columns==b"".join(raw[r*32+16:(r+1)*32] for r in range(64))
        altered=bytearray(whole);altered[-1]^=1;path.write_bytes(altered)
        with pytest.raises(ValueError,match="changed"):
            source.partition(name,0,2,0)
    finally: source.close()
