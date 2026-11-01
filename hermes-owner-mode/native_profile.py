# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Bounded read-only native WO-B artifact validation. No model/GPU dependencies.

This separate format never passes through the compact output-profile loader.
Successful parsing proves artifact integrity, not serving qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct

FORMAT = "flashalign.native_wo_b.v1"
REQUIRES = ("native_e4m3_ue8m0_block32", "immutable_dual_wo_bindings",
            "graph_safe_owner_rows", "pre_tp_reduce_pre_mhc")


def _keys(x, keys):
    if not isinstance(x, dict) or set(x) != set(keys):
        raise ValueError("Unexpected artifact fields")


def _sha(x):
    if not isinstance(x, str) or re.fullmatch(r"[0-9a-f]{64}", x) is None:
        raise ValueError("Invalid SHA-256")
    return x


def _pin(x):
    _keys(x, ("id", "revision"))
    if (not isinstance(x["id"], str) or not x["id"]
            or not isinstance(x["revision"], str)
            or re.fullmatch(r"[0-9a-f]{40}", x["revision"]) is None):
        raise ValueError("Model/source must have an exact revision")
    return x["id"], x["revision"]


def _unique(pairs):
    x = {}
    for key, value in pairs:
        if key in x:
            raise ValueError("Duplicate artifact field")
        x[key] = value
    return x


def _json(raw):
    def invalid(value):
        raise ValueError("Non-finite JSON constant")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)


def _identity(fd):
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("Artifact must be a regular file")
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _open(path):
    return os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                   | getattr(os, "O_NONBLOCK", 0))


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, int]
    offsets: tuple[int, int]
    edited_sha256: str
    stock_sha256: str
    edited_partitions: tuple[str, ...]
    stock_partitions: tuple[str, ...]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    model: tuple[str, str]
    source: tuple[str, str]
    hidden_size: int
    input_size: int
    layers: tuple[int, ...]
    tp_world_size: int
    payload_file: str
    payload_size: int
    payload_sha256: str
    header_size: int
    header_sha256: str
    tensors: tuple[TensorSpec, ...]
    raw_sha256: str
    digest: str

    def tensor(self, name):
        for item in self.tensors:
            if item.name == name:
                return item
        raise ValueError("Tensor is not part of the native profile")


def load_spec(path: Path, *, expected_sha256: str) -> ProfileSpec:
    """Read a pinned small manifest; its payload is validated separately."""
    expected_sha256 = _sha(expected_sha256)
    fd = _open(path)
    try:
        before = _identity(fd)
        if not 0 < before[2] <= 1024 * 1024:
            raise ValueError("Native manifest size out of bounds")
        raw = os.pread(fd, before[2], 0)
        if (_identity(fd) != before or len(raw) != before[2]
                or hashlib.sha256(raw).hexdigest() != expected_sha256):
            raise ValueError("Native manifest changed or digest mismatched")
    finally:
        os.close(fd)
    x = _json(raw)
    _keys(x, ("format", "name", "model", "source", "hidden_size", "input_size",
              "layers", "tp_world_size", "transform", "requires", "strengths",
              "payload", "tensors"))
    if (x["format"] != FORMAT or x["transform"] != "native_matrix_override_v1"
            or x["requires"] != list(REQUIRES) or x["strengths"] != [1.0]
            or type(x["strengths"][0]) not in (float, int)
            or not isinstance(x["name"], str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", x["name"]) is None
            or x["name"] == "off"):
        raise ValueError("Unsupported native artifact contract")
    hidden, width, world = x["hidden_size"], x["input_size"], x["tp_world_size"]
    if (any(type(n) is not int or n <= 0 for n in (hidden, width, world))
            or hidden % 32 or width % (32 * world)):
        raise ValueError("Native dimensions must align with block32 and TP")
    layers = x["layers"]
    if (not isinstance(layers, list) or not layers
            or any(type(n) is not int or n < 0 for n in layers)
            or layers != sorted(set(layers))):
        raise ValueError("Layers must be unique and sorted")
    payload = x["payload"]
    _keys(payload, ("file", "size_bytes", "sha256", "header_size_bytes", "header_sha256"))
    name = payload["file"]
    if (not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.safetensors", name) is None):
        raise ValueError("Payload must use one plain filename")
    size, header_size = payload["size_bytes"], payload["header_size_bytes"]
    if (type(size) is not int or type(header_size) is not int
            or not 2 <= header_size <= 1024 * 1024 or not header_size + 8 < size <= 2**31):
        raise ValueError("Native payload/header size out of bounds")
    _sha(payload["sha256"]); _sha(payload["header_sha256"])
    tensors = x["tensors"]
    expected_names = {f"layers.{layer}.attn.wo_b.{suffix}"
                      for layer in layers for suffix in ("weight", "scale")}
    if not isinstance(tensors, dict) or set(tensors) != expected_names:
        raise ValueError("Exact native weight/scale pairs required")
    entries, intervals = [], []
    for name, item in tensors.items():
        _keys(item, ("dtype", "shape", "data_offsets", "stock_sha256", "edited_sha256",
                     "stock_partitions", "edited_partitions"))
        scale = name.endswith(".scale")
        shape = [hidden // 32, width // 32] if scale else [hidden, width]
        if (item["dtype"] != ("F8_E8M0" if scale else "F8_E4M3")
                or item["shape"] != shape or not isinstance(item["shape"], list)
                or any(type(value) is not int for value in item["shape"])):
            raise ValueError("Native tensor encoding or shape mismatch")
        offsets = item["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] < offsets[1] <= size - header_size - 8
                or offsets[1] - offsets[0] != math.prod(shape)):
            raise ValueError("Native tensor range mismatch")
        partitions = []
        for field in ("edited_partitions", "stock_partitions"):
            values = item[field]
            if not isinstance(values, list) or len(values) != world:
                raise ValueError("All TP partition hashes required")
            partitions.append(tuple(_sha(value) for value in values))
        entries.append(TensorSpec(name, item["dtype"], tuple(shape), tuple(offsets),
                                  _sha(item["edited_sha256"]), _sha(item["stock_sha256"]),
                                  partitions[0], partitions[1]))
        intervals.append(tuple(offsets))
    cursor = 0
    for start, stop in sorted(intervals):
        if start != cursor:
            raise ValueError("Native tensor payload overlaps or has gaps")
        cursor = stop
    if cursor != size - header_size - 8:
        raise ValueError("Unaccounted native payload bytes")
    digest = hashlib.sha256(FORMAT.encode() + b"\0" + raw
                            + bytes.fromhex(payload["sha256"])).hexdigest()
    return ProfileSpec(x["name"], _pin(x["model"]), _pin(x["source"]), hidden, width,
                       tuple(layers), world, payload["file"], size, payload["sha256"],
                       header_size, payload["header_sha256"], tuple(entries),
                       expected_sha256, digest)


class NativePayload:
    """Full file is streamed once; subsequent rank reads use the same bound fd."""
    def __init__(self, directory: Path, spec: ProfileSpec):
        self.spec = spec
        self.fd = _open(Path(directory) / spec.payload_file)
        try:
            self.identity = _identity(self.fd)
            if self.identity[2] != spec.payload_size:
                raise ValueError("Native payload length mismatch")
            prefix = os.pread(self.fd, 8, 0)
            if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != spec.header_size:
                raise ValueError("Native header length mismatch")
            header = os.pread(self.fd, spec.header_size, 8)
            if hashlib.sha256(header).hexdigest() != spec.header_sha256:
                raise ValueError("Native header hash mismatch")
            expected = {t.name: {"dtype": t.dtype, "shape": list(t.shape),
                                 "data_offsets": list(t.offsets)} for t in spec.tensors}
            actual = _json(header)
            if (actual != expected or
                    json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True)):
                raise ValueError("Native header differs from pinned tensor set")
            full = hashlib.sha256()
            tensor_hashes = {t.name: hashlib.sha256() for t in spec.tensors}
            ordered = sorted(spec.tensors, key=lambda t: t.offsets)
            cursor = 0
            full.update(prefix); full.update(header)
            data_offset = 8 + spec.header_size
            for t in ordered:
                remaining = t.offsets[1] - t.offsets[0]
                while remaining:
                    block = os.pread(self.fd, min(8 * 1024 * 1024, remaining), data_offset + cursor)
                    if not block:
                        raise ValueError("Native payload truncated during validation")
                    if b"\xff" in block or (t.dtype == "F8_E4M3" and b"\x7f" in block):
                        raise ValueError("Native payload contains a non-finite encoding")
                    full.update(block); tensor_hashes[t.name].update(block)
                    cursor += len(block); remaining -= len(block)
                if tensor_hashes[t.name].hexdigest() != t.edited_sha256:
                    raise ValueError("Native tensor hash mismatch")
            if full.hexdigest() != spec.payload_sha256 or _identity(self.fd) != self.identity:
                raise ValueError("Native payload changed or hash mismatched")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def partition(self, name: str, rank: int) -> bytes:
        if self.fd is None or _identity(self.fd) != self.identity:
            raise ValueError("Native payload closed or changed")
        if type(rank) is not int or not 0 <= rank < self.spec.tp_world_size:
            raise ValueError("Invalid TP rank")
        t = self.spec.tensor(name)
        rows, columns = t.shape
        width = columns // self.spec.tp_world_size
        output = bytearray(rows * width)
        for row in range(rows):
            offset = 8 + self.spec.header_size + t.offsets[0] + row * columns + rank * width
            block = os.pread(self.fd, width, offset)
            if len(block) != width:
                raise ValueError("Native partition truncated")
            output[row * width:(row + 1) * width] = block
        if (_identity(self.fd) != self.identity
                or hashlib.sha256(output).hexdigest() != t.edited_partitions[rank]):
            raise ValueError("Native partition changed or hash mismatched")
        return bytes(output)

    def verify_partition(self, name: str, rank: int):
        """Hash a strided TP slice without allocating the whole partition."""
        if self.fd is None or _identity(self.fd) != self.identity:
            raise ValueError("Native payload closed or changed")
        if type(rank) is not int or not 0 <= rank < self.spec.tp_world_size:
            raise ValueError("Invalid TP rank")
        tensor = self.spec.tensor(name)
        rows, columns = tensor.shape
        width = columns // self.spec.tp_world_size
        digest = hashlib.sha256()
        for row in range(rows):
            offset = 8 + self.spec.header_size + tensor.offsets[0] + row * columns + rank * width
            remaining = width
            while remaining:
                chunk = os.pread(self.fd, min(8 * 1024 * 1024, remaining), offset)
                if not chunk:
                    raise ValueError("Native partition truncated")
                digest.update(chunk)
                offset += len(chunk)
                remaining -= len(chunk)
        if (_identity(self.fd) != self.identity
                or digest.hexdigest() != tensor.edited_partitions[rank]):
            raise ValueError("Native partition changed or hash mismatched")

    def verify_stock_partition(self, name: str, rank: int, raw: bytes):
        if type(rank) is not int or not 0 <= rank < self.spec.tp_world_size:
            raise ValueError("Invalid TP rank")
        t = self.spec.tensor(name)
        if (len(raw) != math.prod(t.shape) // self.spec.tp_world_size
                or hashlib.sha256(raw).hexdigest() != t.stock_partitions[rank]):
            raise ValueError("Serving stock matrix differs from native profile base")
