# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Read selected native stock ranges against prior full-file integrity receipts."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct


def _read_json(path):
    return json.loads(Path(path).read_bytes())


def _stat(fd):
    s = os.fstat(fd)
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("Stock must be a regular file")
    return {"device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


class StockRanges:
    def __init__(self, root, manifest_path, manifest_sha256, receipts):
        self.root, self.receipts = Path(root), Path(receipts)
        raw = Path(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError("Stock manifest digest mismatch")
        self.manifest = json.loads(raw)["stock"]
        index_raw = (self.root / "model.safetensors.index.json").read_bytes()
        if hashlib.sha256(index_raw).hexdigest() != self.manifest["index_sha256"]:
            raise ValueError("Stock index digest mismatch")
        self.index = json.loads(index_raw)["weight_map"]
        self.shards = {s["file"]: s for s in self.manifest["shards"]}
        self.handles = {}
        self.reads = []

    def close(self):
        for fd, *_ in self.handles.values(): os.close(fd)
        self.handles.clear()

    def _tensor(self, name):
        filename = self.index[name]
        if filename not in self.handles:
            spec = self.shards[filename]
            receipt_path = self.receipts / (filename + ".json")
            receipt_raw = receipt_path.read_bytes()
            receipt = json.loads(receipt_raw)
            if (receipt.get("pass") is not True or receipt.get("sha256_matches_hf_lfs") is not True
                    or receipt.get("file_identity_unchanged") is not True
                    or receipt.get("full_payload_sha256_computed") is not True
                    or receipt.get("sha256") != receipt.get("expected_sha256")):
                raise ValueError("Stock lacks a successful full-file integrity receipt")
            fd = os.open(self.root / filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                identity = _stat(fd)
                expected = {k: receipt["after_fstat"][k] for k in identity}
                if identity != expected or identity["size"] != spec["size_bytes"]:
                    raise ValueError("Stock changed since its full-file verification")
                header_size = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                if not 2 <= header_size <= min(100_000_000, identity["size"]-8):
                    raise ValueError("Invalid stock header size")
                header_raw = os.pread(fd, header_size, 8)
                if hashlib.sha256(header_raw).hexdigest() != spec["header_sha256"]:
                    raise ValueError("Stock header digest mismatch")
                header = json.loads(header_raw)
                if _stat(fd) != identity:
                    raise ValueError("Stock changed while reading its header")
            except BaseException:
                os.close(fd); raise
            self.handles[filename] = (fd, identity, header, 8+header_size,
                                      hashlib.sha256(receipt_raw).hexdigest())
        fd, identity, header, base, receipt_sha = self.handles[filename]
        if _stat(fd) != identity:
            raise ValueError("Stock changed after verified open")
        tensor = header[name]
        if (tensor["dtype"] not in ("F8_E4M3", "F8_E8M0")
                or len(tensor["shape"]) != 2
                or tensor["data_offsets"][1] - tensor["data_offsets"][0]
                != tensor["shape"][0]*tensor["shape"][1]):
            raise ValueError("Expected native one-byte matrix")
        return fd, identity, tensor, base, receipt_sha, filename

    def partition(self, name, rank, world, axis):
        fd, identity, t, base, receipt_sha, filename = self._tensor(name)
        rows, cols = t["shape"]
        if (axis not in (0, 1) or type(rank) is not int or type(world) is not int
                or not 0 <= rank < world or t["shape"][axis] % world):
            raise ValueError("Invalid stock partition")
        offset = base + t["data_offsets"][0]
        if axis == 0:
            shape = (rows//world, cols)
            raw = os.pread(fd, shape[0]*shape[1], offset+rank*shape[0]*cols)
        else:
            shape = (rows, cols//world)
            raw = b"".join(os.pread(fd, shape[1], offset+r*cols+rank*shape[1])
                           for r in range(rows))
        if len(raw) != shape[0]*shape[1] or _stat(fd) != identity:
            raise ValueError("Stock range changed or truncated")
        self.reads.append({"tensor": name, "rank": rank, "world": world, "axis": axis,
                           "shape": list(shape), "dtype": t["dtype"], "bytes": len(raw),
                           "range_sha256": hashlib.sha256(raw).hexdigest(),
                           "full_file_receipt_sha256": receipt_sha, "shard": filename,
                           "source_identity_matches_full_hash_receipt": True})
        return raw, shape, t["dtype"]
