# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Native matrix profile contract, separate from output projections."""
import json
import os
from pathlib import Path
import re
import stat

from flashalign_native_wo.loader import FORMAT, NativePayload, load_spec


PINNED_MODEL = ("deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "1bae0e1088993b9b82de08f59de5be5a34e42170")


def manifest_hash(directory):
    """Read a bounded local checksum record without following file symlinks."""
    path = Path(directory) / 'sha256.json'
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024:
            raise ValueError('Invalid native profile checksum record')
        raw = os.pread(fd, before.st_size, 0)
        after = os.fstat(fd)
        if len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key)
                for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('Native profile checksum record changed while reading')
    finally:
        os.close(fd)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate native checksum entry')
            result[key] = value
        return result
    values = json.loads(raw, object_pairs_hook=unique)
    value = values.get('profile.json') if isinstance(values, dict) else None
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Native profile manifest checksum is required')
    return value, values


class NativeProfile:
    """Small identity object; workers validate and pack the native payload."""

    is_native_wo = True

    def __init__(self, directory, allow_research=False):
        self.directory = Path(directory).resolve(strict=True)
        expected, hashes = manifest_hash(self.directory)
        self.spec = load_spec(self.directory / "profile.json",
                              expected_sha256=expected)
        spec = self.spec
        if (spec.model != PINNED_MODEL
                or spec.hidden_size != 5120 or spec.input_size != 8192
                or spec.tp_world_size != 4 or any(layer not in range(40) for layer in spec.layers)):
            raise ValueError("Unsupported native model or tensor geometry")
        if hashes.get(spec.payload_file) != spec.payload_sha256:
            raise ValueError('Native payload checksum record differs from its manifest')
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", spec.name)
                or spec.name == "off"):
            raise ValueError("Invalid native profile name")
        self.name, self.digest = spec.name, spec.digest
        self.meta = {
            "format": FORMAT,
            "transform": "native_matrix_override_v1",
            "model": dict(zip(("id", "revision"), spec.model)),
            "source": dict(zip(("id", "revision"), spec.source)),
            "status": "experimental_native_profile",
            "strengths": [1.0],
        }
        self.sites = {(layer, "attn_out"): f"L{layer}.native_wo_b" for layer in spec.layers}
        # This is deliberately empty: native execution never enters projection
        # arithmetic or pretends that a native matrix is a direction vector.
        self.operators = {}

    def validate_payload(self):
        with NativePayload(self.directory, self.spec):
            pass


def validate_native_selection(profile, scale):
    if getattr(profile, "is_native_wo", False) and (
            type(scale) not in (int, float) or scale != 1.0):
        raise ValueError("Native matrix profiles require owner_mode_scale exactly 1.0")
