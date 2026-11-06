# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Artifact and protocol checks shared by SGLang lifecycle processes."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_CONTROL = 4 * 1024 * 1024
GENERATION = re.compile(r"g-[0-9a-f]{48}\Z")
TRANSACTION = re.compile(r"[trd]-[0-9a-f]{48}\Z")


def profile_digest(manifest, payload):
    """Canonical compact-artifact identity shared with the public inspector."""
    hashes = {"profile.json": hashlib.sha256(manifest).hexdigest(),
              "profile.safetensors": hashlib.sha256(payload).hexdigest()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strict_json(raw):
    if not isinstance(raw, (str, bytes)) or len(raw) > MAX_CONTROL:
        raise ValueError("Invalid lifecycle message size")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate lifecycle field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Non-finite lifecycle number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def private_directory(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Lifecycle state requires an owned mode-0700 directory")
    return path


def snapshot_file(source, destination, limit):
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("Invalid profile artifact file")
        out = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        try:
            with os.fdopen(out, "wb") as stream:
                total = 0
                while True:
                    block = os.read(fd, 1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > limit:
                        raise ValueError("Profile artifact grew during staging")
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Profile artifact changed during staging")
        except BaseException:
            Path(destination).unlink(missing_ok=True)
            raise
    finally:
        os.close(fd)


def stage_profile(source, destination):
    from . import runtime
    from .flashalign_lifecycle import Artifact
    source, destination = Path(source), private_directory(destination)
    if source.is_symlink() or not source.is_dir() or any(destination.iterdir()):
        raise ValueError("Profile staging needs a local source and empty destination")
    original = runtime.load_profile(source)
    for name, limit in (("profile.json", MAX_CONTROL), ("profile.safetensors", 256 * 1024 * 1024)):
        snapshot_file(source / name, destination / name, limit)
    # Preserve the optional integrity receipt, but never copy executable files.
    if (source / "sha256.json").exists():
        snapshot_file(source / "sha256.json", destination / "sha256.json", MAX_CONTROL)
    checked = runtime.load_profile(destination)
    if checked.digest != original.digest:
        raise ValueError("Profile source changed during staging")
    metadata = strict_json((destination / "profile.json").read_bytes())
    return Artifact(checked.name, checked.digest, checked.model["id"], checked.model["revision"],
                    metadata.get("transform", "projection"), destination,
                    tuple(sorted(metadata.get("requires", []))))


def checked_profile(directory, digest):
    from . import runtime
    profile = runtime.load_profile(directory)
    if profile.digest != digest:
        raise ValueError("Staged profile digest differs from committed identity")
    # The backend's running model contract comes from the profiles validated at
    # startup, not from a caller-supplied model or mutable alias.
    existing = runtime.registry()[0]
    models = getattr(runtime, "_flashalign_model_contracts", None)
    if models is None:
        models = {json.dumps(p.model, sort_keys=True) for p in existing.values()}
        runtime._flashalign_model_contracts = models
    scope = os.environ.get("FLASHALIGN_MODEL_SCOPE")
    compatible = getattr(profile, "compatible_models", ())
    if compatible:
        scopes = set()
        for item in compatible:
            model = {"id": item["id"], "revision": item["revision"], "sha256": {
                "config.json": item["config_sha256"], "model.safetensors.index.json": item["index_sha256"]}}
            scopes.add(hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest())
        if scope not in scopes:
            raise ValueError("Profile is incompatible with the loaded model fingerprints")
    elif json.dumps(profile.model, sort_keys=True) not in models:
        raise ValueError("Profile model differs from the running model contract")
    return profile


def tensor_bytes(profile):
    # Count each site reference. Backends may create one device copy per writer
    # even when the serialized operator shares a CPU tensor between writers.
    size = 0
    for operator in profile.sites.values():
        for value in operator.values():
            if hasattr(value, "numel"):
                size += value.numel() * value.element_size()
    return size


def recovery_generations(snapshot):
    """Committed resources plus durable commit decisions, excluding discards."""
    pending = snapshot.get("pending")
    discard = set()
    if pending is not None:
        if pending["operation"] == "retire":
            discard.add(pending["generation"])
        elif not pending["was_allocated"] and pending["phase"] in {"preparing", "fencing", "aborting"}:
            discard.add(pending["generation"])
    return {gid: row for gid, row in snapshot["generations"].items()
            if row["allocated"] and row["state"] not in {"retired", "retiring"} and gid not in discard}
