# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Bounded input and private, crash-safe files for evaluation artifacts."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat


MAX_JSON_BYTES = 16 * 1024 * 1024


class EvaluationBusyError(ValueError):
    """Another process currently holds the evaluation's exclusive lock."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def loads(value):
    def invalid(_):
        raise ValueError("Non-finite JSON number")
    return json.loads(value, object_pairs_hook=_unique, parse_constant=invalid)


def read_bytes(path, limit=MAX_JSON_BYTES):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("Input must be a bounded regular file")
        chunks = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        if len(data) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Input changed during the read")
        return data
    finally:
        os.close(fd)


def read_json(path, *, limit=MAX_JSON_BYTES):
    return loads(read_bytes(path, limit=limit))


def read_jsonl(path):
    return [loads(line) for line in read_bytes(path).splitlines() if line.strip()]


def private_directory(path, *, create=False):
    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink():
        raise ValueError("Private output must not be a symlink")
    path = supplied.resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Evaluation outputs must stay outside Git checkouts")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        path.chmod(0o700)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Evaluation directory must be owned and private")
    return path


def atomic_bytes(path, data, *, replace=True):
    """Publish complete bytes only; reject special-file and foreign destinations."""
    path = Path(path)
    parent = path.parent
    info = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Output parent must be an owned private directory")
    if path.exists() or path.is_symlink():
        prior = path.stat(follow_symlinks=False)
        if not replace or not stat.S_ISREG(prior.st_mode) or prior.st_uid != os.getuid() or prior.st_nlink != 1:
            raise ValueError("Unsafe or existing evaluation output")
    temporary = parent / ("." + path.name + "." + secrets.token_hex(12))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value, *, replace=True):
    atomic_bytes(path, (canonical(value) + "\n").encode(), replace=replace)


@contextmanager
def run_lock(directory):
    path = Path(directory) / ".run.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("Unsafe evaluation lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvaluationBusyError("Another process owns this evaluation") from None
        yield
    finally:
        os.close(fd)
