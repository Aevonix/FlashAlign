# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Engine-agnostic request authentication. Never log key or header values."""
import hashlib
import hmac
import fcntl
import os
from pathlib import Path
import secrets
import stat
import tempfile
import time


def ensure_key(path):
    """Publish a complete private key atomically; concurrent setup reuses it."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = os.open(str(path)+'.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    temporary = None
    try:
        info = os.fstat(lock)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ValueError('invalid key creation lock')
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.lexists(path):
            read_key(path)
            return False
        fd, temporary = tempfile.mkstemp(prefix='.flashalign-key-', dir=path.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(secrets.token_bytes(32))
            stream.flush()
            os.fsync(stream.fileno())
        # Publication must also preserve a key created by a noncooperating
        # writer after the initial check. A hard link is atomic and never
        # replaces an existing destination.
        try:
            os.link(temporary, path)
        except FileExistsError:
            read_key(path)
            return False
        os.unlink(temporary)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    finally:
        if temporary is not None:
            os.unlink(temporary)
        os.close(lock)


def read_key(path):
    fd = os.open(os.path.expanduser(str(path)), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise ValueError("invalid key file permissions")
        key = os.read(fd, 33)
        if len(key) != 32:
            raise ValueError("invalid key size")
        return key
    finally:
        os.close(fd)


def read_api_key(path):
    """Read an optional upstream token with the same file protections as signing keys."""
    fd = os.open(os.path.expanduser(str(path)), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise ValueError('invalid API key file permissions')
        value = os.read(fd, 8193).decode('utf-8')
        if not value or len(value) > 8192 or '\r' in value or '\n' in value:
            raise ValueError('invalid API key')
        return value
    finally:
        os.close(fd)


def sign(key, body, *, now=None, nonce=None):
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("invalid key")
    ts = str(int(time.time() if now is None else now))
    nonce = nonce or secrets.token_hex(24)
    digest = hashlib.sha256(body).hexdigest()
    mac = hmac.new(key, (ts + "\n" + nonce + "\n" + digest).encode(), hashlib.sha256).hexdigest()
    return "v1:" + ts + ":" + nonce + ":" + mac
