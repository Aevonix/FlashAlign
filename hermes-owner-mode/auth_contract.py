# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Shared wire parsing. Replay consumption remains an atomic backend operation."""
from dataclasses import dataclass
import hashlib
import hmac
import math
import os
import re
import stat
import time


HEADER = re.compile(r'v1:([0-9]{1,12}):([A-Za-z0-9_-]{16,128}):([0-9a-f]{64})\Z')


@dataclass(frozen=True)
class AuthenticatedHeader:
    timestamp: int
    nonce: str
    expires_at: int


def authenticate_header(key, body, headers, *, now=None):
    """Return a verified header before replay storage, or None for any bad input."""
    if not isinstance(key, bytes) or len(key) != 32 or not isinstance(body, bytes):
        return None
    if not isinstance(headers, (list, tuple)) or len(headers) != 1 or not isinstance(headers[0], str):
        return None
    match = HEADER.fullmatch(headers[0])
    if match is None:
        return None
    current = time.time() if now is None else now
    if type(current) not in (int, float) or not math.isfinite(current):
        return None
    stamp, nonce, supplied = match.groups()
    timestamp = int(stamp)
    if abs(current - timestamp) > 60:
        return None
    message = stamp + '\n' + nonce + '\n' + hashlib.sha256(body).hexdigest()
    expected = hmac.new(key, message.encode('ascii'), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, supplied):
        return None
    return AuthenticatedHeader(timestamp, nonce, timestamp + 61)


def read_signing_key(path):
    """Open an owned, single-link 0600 regular file without following a symlink."""
    fd = os.open(os.path.expanduser(str(path)), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise ValueError('Signing key must be an owned mode-0600 regular file with one link')
        value = os.read(fd, 33)
        if len(value) != 32:
            raise ValueError('Signing key must contain exactly 32 bytes')
        return value
    finally:
        os.close(fd)
