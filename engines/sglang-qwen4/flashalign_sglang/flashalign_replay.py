# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Persistent replay protection after successful FlashAlign HMAC validation.

Stores only key-derived namespace/nonce digests and expiry times. This module
uses the Python standard library and contains no profile or model logic.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time


class ReplayStoreError(RuntimeError):
    """The private replay store cannot safely be used."""


def deployment_replay_store(key: bytes, path=None):
    """Open stable deployment state, creating only absent directories.

    Precedence: explicit path, FLASHALIGN_REPLAY_STORE, then XDG_STATE_HOME
    (or ~/.local/state)/flashalign/replay/nonces.sqlite3. Existing directory
    permissions are never changed; ReplayStore rejects an unsafe leaf.
    """
    configured = path if path is not None else os.environ.get('FLASHALIGN_REPLAY_STORE')
    if configured:
        selected = Path(configured).expanduser().absolute()
    else:
        configured_state = os.environ.get('XDG_STATE_HOME')
        state = Path(configured_state).expanduser() if configured_state else None
        if state is None or not state.is_absolute():
            state = Path.home()/'.local'/'state'
        selected = state/'flashalign'/'replay'/'nonces.sqlite3'
    selected.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return ReplayStore(selected, key)


class ReplayStore:
    """Atomically consume authenticated nonces across processes and restarts.

    The caller must verify the request HMAC and the +/-60 second timestamp
    window first. Pass ``expires_at=authenticated_timestamp+61``. A false
    result always means that the caller must not enable owner mode.

    The parent directory must already be owned by the current user and have
    no group/other permissions. Keep this path stable while reusing a key.
    """

    def __init__(self, path, key: bytes, *, capacity=100_000, timeout=0.25, clock=time.time):
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError('Replay namespace requires a 32-byte deployment key')
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Replay capacity must be a positive integer')
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('Invalid replay lock timeout')
        self.path = Path(path).absolute()
        self.capacity, self.timeout, self.clock = capacity, timeout, clock
        self.namespace = hmac.digest(key, b'FlashAlign replay namespace v1', 'sha256')
        self._nonce_key = hmac.digest(key, b'FlashAlign replay nonce key v1', 'sha256')
        self._identity = None
        try:
            self._check_private_path(create=True)
            with closing(self._connect()) as connection, connection:
                version = connection.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, 1):
                    raise ReplayStoreError('Unsupported replay schema')
                connection.execute('''CREATE TABLE IF NOT EXISTS replay_nonces (
                    namespace BLOB NOT NULL CHECK(length(namespace)=32),
                    nonce_digest BLOB NOT NULL CHECK(length(nonce_digest)=32),
                    expires_at REAL NOT NULL,
                    PRIMARY KEY(namespace, nonce_digest)
                ) WITHOUT ROWID''')
                connection.execute('CREATE INDEX IF NOT EXISTS replay_expiry ON replay_nonces(expires_at)')
                columns = [(r[1], r[2], r[3], r[5]) for r in connection.execute('PRAGMA table_info(replay_nonces)')]
                if columns != [('namespace', 'BLOB', 1, 1), ('nonce_digest', 'BLOB', 1, 2), ('expires_at', 'REAL', 1, 0)]:
                    raise ReplayStoreError('Invalid replay schema')
                connection.execute('PRAGMA user_version=1')
        except (OSError, sqlite3.Error) as error:
            raise ReplayStoreError('Replay store unavailable') from error

    def _check_private_path(self, *, create=False):
        directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parent = os.fstat(directory_fd)
            if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
                raise ReplayStoreError('Replay directory must be owned and private')
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if create:
                flags |= os.O_CREAT
            descriptor = os.open(self.path.name, flags, 0o600, dir_fd=directory_fd)
            try:
                info = os.fstat(descriptor)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) & 0o077 or info.st_nlink != 1):
                    raise ReplayStoreError('Replay database must be an owned private regular file')
                identity = (info.st_dev, info.st_ino)
                if self._identity is not None and identity != self._identity:
                    raise ReplayStoreError('Replay database was replaced')
                self._identity = identity
            finally:
                os.close(descriptor)
        finally:
            os.close(directory_fd)

    def _connect(self):
        # The checked parent is private, so untrusted users cannot replace the
        # database path or its SQLite rollback journal between these operations.
        self._check_private_path()
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        try:
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute('PRAGMA synchronous=FULL')
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def consume(self, nonce: str, expires_at: float, *, now=None) -> bool:
        """Return true once; return false on replay, full storage or any failure."""
        current = self.clock() if now is None else now
        if (not isinstance(nonce, str) or re.fullmatch(r'[A-Za-z0-9_-]{16,128}', nonce) is None
                or type(expires_at) not in (int, float) or type(current) not in (int, float)
                or not math.isfinite(expires_at) or not math.isfinite(current)
                or expires_at < current or expires_at > current + 121):
            return False
        digest = hmac.digest(self._nonce_key, nonce.encode('ascii'), 'sha256')
        connection = None
        try:
            connection = self._connect()
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('DELETE FROM replay_nonces WHERE expires_at < ?', (current,))
            if connection.execute('SELECT count(*) FROM replay_nonces').fetchone()[0] >= self.capacity:
                connection.rollback()
                return False
            cursor = connection.execute('INSERT OR IGNORE INTO replay_nonces VALUES (?, ?, ?)',
                                        (self.namespace, digest, expires_at))
            accepted = cursor.rowcount == 1
            connection.commit()
            return accepted
        except (OSError, sqlite3.Error, ReplayStoreError):
            if connection is not None:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            return False
        finally:
            if connection is not None:
                connection.close()
