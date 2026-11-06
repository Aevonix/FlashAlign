# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Start an owned SSH Unix-socket forward, refusing an active existing listener."""
import os
from pathlib import Path
import socket
import stat
import sys


def prepare(path):
    path = Path(path)
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700:
        raise ValueError('Administration forward requires a private owned directory')
    try:
        before = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(before.st_mode) or before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600:
        raise ValueError('Administration forward path is unsafe')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(.5)
        try:
            connection.connect(str(path))
        except ConnectionRefusedError:
            after = path.lstat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError('Administration socket changed during inspection')
            path.unlink()
        else:
            raise ValueError('Administration forward is already active')


if __name__ == '__main__':
    prepare(sys.argv[1])
    os.umask(0o177)
    os.execvp(sys.argv[2], sys.argv[2:])
