# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Content-free decision logging shared by local integrations."""
import json
import logging
import os
from pathlib import Path
import threading
import time


logger = logging.getLogger(__name__)


class AuditLog:
    """One JSON line per model call. Callers pass fixed fields only: no prompt text, key or header value."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row):
        line = json.dumps(dict({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, **row), sort_keys=True) + "\n"
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)


def warn_fallback(audit, decision):
    """Report a fixed fallback code without request data or exception text."""
    try:
        logger.warning("FlashAlign fallback: %s", decision)
    except Exception:
        pass
    if audit is not None:
        try:
            audit({"decision": decision})
        except Exception:
            pass
