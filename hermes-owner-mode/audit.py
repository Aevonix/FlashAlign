# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Content-free decision logging shared by local integrations."""
import argparse
from collections import deque
from datetime import datetime
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


DECISION_FIELDS = frozenset({
    "ts", "route", "platform", "owner_verified", "activation_mode", "decision",
    "identity_ms", "identity_reason", "owner_mode_sent", "owner_mode_scale_sent",
    "effective", "status", "ms",
})


def recent_decisions(paths, limit=10):
    """Read only decision fields, tolerating partial or malformed log lines."""
    rows = []
    for path in dict.fromkeys(Path(path).expanduser() for path in paths):
        recent = deque(maxlen=limit)
        try:
            with path.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    try:
                        row = json.loads(line)
                        if (not isinstance(row, dict) or not isinstance(row.get("decision"), str)
                                or not isinstance(row.get("ts"), str)):
                            continue
                        time_format = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in row["ts"] else "%Y-%m-%dT%H:%M:%S%z"
                        timestamp = datetime.strptime(row["ts"], time_format).timestamp()
                        selected = {key: value for key, value in row.items() if key in DECISION_FIELDS
                                    and (value is None or type(value) in (str, int, float, bool))}
                        recent.append((timestamp, selected))
                    except (ValueError, TypeError, KeyError, OverflowError):
                        continue
        except OSError:
            pass
        rows.extend(recent)
    return [row for _, row in sorted(rows, key=lambda item: item[0])[-limit:]]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Show recent FlashAlign decisions without message content")
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--log", type=Path, action="append", help="Read an explicit audit log (repeatable)")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    paths = args.log
    if paths is None:
        paths = [args.home / ".hermes/logs/owner-mode.log",
                 args.home / ".config/owner-mode/owner-mode.log",
                 *sorted((args.home / ".local/share/flashalign/gateways").glob("*/owner-mode.log"))]
    for row in recent_decisions(paths, args.limit):
        print(json.dumps(row, sort_keys=True))
    return 0
