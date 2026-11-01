# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Opt-in, bounded execution receipts. Never stores request IDs or row arrays."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import threading
import time


class ExecutionDiagnostics:
    def __init__(self, directory, profile_digest, *, mixed_attempts=8):
        if re.fullmatch(r"[0-9a-f]{64}", profile_digest) is None:
            raise ValueError("Invalid diagnostic profile digest")
        if type(mixed_attempts) is not int or not 1 <= mixed_attempts <= 8:
            raise ValueError("Mixed diagnostic attempts must be bounded at eight")
        self.directory = None if not directory else Path(directory)
        if self.directory is not None:
            if not self.directory.is_absolute():
                raise ValueError("Diagnostic directory must be absolute")
            info = self.directory.lstat()
            if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                    or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
                raise ValueError("Diagnostic directory must be precreated, owned and private")
        self.profile_digest = profile_digest
        self.mixed_attempts = mixed_attempts
        self.states = {}
        self.lock = threading.Lock()

    @staticmethod
    def _key(case, graph_mode):
        if isinstance(case, bool) or not isinstance(case, int) or case not in (0, 1, 2):
            raise ValueError("Unknown native diagnostic class")
        if type(graph_mode) is not str or graph_mode not in ("FULL", "PIECEWISE", "NONE"):
            raise ValueError("Unknown native diagnostic graph mode")
        return int(case), graph_mode

    def pending(self, case, graph_mode):
        if self.directory is None:
            return False
        record = self.states.get(self._key(case, graph_mode))
        return record is None or not record["finished"]

    def record(self, case, digest, count_reader, *, graph_mode, requests, max_rows):
        if self.directory is None:
            return None
        key = self._key(case, graph_mode)
        if digest != self.profile_digest:
            raise ValueError("Diagnostic profile digest mismatch")
        if (type(requests) is not int or requests <= 0
                or type(max_rows) is not int or max_rows <= 0):
            raise ValueError("Invalid diagnostic batch bounds")
        with self.lock:
            if not self.pending(case, graph_mode):
                return None
            counts = count_reader()
            expected = {"live_tokens", "owner_tokens", "off_tokens", "invalid_flags"}
            if (not isinstance(counts, dict) or set(counts) != expected
                    or any(type(x) is not int or x < 0 for x in counts.values())
                    or not 0 < counts["live_tokens"] <= max_rows
                    or counts["owner_tokens"] + counts["off_tokens"] != counts["live_tokens"]
                    or counts["invalid_flags"] != 0):
                raise ValueError("Final GPU ownership counts violate their bounds")
            if case == 0 and counts["owner_tokens"]:
                raise ValueError("OFF execution contains active owner rows")
            if case == 1 and counts["off_tokens"]:
                raise ValueError("OWNER execution contains active off rows")
            actual_mixed = counts["owner_tokens"] > 0 and counts["off_tokens"] > 0
            names = ("off", "owner", "mixed")
            record = self.states.setdefault(key, {
                "schema": 2, "pid": os.getpid(), "profile_digest": digest,
                "execution_case": names[case], "execution_case_value": int(case),
                "graph_mode": graph_mode,
                "source": "final_gpu_query_boundaries_and_owner_buffer_after_forward",
                "observations": [], "finished": False, "complete": False,
                "actual_mixed_rows": False, "attempts_exhausted": False,
                "scope": "execution class/count evidence only; no output parity proof",
            })
            record["observations"].append({"observed_unix": time.time(), "requests": requests,
                                            **counts, "actual_mixed_rows": actual_mixed})
            record["actual_mixed_rows"] = actual_mixed
            record["complete"] = case != 2 or actual_mixed
            record["attempts_exhausted"] = case == 2 and not actual_mixed and (
                len(record["observations"]) >= self.mixed_attempts)
            record["finished"] = record["complete"] or record["attempts_exhausted"]
            self._write(names[case], graph_mode, record)
            return record.copy()

    def _write(self, name, graph_mode, record):
        directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        filename = f"native-execution-{os.getpid()}-{graph_mode.lower()}-{name}.json"
        temporary = f".{filename}.{threading.get_ident()}.{time.time_ns()}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(record, stream, indent=2, allow_nan=False)
                    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, filename, src_dir_fd=directory, dst_dir_fd=directory)
                os.fsync(directory)
            except BaseException:
                try: os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError: pass
                raise
        finally:
            os.close(directory)
