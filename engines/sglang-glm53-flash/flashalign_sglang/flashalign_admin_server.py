# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Local HTTP administration transport for engine lifecycle adapters.

The socket and parent directory are private to their owner. Linux SO_PEERCRED
is checked before parsing requests. Code running as that UID is trusted;
generation-signing keys never authorize this control plane.
"""
import asyncio
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import sys


def runtime_counters(engine):
    """Read existing in-process metrics without importing a metrics dependency."""
    names = {
        "output_tokens_total": engine + ":generation_tokens_total",
        "ttft_seconds_sum": engine + ":time_to_first_token_seconds_sum",
        "ttft_seconds_count": engine + ":time_to_first_token_seconds_count",
    }
    unavailable = dict.fromkeys(names)
    metrics = sys.modules.get("prometheus_client")
    registry = getattr(metrics, "REGISTRY", None)
    if registry is None:
        return unavailable
    samples = {name: [] for name in names.values()}
    try:
        for family in registry.collect():
            for sample in family.samples:
                if sample.name in samples:
                    samples[sample.name].append(sample.value)
        result = {}
        for field, name in names.items():
            values = samples[name]
            if (not values or any(type(value) not in (int, float)
                    or not math.isfinite(value) or value < 0 for value in values)):
                result[field] = None
                continue
            total = sum(values)
            if not math.isfinite(total) or (field != "ttft_seconds_sum"
                    and any(value != int(value) for value in values)):
                result[field] = None
            else:
                result[field] = float(total) if field == "ttft_seconds_sum" else int(total)
        if result["ttft_seconds_sum"] is None or result["ttft_seconds_count"] is None:
            result["ttft_seconds_sum"] = result["ttft_seconds_count"] = None
        return result
    except Exception:
        # Optional telemetry must not prevent status or profile administration.
        return unavailable


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate operation field")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Non-finite operation value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def dispatch_operation(manager, permit, request, *, evict_cache=None):
    """Canonical admin schema, shared by every engine-specific coordinator."""
    if not isinstance(request, dict):
        raise ValueError("Operation must be an object")
    fields = {"stage": {"path"}, "activate": {"alias", "generation", "expected_revision"},
              "rollback": {"alias", "generation", "expected_revision"}, "retire": {"generation"},
              "deactivate": {"alias", "expected_revision"}, "recover": set(), "evict_cache": set()}
    action = request.get("action")
    if not isinstance(action, str) or action not in fields or set(request) != fields[action] | {"action"}:
        raise ValueError("Invalid lifecycle operation fields")
    for key in ("alias", "generation"):
        if key in request and (not isinstance(request[key], str) or not request[key] or len(request[key]) > 256):
            raise ValueError("Invalid lifecycle identifier")
    if "expected_revision" in request and (type(request["expected_revision"]) is not int or request["expected_revision"] < 0):
        raise ValueError("Invalid expected alias revision")
    if action == "stage":
        if not isinstance(request["path"], str) or not request["path"] or len(request["path"]) > 4096:
            raise ValueError("Invalid local profile path")
        result = manager.stage(permit, request["path"])
        return {"generation": result.id, "artifact": result.artifact.record()}
    if action in {"activate", "rollback"}:
        return getattr(manager, action)(permit, request["alias"], request["generation"],
                                        expected_revision=request["expected_revision"])
    if action == "deactivate":
        method = getattr(manager, "deactivate", None)
        if method is None:
            raise ValueError("Alias deactivation is unavailable in this coordinator version")
        return method(permit, request["alias"], expected_revision=request["expected_revision"])
    if action == "retire":
        manager.retire(permit, request["generation"])
    elif action == "recover":
        manager.recover(permit)
    elif action == "evict_cache":
        if evict_cache is None:
            raise ValueError("Cache eviction is unavailable in this adapter")
        evict_cache()
    return manager.status()


class LocalAdminServer:
    def __init__(self, address, *, status, operation, conflict_errors=()):
        self.address = Path(address).absolute()
        self.status = status
        self.operation = operation
        self.conflict_errors = tuple(conflict_errors)
        self.server = None
        self._identity = None
        self._slots = asyncio.Semaphore(8)
        self._tasks = set()

    async def _accepted(self, reader, writer):
        task = asyncio.current_task()
        self._tasks.add(task)
        try:
            await self._serve(reader, writer)
        finally:
            self._tasks.discard(task)

    async def _serve(self, reader, writer):
        async with self._slots:
            status = 400
            result = {"ok": False, "error": {"code": "InvalidRequest", "message": "Invalid lifecycle request"}}
            try:
                peer = writer.get_extra_info("socket").getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                _, uid, _ = struct.unpack("3i", peer)
                if uid != os.getuid():
                    status = 403
                    raise PermissionError()
                head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
                if len(head) > 8192:
                    raise ValueError()
                lines = head.decode("ascii").split("\r\n")
                method, path, version = lines[0].split(" ")
                if version != "HTTP/1.1":
                    raise ValueError()
                headers = {}
                for line in lines[1:-2]:
                    key, value = line.split(":", 1)
                    key = key.lower()
                    if key in headers:
                        raise ValueError()
                    headers[key] = value.strip()
                if "transfer-encoding" in headers:
                    raise ValueError()
                length = int(headers.get("content-length", "0"))
                if not 0 <= length <= 65536:
                    raise ValueError()
                if method == "GET" and path == "/status" and length == 0:
                    value = await asyncio.to_thread(self.status)
                elif method == "POST" and path == "/operations" and length:
                    body = await asyncio.wait_for(reader.readexactly(length), timeout=10)
                    value = await asyncio.to_thread(self.operation, _json(body))
                else:
                    raise ValueError()
                status, result = 200, {"ok": True, "result": value}
            except self.conflict_errors as error:
                status, result = 409, {"ok": False, "error": {"code": type(error).__name__,
                    "message": "Lifecycle operation did not complete; inspect status before retrying"}}
            except Exception as error:
                # Never return exception text; adapters may mention paths or keys.
                result["error"]["code"] = type(error).__name__
            finally:
                try:
                    body = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
                    writer.write(f"HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
                    await writer.drain()
                except (ConnectionError, BrokenPipeError):
                    pass
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except (ConnectionError, BrokenPipeError):
                        pass

    async def start(self):
        info = self.address.parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("Administration socket requires an owned mode-0700 parent")
        if self.address.exists() or self.address.is_symlink():
            info = self.address.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("Unsafe administration socket path")
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(self.address))
            except ConnectionRefusedError:
                self.address.unlink()
            else:
                raise ValueError("Administration socket is already in use")
            finally:
                probe.close()
        self.server = await asyncio.start_unix_server(self._accepted, path=str(self.address), limit=8192)
        os.chmod(self.address, 0o600)
        info = self.address.lstat()
        self._identity = (info.st_dev, info.st_ino)
        return self

    async def close(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            if self._tasks:
                await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
            try:
                info = self.address.lstat()
                if self._identity == (info.st_dev, info.st_ino):
                    self.address.unlink()
            except FileNotFoundError:
                pass
