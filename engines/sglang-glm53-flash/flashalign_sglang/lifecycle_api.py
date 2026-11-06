# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""SGLang lifecycle coordinator and owner-UID Unix administration endpoint."""
import asyncio
from concurrent.futures import TimeoutError as WorkerTimeout
import hashlib
import json
import os
from pathlib import Path
import secrets

from .flashalign_lifecycle import LifecycleError, LifecycleManager, RecoveryRequired, WorkerAck
from .flashalign_admin_server import LocalAdminServer, dispatch_operation
from .lifecycle_support import MAX_CONTROL, checked_profile, private_directory, recovery_generations, stage_profile, strict_json, tensor_bytes

_CONTROLLER = None
_SERVER = None
_SOCKET = None
_INSTANCE = secrets.token_hex(16)
_REQUESTS = {"active": 0, "errors": 0}


def request_enter():
    _REQUESTS["active"] += 1


def request_exit(error=False):
    _REQUESTS["active"] -= 1
    if error:
        _REQUESTS["errors"] += 1


def rejected_request():
    _REQUESTS["errors"] += 1


def install_tokenizer_control(tokenizer):
    from sglang.srt.managers.io_struct import FlashAlignControlReqOutput
    from sglang.utils import TypeBasedDispatcher
    tokenizer._flashalign_futures = {}
    tokenizer._flashalign_control_lock = asyncio.Lock()
    def receive(output):
        future = tokenizer._flashalign_futures.get(output.rid)
        if future is not None and not future.done():
            future.set_result(output.payload)
    tokenizer._result_dispatcher += TypeBasedDispatcher([(FlashAlignControlReqOutput, receive)])


async def control(tokenizer, payload, *, timeout=120):
    from sglang.srt.managers.io_struct import FlashAlignControlReqInput
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if len(encoded) > MAX_CONTROL:
        raise LifecycleError("Control message exceeds its limit")
    async with tokenizer._flashalign_control_lock:
        tokenizer.auto_create_handle_loop()
        rid = "flashalign-control-" + secrets.token_hex(24)
        future = asyncio.get_running_loop().create_future()
        tokenizer._flashalign_futures[rid] = future
        try:
            request = FlashAlignControlReqInput(rid=rid, payload=encoded)
            await tokenizer._async_dispatch_to_scheduler(request)
            raw = await asyncio.wait_for(future, timeout=timeout)
            result = strict_json(raw)
            if not isinstance(result, dict) or set(result) != {"workers"}:
                raise LifecycleError("Invalid worker acknowledgement")
            return result["workers"]
        finally:
            tokenizer._flashalign_futures.pop(rid, None)


class Backend:
    def __init__(self, tokenizer, loop):
        from . import runtime
        self.tokenizer, self.loop = tokenizer, loop
        self.profiles, self.gate = runtime.registry()
        self.models = sorted({json.dumps(p.model, sort_keys=True) for p in self.profiles.values()})
        runtime._flashalign_model_contracts = set(self.models)
        identity = hashlib.sha256(json.dumps([self.models, os.getenv("FLASHALIGN_MODEL_SCOPE", "")]).encode()).hexdigest()
        self.identity = "sglang-0.5.20:" + identity
        self.worker_ids = tuple("tp-" + str(i) for i in range(tokenizer.server_args.tp_size))

    def call(self, action, transaction=None, generation=None, snapshot=None):
        payload = {"action": action, "transaction": transaction or {"id": "r-" + secrets.token_hex(24)}}
        if generation is not None:
            payload["generation"] = {"id": generation.id, "digest": generation.artifact.digest}
        if snapshot is not None:
            payload["snapshot"] = snapshot
        timeout = 5 if action == "inspect" else 120
        future = asyncio.run_coroutine_threadsafe(control(self.tokenizer, payload, timeout=timeout), self.loop)
        try:
            rows = future.result(timeout=timeout + 5)
        except WorkerTimeout:
            future.cancel()
            raise LifecycleError("Worker control operation timed out") from None
        if not isinstance(rows, list) or len(rows) != len(self.worker_ids):
            raise LifecycleError("Incomplete worker cohort")
        if {row.get("worker_id") for row in rows} != set(self.worker_ids):
            raise LifecycleError("Worker cohort identity mismatch")
        if any(row.get("ok") is not True for row in rows):
            raise LifecycleError("At least one worker rejected the lifecycle operation")
        return rows

    def acks(self, action, transaction, generation=None, snapshot=None):
        rows = self.call(action, transaction, generation, snapshot)
        return [WorkerAck(row["worker_id"], row["transaction_id"], row.get("generation_id"),
                          row.get("artifact_digest"), row["phase"]) for row in rows]

    def validate(self, artifact):
        checked_profile(artifact.directory, artifact.digest)

    def estimate(self, artifact):
        profile = checked_profile(artifact.directory, artifact.digest)
        size = tensor_bytes(profile)
        return {"host": size + (artifact.directory / "profile.safetensors").stat().st_size,
                **{worker: size * 2 for worker in self.worker_ids}}

    def reserve(self, transaction, generation):
        # Coordinator reserves declared host/per-rank budgets before any worker
        # allocation. Actual allocation failure aborts without alias mutation.
        self.validate(generation.artifact)

    def prepare(self, transaction, generation):
        profile = checked_profile(generation.artifact.directory, generation.artifact.digest)
        self.profiles[generation.id] = profile
        return self.acks("prepare", transaction, generation)

    def fence(self, transaction):
        return self.acks("fence", transaction)

    def commit(self, transaction, generation):
        acks = self.acks("commit", transaction, generation)
        if transaction["operation"] == "deactivate":
            self.profiles.pop(transaction["alias"], None)
        else:
            self.profiles[transaction["alias"]] = self.profiles[generation.id]
        return acks

    def resume(self, transaction):
        return self.acks("resume", transaction)

    def abort(self, transaction, generation):
        acks = self.acks("abort", transaction, generation)
        if not transaction["was_allocated"]:
            self.profiles.pop(generation.id, None)
        return acks

    def references(self, generation):
        return sum(row["references"] for row in self.call("references", generation=generation))

    def retire(self, transaction, generation):
        acks = self.acks("retire", transaction, generation)
        self.profiles.pop(generation.id, None)
        return acks

    def reconcile(self, transaction, snapshot):
        acks = self.acks("reconcile", transaction, snapshot=snapshot)
        profiles = {}
        for gid, row in recovery_generations(snapshot).items():
            profiles[gid] = checked_profile(Path(snapshot["artifact_root"]) / gid, row["artifact"]["digest"])
        for alias, row in snapshot["aliases"].items():
            if row["generation"] is not None:
                profiles[alias] = profiles[row["generation"]]
        self.profiles.clear()
        self.profiles.update(profiles)
        return acks


class Controller:
    def __init__(self, backend):
        self.backend = backend
        self._secret = secrets.token_bytes(32)
        limits = strict_json(os.environ.get("FLASHALIGN_LIFECYCLE_BUDGET", "{}"))
        if not isinstance(limits, dict):
            raise ValueError("Lifecycle budget must be a JSON object")
        limits = {"host": 1024**3, **{worker: 512 * 1024**2 for worker in backend.worker_ids}, **limits}
        self.manager = LifecycleManager(os.environ["FLASHALIGN_LIFECYCLE_STATE"], backend=backend,
                                        stager=stage_profile, admin_key=self._secret, resource_limits=limits)

    def initialize(self):
        self.backend.call("inspect")
        permit = self.manager.admin(self._secret)
        if self.manager.journal_was_present:
            self.manager.recover(permit)
        else:
            for source in os.environ.get("FLASHALIGN_PROFILES", "").split(","):
                if source.strip():
                    generation = self.manager.stage(permit, source.strip())
                    self.manager.activate(permit, generation.artifact.name, generation.id, expected_revision=0)
        self.backend.call("finish_bootstrap")

    def bind(self, selected):
        from .gate import FieldError, Selection
        try:
            lease = self.manager.pin(selected.name)
        except RecoveryRequired:
            raise
        except LifecycleError:
            raise FieldError('owner_mode names no profile this server loaded') from None
        try:
            internal = Selection(lease.generation.id, lease.generation.artifact.digest, selected.scale)
            # Includes operator-specific fixed-strength validation where declared.
            self.backend.gate.from_salt(self.backend.gate.cache_salt(internal, None))
            return internal, lease
        except BaseException:
            lease.close()
            raise

    def operation(self, request):
        # Called only after the Unix peer owner has been checked.
        permit = self.manager.admin(self._secret)
        return dispatch_operation(self.manager, permit, request,
                                  evict_cache=lambda: self.backend.call("evict_cache"))


def bind(selected):
    return (_CONTROLLER.bind(selected) if _CONTROLLER is not None else (selected, None))


def advertised_profiles(profiles):
    if _CONTROLLER is None:
        return list(profiles.items())
    aliases = _CONTROLLER.manager.status()["aliases"]
    return [(name, profiles[row["generation"]]) for name, row in aliases.items()
            if row["generation"] is not None]


async def public_status():
    result = {"schema": "flashalign.runtime-status.v1", "instance_id": _INSTANCE, "engine": "sglang",
              "version": "0.5.20", "numerical_attestation": False,
              "active_requests": _REQUESTS["active"], "request_errors_total": _REQUESTS["errors"]}
    if _CONTROLLER is not None:
        state = _CONTROLLER.manager.status()
        models = {json.loads(value)["id"] for value in _CONTROLLER.backend.models}
        if len(models) == 1:
            result["model"] = next(iter(models))
        active = {row["generation"] for row in state["aliases"].values() if row["generation"] is not None}
        result["generation"] = (next(iter(active)) if len(active) == 1 else
                                "epoch-" + hashlib.sha256(json.dumps(state["aliases"], sort_keys=True).encode()).hexdigest())
        if len(active) == 1:
            current = next(row for row in state["generations"] if row["id"] in active)
            result["profile_digest"] = current["digest"]
        method = getattr(_CONTROLLER.backend.tokenizer.server_args, "speculative_algorithm", None)
        result["speculation_enabled"] = method is not None
        if method is not None:
            result["speculative_method"] = str(method)
        try:
            workers = await asyncio.to_thread(_CONTROLLER.backend.call, "inspect")
            scheduler_active = [worker.get("active_requests") for worker in workers]
            cancellations = [worker.get("cancellation_matches_total") for worker in workers]
            if scheduler_active and all(type(count) is int and count >= 0 for count in scheduler_active):
                result["scheduler_active_requests"] = max(scheduler_active)
                result["active_requests"] = max(result["active_requests"], result["scheduler_active_requests"])
            idle = [worker.get("idle") for worker in workers]
            if idle and all(type(value) is bool for value in idle):
                result["scheduler_idle"] = all(idle)
            if cancellations and all(type(count) is int and count >= 0 for count in cancellations):
                result["cancellation_matches_total"] = max(cancellations)
            graphs = [item for worker in workers for item in worker.get("graph", [])]
            modes = {row["graph_mode"] for row in graphs}
            if modes:
                result["graph_mode"] = modes.pop() if len(modes) == 1 else "mixed"
            result["graph_worker_status"] = graphs
            if len(active) == 1:
                gid = next(iter(active))
                expected = [worker.get("expected_sites_by_generation", {}).get(gid) for worker in workers]
                installed = [worker.get("installed_sites_by_generation", {}).get(gid) for worker in workers]
                if expected and None not in expected and len(set(expected)) == 1:
                    result["expected_sites"] = expected[0]
                if installed and None not in installed:
                    result["installed_sites"] = min(installed)
        except Exception:
            result["worker_status_available"] = False
    return result


def capabilities():
    return {"schema": "flashalign.capabilities.v1", "hot_loading": _CONTROLLER is not None,
            "execution_attestation": False, "status": True}


async def start_api(tokenizer):
    global _CONTROLLER, _SERVER, _SOCKET
    state, address = os.getenv("FLASHALIGN_LIFECYCLE_STATE"), os.getenv("FLASHALIGN_ADMIN_SOCKET")
    if not state and not address:
        return
    if not state or not address:
        raise ValueError("Lifecycle state and admin socket must be configured together")
    _CONTROLLER = Controller(Backend(tokenizer, asyncio.get_running_loop()))
    try:
        await asyncio.to_thread(_CONTROLLER.initialize)
        _SERVER = await LocalAdminServer(address, status=_CONTROLLER.manager.status,
                                         operation=_CONTROLLER.operation, conflict_errors=(LifecycleError,)).start()
    except BaseException:
        _CONTROLLER.manager.close()
        _CONTROLLER = None
        raise


async def stop_api():
    global _SERVER, _CONTROLLER
    if _SERVER is not None:
        await _SERVER.close()
        _SERVER = None
    if _CONTROLLER is not None:
        _CONTROLLER.manager.close()
        _CONTROLLER = None
