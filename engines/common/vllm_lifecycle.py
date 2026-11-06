# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fixed internal vLLM RPC for immutable profile generations.

This module is shared by the model adapters. It does not expose vLLM's
arbitrary-method RPC on HTTP. Administration calls run on a separate thread;
the engine client continues making progress on its original asyncio loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
import threading

from owner_mode.profile_lifecycle import (
    Artifact, GenerationBusy, LifecycleError, RecoveryRequired, ResourceLimit, WorkerAck,
)
from owner_mode.profiles import profile_fingerprint, validate_profile


RPC_METHOD = "flashalign_lifecycle"
SCHEMA = "flashalign.vllm-lifecycle.v1"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,95}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def checked_artifact(record):
    if not isinstance(record, dict) or set(record) != {
            "name", "digest", "model_id", "model_revision", "operator", "requirements", "directory"}:
        raise LifecycleError("Invalid internal artifact record")
    path = Path(record["directory"])
    if not path.is_absolute():
        raise LifecycleError("Artifact path must be absolute")
    artifact = Artifact(record["name"], record["digest"], record["model_id"],
                        record["model_revision"], record["operator"], path,
                        tuple(record["requirements"]))
    actual = validate_profile(path)
    if (profile_fingerprint(actual) != artifact.digest or actual["name"] != artifact.name
            or actual["model"] != {"id": artifact.model_id, "revision": artifact.model_revision}
            or actual["transform"] != artifact.operator
            or tuple(sorted(actual["requires"])) != artifact.requirements):
        raise LifecycleError("Staged artifact identity changed")
    return artifact


def artifact_record(artifact):
    return {**artifact.record(), "directory": str(artifact.directory)}


@dataclass
class WorkerGeneration:
    artifact: Artifact
    handle: object
    committed: bool = False
    cached: bool = False


class WorkerRegistry:
    """Worker-owned tensors. Public aliases never select a tensor here."""
    def __init__(self, adapter, worker_id):
        self.adapter = adapter
        self.worker_id = str(worker_id)
        self.generations = {}
        self.fenced = True
        self.aliases = {}

    def get(self, name):
        return self.adapter.profiles.get(name)

    def resolve(self, generation_id, digest):
        record = self.generations.get(generation_id)
        if record is None or not record.committed or record.artifact.digest != digest:
            raise LifecycleError("Unknown or uncommitted worker generation")
        record.cached = True
        return record.handle

    def references(self, generation_id):
        record = self.generations.get(generation_id)
        if record is None:
            return 0
        return int(record.cached) + self.adapter.references(generation_id)

    def _prepare(self, gid, artifact):
        previous = self.generations.get(gid)
        if previous is not None:
            if previous.artifact != artifact:
                raise LifecycleError("Generation identity is immutable")
            callback = getattr(self.adapter, 'prepare_existing', None)
            if callback is not None:
                callback(gid, artifact, previous.handle)
            return
        self.adapter.validate(artifact)
        self.generations[gid] = WorkerGeneration(artifact, self.adapter.prepare(gid, artifact))

    def dispatch(self, envelope):
        if not isinstance(envelope, dict) or envelope.get("schema") != SCHEMA:
            raise LifecycleError("Invalid internal lifecycle request")
        allowed = {"schema", "operation", "transaction", "generation", "artifact", "snapshot"}
        if set(envelope) - allowed:
            raise LifecycleError("Unknown internal lifecycle field")
        op = envelope.get("operation")
        if op in ("inventory", "status"):
            if set(envelope) != {"schema", "operation"}:
                raise LifecycleError("Invalid inventory request")
            if op == 'status':
                return {'worker_id': self.worker_id, **self.adapter.statistics()}
            return {"worker_id": self.worker_id, **self.adapter.inventory()}
        tx = envelope.get("transaction")
        if not isinstance(tx, dict) or not _ID.fullmatch(str(tx.get("id", ""))):
            raise LifecycleError("Invalid lifecycle transaction")
        gid = envelope.get("generation")
        artifact = None
        phases = {"prepare": "prepared", "fence": "fenced", "commit": "committed",
                  "resume": "resumed", "abort": "aborted", "retire": "retired",
                  "reconcile": "reconciled"}
        if op not in {*phases, "references", "evict_cache"}:
            raise LifecycleError("Unsupported lifecycle operation")
        if op in {"prepare", "commit", "abort", "retire", "references"}:
            if not isinstance(gid, str) or not _ID.fullmatch(gid):
                raise LifecycleError("Invalid generation identity")
            artifact = checked_artifact(envelope.get("artifact"))
            if tx.get("generation") != gid:
                raise LifecycleError("Transaction generation mismatch")
        if op == "references":
            return {"worker_id": self.worker_id, "references": self.references(gid)}
        if op == "evict_cache":
            if self.adapter.all_references():
                raise GenerationBusy("Requests are still running")
            callback = getattr(self.adapter, 'prune_graphs', None)
            if callback is not None:
                callback(set(self.aliases.values()))
            for record in self.generations.values():
                record.cached = False
            return {"worker_id": self.worker_id, "evicted": True}
        if op == "prepare":
            self._prepare(gid, artifact)
        elif op == "fence":
            self.fenced = True
        elif op == "commit":
            record = self.generations.get(gid)
            if not self.fenced or record is None or record.artifact != artifact:
                raise RecoveryRequired("Commit has no matching prepared generation or fence")
            callback = getattr(self.adapter, 'commit', None)
            if callback is not None:
                callback(gid, record.handle, tx)
            record.committed = True
            if tx.get('operation') == 'deactivate':
                self.aliases.pop(tx['alias'], None)
            elif tx.get('alias') is not None:
                self.aliases[tx['alias']] = gid
        elif op == "resume":
            self.fenced = False
        elif op in {"abort", "retire"}:
            record = self.generations.get(gid)
            if record is not None:
                if record.artifact != artifact:
                    raise LifecycleError("Generation identity changed")
                if op == "retire" or not tx.get("was_allocated", False):
                    if self.references(gid):
                        raise GenerationBusy("Worker generation is still referenced")
                    self.adapter.retire(gid, record.handle)
                    del self.generations[gid]
        elif op == "reconcile":
            self.fenced = True
            snapshot = envelope.get("snapshot")
            if not isinstance(snapshot, dict):
                raise LifecycleError("Recovery requires a journal snapshot")
            root = Path(snapshot["artifact_root"])
            wanted = set()
            pending = snapshot.get('pending') or {}
            discard_prepared = (pending.get('operation') == 'activate'
                and pending.get('phase') in ('preparing', 'fencing', 'aborting')
                and pending.get('was_allocated') is False)
            for key, row in snapshot["generations"].items():
                if not _ID.fullmatch(key):
                    raise LifecycleError("Invalid recovery generation")
                if (row["allocated"] and row.get('state') != 'retiring'
                        and not (discard_prepared and key == pending.get('generation'))):
                    recovered = checked_artifact({**row["artifact"], "directory": str(root / key)})
                    self._prepare(key, recovered)
                    self.generations[key].committed = True
                    wanted.add(key)
            for key in set(self.generations) - wanted:
                if self.references(key):
                    raise RecoveryRequired("Unjournaled generation still in use")
                self.adapter.retire(key, self.generations[key].handle)
                del self.generations[key]
            self.aliases = {key: row['generation'] for key, row in snapshot['aliases'].items()
                            if row['generation'] is not None}
            callback = getattr(self.adapter, 'recover', None)
            if callback is not None:
                callback(wanted)
        return {"worker_id": self.worker_id, "transaction_id": tx["id"],
                "generation_id": gid if artifact else None,
                "artifact_digest": artifact.digest if artifact else None, "phase": phases[op]}


class WorkerExtension:
    """One named internal entrypoint, installed via --worker-extension-cls."""
    def flashalign_lifecycle(self, envelope):
        registry = getattr(self.model_runner, "_flashalign_lifecycle", None)
        if registry is None:
            raise LifecycleError("Worker has no FlashAlign lifecycle adapter")
        return registry.dispatch(envelope)


class VllmBackend:
    def __init__(self, engine_client, loop, *, validate, estimate, timeout=120):
        self.client, self.loop = engine_client, loop
        self._validate, self._estimate, self.timeout = validate, estimate, timeout
        self.identity = "vllm"
        inventories = self._call("inventory")
        self.worker_ids = tuple(row["worker_id"] for row in inventories)
        if not self.worker_ids or len(set(self.worker_ids)) != len(self.worker_ids):
            raise RecoveryRequired("Worker inventory is incomplete")
        self.inventories = inventories

    def _call(self, operation, transaction=None, generation=None, snapshot=None):
        if threading.get_ident() == getattr(self.loop, "_thread_id", None):
            raise RuntimeError("Lifecycle RPC must not block its asyncio event loop")
        message = {"schema": SCHEMA, "operation": operation}
        if transaction is not None:
            message["transaction"] = transaction
        if generation is not None:
            message.update(generation=generation.id, artifact=artifact_record(generation.artifact))
        if snapshot is not None:
            message["snapshot"] = snapshot
        async def invoke():
            return await self.client.collective_rpc(RPC_METHOD, timeout=self.timeout, args=(message,))
        future = asyncio.run_coroutine_threadsafe(invoke(), self.loop)
        try:
            result = future.result(self.timeout + 5)
        except BaseException:
            future.cancel()
            raise
        if not isinstance(result, list):
            raise RecoveryRequired("Invalid collective RPC result")
        return result

    def _acks(self, operation, transaction, generation=None, snapshot=None):
        return [WorkerAck(**row) for row in self._call(operation, transaction, generation, snapshot)]

    def validate(self, artifact):
        self._validate(artifact)

    def estimate(self, artifact):
        return self._estimate(artifact)

    def reserve(self, transaction, generation):
        # The coordinator reserves overlap budget before any device allocation.
        self.validate(generation.artifact)

    def prepare(self, transaction, generation):
        return self._acks("prepare", transaction, generation)

    def fence(self, transaction):
        return self._acks("fence", transaction)

    def commit(self, transaction, generation):
        return self._acks("commit", transaction, generation)

    def resume(self, transaction):
        return self._acks("resume", transaction)

    def abort(self, transaction, generation):
        return self._acks("abort", transaction, generation)

    def retire(self, transaction, generation):
        return self._acks("retire", transaction, generation)

    def references(self, generation):
        rows = self._call("references", {"id": "references", "generation": generation.id}, generation)
        if {r.get("worker_id") for r in rows} != set(self.worker_ids):
            raise RecoveryRequired("Worker reference inventory changed")
        if any(type(r.get("references")) is not int or r["references"] < 0 for r in rows):
            raise RecoveryRequired("Invalid worker reference count")
        return sum(row["references"] for row in rows)

    def reconcile(self, transaction, snapshot):
        return self._acks("reconcile", transaction, snapshot=snapshot)

    def evict_cache(self):
        async def reset():
            return await self.client.reset_prefix_cache(reset_running_requests=False)
        future = asyncio.run_coroutine_threadsafe(reset(), self.loop)
        if future.result(self.timeout + 5) is not True:
            raise GenerationBusy("Prefix cache is in use")
        rows = self._call("evict_cache", {"id": "evict-cache"})
        if {r.get("worker_id") for r in rows} != set(self.worker_ids) or not all(r.get("evicted") is True for r in rows):
            raise RecoveryRequired("Cache eviction acknowledgment is incomplete")
        return {"evicted": True}
