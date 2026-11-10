# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Durable profile generations, separate from model execution and request auth.

Backends implement resource ownership and worker transport. This coordinator
does not load model code, interpret tensors, or trust a profile's executable
metadata. The staging callback must produce a validated private snapshot.

Mutation requires a separate administrator capability. Generation selection
is deliberately not an administration capability. Same-process code and code
running as the state-directory owner are part of the trusted boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import threading
from typing import Callable, Mapping, Protocol, Sequence


SCHEMA = "flashalign.profile-lifecycle.v1"
MAX_JOURNAL = 4 * 1024 * 1024
MAX_STAGED_ENTRIES = 1024
MAX_STAGED_BYTES = 16 * 1024**3
MAX_STAGED_DEPTH = 8
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9_-]{0,95}\Z")


class LifecycleError(RuntimeError):
    """An operation was rejected without changing the active profile."""


class AdministrationDenied(LifecycleError):
    pass


class RevisionConflict(LifecycleError):
    pass


class ResourceLimit(LifecycleError):
    pass


class GenerationBusy(LifecycleError):
    pass


class RecoveryRequired(LifecycleError):
    """New admission is fenced until the durable transaction is recovered."""


class JournalError(RecoveryRequired):
    pass


def _text(value, *, maximum=256):
    if (not isinstance(value, str) or not value or len(value) > maximum
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise LifecycleError("Invalid lifecycle identifier")
    return value


def _digest(value):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise LifecycleError("Invalid artifact digest")
    return value


def _resources(value):
    if not isinstance(value, Mapping) or not value or len(value) > 128:
        raise LifecycleError("Invalid resource declaration")
    result = {}
    for name, amount in value.items():
        _text(name, maximum=128)
        if type(amount) is not int or amount < 0 or amount > (1 << 63) - 1:
            raise LifecycleError("Invalid resource amount")
        result[name] = amount
    return result


def _strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise JournalError("Duplicate journal field")
            result[key] = value
        return result
    def invalid(_):
        raise JournalError("Non-finite journal value")
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise JournalError("Invalid lifecycle journal") from exc


@dataclass(frozen=True)
class Artifact:
    """Validated semantic identity. The directory is an owned staged snapshot."""

    name: str
    digest: str
    model_id: str
    model_revision: str
    operator: str
    directory: Path
    requirements: tuple[str, ...] = ()

    def __post_init__(self):
        _text(self.name)
        if self.name == "off":
            raise LifecycleError("off is reserved for stock execution")
        _digest(self.digest)
        _text(self.model_id, maximum=2048)
        _text(self.model_revision)
        _text(self.operator)
        if not isinstance(self.requirements, tuple):
            raise LifecycleError("Invalid operator requirements")
        for requirement in self.requirements:
            _text(requirement)
        if len(set(self.requirements)) != len(self.requirements):
            raise LifecycleError("Duplicate operator requirements")
        object.__setattr__(self, "directory", Path(self.directory))

    def record(self):
        return {"name": self.name, "digest": self.digest, "model_id": self.model_id,
                "model_revision": self.model_revision, "operator": self.operator,
                "requirements": list(self.requirements)}


@dataclass(frozen=True)
class Generation:
    id: str
    artifact: Artifact


@dataclass(frozen=True)
class WorkerAck:
    worker_id: str
    transaction_id: str
    generation_id: str | None
    artifact_digest: str | None
    phase: str


class LifecycleBackend(Protocol):
    """Trusted adapter contract; all operations must be idempotent by tx/id.

    prepare must leave old bindings unchanged. fence pauses new admissions on
    every worker; commit occurs only after the durable decision. resume may
    run only after every worker has acknowledged that decision. abort releases
    only resources newly owned by that transaction. reconcile restores all
    journaled allocated generations and live reference accounting after a
    coordinator restart. references includes pending GPU, cache and graph work.
    retire must refuse concurrent references and never reuse generation IDs.

    estimate returns named profile admission-budget estimates. These counters
    do not establish physical allocator usage or an engine-wide memory cap.
    Any physical allocation limit must be declared and enforced separately by
    the backend; reserve alone does not prove physical memory was acquired.
    """

    identity: str
    worker_ids: Sequence[str]

    def validate(self, artifact: Artifact) -> None: ...
    def estimate(self, artifact: Artifact) -> Mapping[str, int]: ...
    def reserve(self, transaction: Mapping, generation: Generation) -> None: ...
    def prepare(self, transaction: Mapping, generation: Generation) -> Sequence[WorkerAck]: ...
    def fence(self, transaction: Mapping) -> Sequence[WorkerAck]: ...
    def commit(self, transaction: Mapping, generation: Generation) -> Sequence[WorkerAck]: ...
    def resume(self, transaction: Mapping) -> Sequence[WorkerAck]: ...
    def abort(self, transaction: Mapping, generation: Generation) -> Sequence[WorkerAck]: ...
    def references(self, generation: Generation) -> int: ...
    def retire(self, transaction: Mapping, generation: Generation) -> Sequence[WorkerAck]: ...
    def reconcile(self, transaction: Mapping, snapshot: Mapping) -> Sequence[WorkerAck]: ...


class AdminPermit:
    __slots__ = ("_authority",)

    def __init__(self, authority):
        self._authority = authority


class Lease:
    """A trusted request/cache/GPU reference to one immutable generation."""

    def __init__(self, manager, generation, kind):
        self._manager = manager
        self.generation = generation
        self.kind = kind
        self._closed = False

    def retain(self, kind="work"):
        with self._manager._lock:
            if self._closed:
                raise GenerationBusy("Cannot retain a released generation")
            return self._manager._lease(self.generation.id, kind)

    def close(self):
        with self._manager._lock:
            if not self._closed:
                self._closed = True
                refs = self._manager._references[self.generation.id]
                refs[self.kind] -= 1
                if refs[self.kind] == 0:
                    del refs[self.kind]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class LifecycleManager:
    """One serialized administrator, concurrent generation-pinned requests."""

    def __init__(self, state_dir, *, backend: LifecycleBackend,
                 stager: Callable[[Path, Path], Artifact], admin_key: bytes,
                 resource_limits: Mapping[str, int], max_generations=64):
        if not isinstance(admin_key, bytes) or len(admin_key) != 32:
            raise AdministrationDenied("A separate 32-byte administration key is required")
        if type(max_generations) is not int or not 2 <= max_generations <= 4096:
            raise LifecycleError("Invalid generation limit")
        self.backend, self.stager = backend, stager
        self.identity = _text(backend.identity, maximum=2048)
        self.workers = tuple(backend.worker_ids)
        if (not self.workers or len(self.workers) > 256
                or len(set(self.workers)) != len(self.workers)):
            raise LifecycleError("Invalid worker inventory")
        for worker in self.workers:
            _text(worker)
        self.limits = _resources(resource_limits)
        self.max_generations = max_generations
        self._admin_key, self._authority = admin_key, object()
        self._lock, self._mutation = threading.RLock(), threading.RLock()
        self._references = {}
        self._generations, self._aliases = {}, {}
        self._pending, self._blocked = None, False
        self._journal_failed = False
        self._closed = False
        self.root = Path(state_dir).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._private_dir(self.root)
        self.artifacts = self.root / "generations"
        self.artifacts.mkdir(mode=0o700, exist_ok=True)
        self._private_dir(self.artifacts)
        self.journal = self.root / "lifecycle.json"
        self._lock_fd = None
        try:
            self._acquire_process_lock()
            self._load()
        except BaseException:
            self.close()
            raise

    @property
    def journal_was_present(self):
        """Whether this opener found a journal while holding its process lock.

        Empty journals are deliberate durable state. This lifetime fact lets
        startup adapters distinguish them from a new deployment without a
        check-before-lock race or inferring presence from inventory contents.
        """
        return self._journal_was_present

    @staticmethod
    def _private_dir(path):
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
                or info.st_uid != os.getuid()):
            raise JournalError("Lifecycle state must be an owned mode-0700 directory")

    def _acquire_process_lock(self):
        import fcntl
        fd = os.open(self.root / "coordinator.lock",
                     os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise JournalError("Invalid coordinator lock")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise JournalError("Another coordinator owns this state") from None
        self._lock_fd = fd

    def admin(self, key):
        if not isinstance(key, bytes) or not hmac.compare_digest(key, self._admin_key):
            raise AdministrationDenied("Administration denied")
        return AdminPermit(self._authority)

    def _authorize(self, permit):
        if (self._closed or not isinstance(permit, AdminPermit)
                or permit._authority is not self._authority):
            raise AdministrationDenied("Administration denied")

    def _generation(self, gid):
        if not isinstance(gid, str) or not _ID.fullmatch(gid):
            raise LifecycleError("Invalid generation identifier")
        row = self._generations.get(gid)
        if row is None or row["state"] == "retired":
            raise LifecycleError("Unknown generation")
        meta = row["artifact"]
        artifact = Artifact(meta["name"], meta["digest"], meta["model_id"],
                            meta["model_revision"], meta["operator"],
                            self.artifacts / gid, tuple(meta["requirements"]))
        return Generation(gid, artifact)

    def _snapshot(self):
        return {"schema": SCHEMA, "backend": self.identity, "workers": list(self.workers),
                "aliases": self._aliases, "generations": self._generations,
                "pending": self._pending}

    def _save(self):
        encoded = json.dumps(self._snapshot(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        if len(encoded) > MAX_JOURNAL:
            self._blocked = True
            raise JournalError("Lifecycle journal exceeds its limit")
        temporary = self.root / (".journal-" + secrets.token_hex(16))
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.journal)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except (OSError, ValueError) as exc:
            self._blocked = True
            self._journal_failed = True
            raise JournalError("Cannot commit lifecycle journal") from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _load(self):
        try:
            fd = os.open(self.journal, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            self._journal_was_present = False
            self._save()
            return
        self._journal_was_present = True
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                    or info.st_size > MAX_JOURNAL):
                raise JournalError("Invalid lifecycle journal permissions or size")
            value = _strict_json(stream.read(MAX_JOURNAL + 1))
        if (not isinstance(value, dict) or set(value) !=
                {"schema", "backend", "workers", "aliases", "generations", "pending"}
                or value["schema"] != SCHEMA or value["backend"] != self.identity
                or value["workers"] != list(self.workers)):
            raise JournalError("Lifecycle journal identity mismatch")
        generations, aliases = value["generations"], value["aliases"]
        if (not isinstance(generations, dict) or len(generations) > self.max_generations
                or not isinstance(aliases, dict) or len(aliases) > self.max_generations):
            raise JournalError("Invalid lifecycle inventory")
        self._generations, self._aliases = generations, aliases
        try:
            for gid, row in generations.items():
                if (not isinstance(gid, str) or not _ID.fullmatch(gid)
                        or not isinstance(row, dict)
                        or set(row) != {"artifact", "resources", "allocated", "state"}
                        or not isinstance(row["artifact"], dict)
                        or set(row["artifact"]) != {"name", "digest", "model_id", "model_revision", "operator", "requirements"}):
                    raise LifecycleError("Invalid generation record")
                if type(row["allocated"]) is not bool or row["state"] not in {"staged", "active", "retained", "retiring", "retired"}:
                    raise LifecycleError("Invalid generation state")
                _resources(row["resources"])
                if row["state"] != "retired":
                    self._private_dir(self.artifacts / gid)
                    self.backend.validate(self._generation(gid).artifact)
                self._references[gid] = {}
            for alias, row in aliases.items():
                _text(alias)
                if (alias == "off" or not isinstance(row, dict)
                        or set(row) != {"generation", "revision"}
                        or type(row["revision"]) is not int or row["revision"] < 1
                        or (row["generation"] is not None and
                            (row["generation"] not in generations or not generations[row["generation"]]["allocated"]))):
                    raise LifecycleError("Invalid alias record")
            pending = value["pending"]
            if pending is not None:
                self._check_transaction(pending)
            self._pending = pending
            self._check_budget({})
        except (KeyError, TypeError, ValueError, LifecycleError) as exc:
            raise JournalError("Invalid lifecycle records") from exc
        self._blocked = bool(pending or any(row["allocated"] for row in generations.values()))

    def _check_transaction(self, transaction):
        if (not isinstance(transaction, dict) or set(transaction) !=
                {"id", "operation", "alias", "generation", "previous", "revision", "phase", "was_allocated"}
                or not _ID.fullmatch(transaction["id"])
                or transaction["generation"] not in self._generations
                or type(transaction["was_allocated"]) is not bool):
            raise JournalError("Invalid pending transaction")
        if transaction["operation"] == "retire":
            if (transaction["phase"] != "retiring" or transaction["alias"] is not None
                    or transaction["previous"] is not None or transaction["revision"] != 0):
                raise JournalError("Invalid retirement transaction")
            return
        if (transaction["operation"] not in {"activate", "deactivate"}
                or transaction["phase"] not in {"preparing", "fencing", "committing", "resuming", "aborting"}
                or type(transaction["revision"]) is not int or transaction["revision"] < 1):
            raise JournalError("Invalid activation transaction")
        _text(transaction["alias"])
        if transaction["previous"] is not None and transaction["previous"] not in self._generations:
            raise JournalError("Unknown prior generation")
        if transaction["operation"] == "deactivate" and (
                transaction["previous"] != transaction["generation"] or not transaction["was_allocated"]):
            raise JournalError("Invalid deactivation transaction")

    def _check_budget(self, addition):
        totals = {name: 0 for name in self.limits}
        for row in self._generations.values():
            if row["allocated"]:
                for name, amount in row["resources"].items():
                    if name not in totals:
                        raise ResourceLimit("Unknown resource dimension")
                    totals[name] += amount
        for name, amount in addition.items():
            if name not in totals:
                raise ResourceLimit("Resource has no configured limit")
            totals[name] += amount
        if any(amount > self.limits[name] for name, amount in totals.items()):
            raise ResourceLimit("Insufficient profile overlap budget")

    def _sync_snapshot(self, path):
        """Make the complete owned snapshot durable without following links."""
        counts = {"entries": 0, "bytes": 0}

        def identity(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)

        def visit(directory, depth):
            if depth > MAX_STAGED_DEPTH:
                raise ResourceLimit("Staged artifact nesting exceeds its limit")
            directory_before = os.fstat(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    counts["entries"] += 1
                    if counts["entries"] > MAX_STAGED_ENTRIES:
                        raise ResourceLimit("Staged artifact entry count exceeds its limit")
                    before = entry.stat(follow_symlinks=False)
                    if before.st_uid != os.getuid():
                        raise LifecycleError("Staged artifacts must be owned by the coordinator user")
                    if stat.S_ISDIR(before.st_mode):
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                        dir_fd=directory)
                        try:
                            opened = os.fstat(child)
                            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                                raise LifecycleError("Staged directory changed during durability check")
                            visit(child, depth + 1)
                            current = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
                            if identity(current) != identity(before):
                                raise LifecycleError("Staged directory changed during durability check")
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                        counts["bytes"] += before.st_size
                        if counts["bytes"] > MAX_STAGED_BYTES:
                            raise ResourceLimit("Staged artifact byte size exceeds its limit")
                        descriptor = os.open(entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                             dir_fd=directory)
                        try:
                            if identity(os.fstat(descriptor)) != identity(before):
                                raise LifecycleError("Staged file changed during durability check")
                            os.fsync(descriptor)
                            after = os.fstat(descriptor)
                            current = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
                            if identity(after) != identity(before) or identity(current) != identity(before):
                                raise LifecycleError("Staged file changed during durability check")
                        finally:
                            os.close(descriptor)
                    else:
                        raise LifecycleError("Staged artifacts cannot contain links or special files")
            os.fsync(directory)
            if identity(os.fstat(directory)) != identity(directory_before):
                raise LifecycleError("Staged directory changed during durability check")

        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            visit(descriptor, 0)
        except OSError as error:
            raise LifecycleError("Cannot synchronize the staged artifact") from error
        finally:
            os.close(descriptor)

    def _acks(self, result, transaction, phase, generation=None):
        if not isinstance(result, (list, tuple)) or len(result) != len(self.workers):
            raise RecoveryRequired("Missing worker acknowledgments")
        seen = set()
        for ack in result:
            if (not isinstance(ack, WorkerAck) or ack.worker_id not in self.workers
                    or ack.worker_id in seen or ack.transaction_id != transaction["id"]
                    or ack.phase != phase or ack.generation_id != (generation.id if generation else None)
                    or ack.artifact_digest != (generation.artifact.digest if generation else None)):
                raise RecoveryRequired("Invalid worker acknowledgment")
            seen.add(ack.worker_id)

    def stage(self, permit, source):
        self._authorize(permit)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", str(source)):
            raise LifecycleError("Stage an operator-provided local artifact")
        source = Path(source)
        if not source.is_absolute():
            source = source.absolute()
        if "://" in str(source):
            raise LifecycleError("Stage an operator-provided local artifact")
        with self._mutation:
            with self._lock:
                if self._blocked:
                    raise RecoveryRequired("Lifecycle recovery is required")
                self._purge_retired()
                if len(self._generations) >= self.max_generations:
                    raise ResourceLimit("Generation inventory is full")
            gid = "g-" + secrets.token_hex(24)
            temporary = self.artifacts / ("stage-" + secrets.token_hex(24))
            temporary.mkdir(mode=0o700)
            published = False
            try:
                artifact = self.stager(source, temporary)
                if not isinstance(artifact, Artifact) or artifact.directory.absolute() != temporary:
                    raise LifecycleError("Stager did not return its validated snapshot")
                self.backend.validate(artifact)
                resources = _resources(self.backend.estimate(artifact))
                final = self.artifacts / gid
                self._sync_snapshot(temporary)
                os.replace(temporary, final)
                published = True
                descriptor = os.open(self.artifacts, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                artifact = replace(artifact, directory=final)
                with self._lock:
                    self._generations[gid] = {"artifact": artifact.record(), "resources": resources,
                                              "allocated": False, "state": "staged"}
                    self._references[gid] = {}
                    self._save()
                return Generation(gid, artifact)
            finally:
                if not published and temporary.exists():
                    shutil.rmtree(temporary)

    def activate_configured(self, permit, source, *, name):
        """Ensure a configured name has an active alias after journal recovery."""
        self._authorize(permit)
        with self._mutation:
            current = self.status()["aliases"].get(name)
            if current is not None and current["generation"] is not None:
                return
            generation = self.stage(permit, source)
            if generation.artifact.name != name:
                raise LifecycleError("Configured profile name changed during staging")
            self.activate(permit, name, generation.id,
                          expected_revision=current["revision"] if current else 0)

    def activate(self, permit, alias, generation_id, *, expected_revision):
        self._authorize(permit)
        _text(alias)
        if re.fullmatch(r"g-[0-9a-f]{48}", alias):
            raise LifecycleError("Alias uses the reserved immutable generation namespace")
        if alias == "off" or type(expected_revision) is not int or expected_revision < 0:
            raise RevisionConflict("Invalid alias or expected revision")
        with self._mutation:
            with self._lock:
                if self._blocked or self._pending is not None:
                    raise RecoveryRequired("Lifecycle recovery is required")
                current = self._aliases.get(alias)
                if current is None and len(self._aliases) >= self.max_generations:
                    raise ResourceLimit("Alias inventory is full")
                revision = current["revision"] if current else 0
                if expected_revision != revision:
                    raise RevisionConflict("Alias revision changed")
                generation = self._generation(generation_id)
                row = self._generations[generation_id]
                allocated = row["allocated"]
                if not allocated:
                    self._check_budget(row["resources"])
                transaction = {"id": "t-" + secrets.token_hex(24), "operation": "activate", "alias": alias,
                               "generation": generation_id, "previous": current["generation"] if current else None,
                               "revision": revision + 1, "phase": "preparing", "was_allocated": allocated}
                self._pending = transaction
                row["allocated"] = True
                self._save()
            try:
                if not allocated:
                    self.backend.reserve(dict(transaction), generation)
                self._acks(self.backend.prepare(dict(transaction), generation), transaction, "prepared", generation)
                with self._lock:
                    transaction["phase"] = "fencing"
                    self._blocked = True
                    self._save()
                self._acks(self.backend.fence(dict(transaction)), transaction, "fenced")
                with self._lock:
                    transaction["phase"] = "committing"
                    self._save()  # Durable decision precedes every commit RPC.
                self._finish_commit(transaction, generation)
            except BaseException as exc:
                if transaction["phase"] in {"committing", "resuming"}:
                    with self._lock:
                        self._blocked = True
                    raise RecoveryRequired("Committed update requires recovery") from exc
                self._abort(transaction, generation, exc)
                raise LifecycleError("Profile preparation failed; prior selection retained") from exc
            return {"alias": alias, "generation": generation.id, "revision": transaction["revision"],
                    "digest": generation.artifact.digest}

    def _finish_commit(self, transaction, generation):
        self._acks(self.backend.commit(dict(transaction), generation), transaction, "committed", generation)
        with self._lock:
            target = None if transaction["operation"] == "deactivate" else generation.id
            self._aliases[transaction["alias"]] = {"generation": target,
                                                   "revision": transaction["revision"]}
            for gid, row in self._generations.items():
                if row["allocated"]:
                    row["state"] = "active" if any(a["generation"] == gid for a in self._aliases.values()) else "retained"
            transaction["phase"] = "resuming"
            self._save()
        self._acks(self.backend.resume(dict(transaction)), transaction, "resumed")
        with self._lock:
            self._pending = None
            self._save()
            self._blocked = False

    def _abort(self, transaction, generation, cause):
        with self._lock:
            self._blocked = True
            transaction["phase"] = "aborting"
            self._save()
        try:
            self._acks(self.backend.abort(dict(transaction), generation), transaction, "aborted", generation)
            self._acks(self.backend.resume(dict(transaction)), transaction, "resumed")
            with self._lock:
                row = self._generations[generation.id]
                row["allocated"] = transaction["was_allocated"]
                if not row["allocated"]:
                    row["state"] = "staged"
                self._pending = None
                self._save()
                self._blocked = False
        except BaseException as exc:
            raise RecoveryRequired("Aborted update requires recovery") from exc

    def recover(self, permit):
        self._authorize(permit)
        if self._journal_failed:
            raise RecoveryRequired("Reopen the coordinator to recover its durable journal")
        with self._mutation:
            with self._lock:
                self._blocked = True
                transaction = self._pending or {"id": "r-" + secrets.token_hex(24)}
                snapshot = json.loads(json.dumps(self._snapshot()))
                snapshot["artifact_root"] = str(self.artifacts)
            try:
                self._acks(self.backend.reconcile(dict(transaction), snapshot), transaction, "reconciled")
                if self._pending is not None:
                    generation = self._generation(transaction["generation"])
                    if transaction["operation"] == "retire":
                        self._finish_retirement(transaction, generation)
                        self._acks(self.backend.resume(dict(transaction)), transaction, "resumed")
                        with self._lock:
                            self._blocked = False
                    elif transaction["phase"] in {"preparing", "fencing", "aborting"}:
                        self._abort(transaction, generation, None)
                    else:
                        self._acks(self.backend.fence(dict(transaction)), transaction, "fenced")
                        if transaction["operation"] != "deactivate":
                            self._acks(self.backend.prepare(dict(transaction), generation), transaction, "prepared", generation)
                        self._finish_commit(transaction, generation)
                else:
                    self._acks(self.backend.resume(dict(transaction)), transaction, "resumed")
                    with self._lock:
                        self._purge_retired()
                        self._save()
                        self._blocked = False
            except BaseException as exc:
                with self._lock:
                    self._blocked = True
                raise RecoveryRequired("Workers have not recovered the committed state") from exc

    def deactivate(self, permit, alias, *, expected_revision):
        """Tombstone an alias using CAS; admitted work retains its generation.

        The backend receives commit(operation='deactivate', generation=prior)
        and must remove only the admission alias, not prior worker resources.
        """
        self._authorize(permit)
        _text(alias)
        if alias == "off" or type(expected_revision) is not int or expected_revision < 0:
            raise RevisionConflict("Invalid alias or expected revision")
        with self._mutation:
            with self._lock:
                if self._blocked or self._pending is not None:
                    raise RecoveryRequired("Lifecycle recovery is required")
                current = self._aliases.get(alias)
                revision = current["revision"] if current else 0
                if expected_revision != revision:
                    raise RevisionConflict("Alias revision changed")
                if current is None or current["generation"] is None:
                    raise LifecycleError("Alias is not active")
                generation = self._generation(current["generation"])
                transaction = {"id": "t-" + secrets.token_hex(24), "operation": "deactivate", "alias": alias,
                               "generation": generation.id, "previous": generation.id, "revision": revision + 1,
                               "phase": "fencing", "was_allocated": True}
                self._pending = transaction
                self._blocked = True
                self._save()
            try:
                self._acks(self.backend.fence(dict(transaction)), transaction, "fenced")
                with self._lock:
                    transaction["phase"] = "committing"
                    self._save()
                self._finish_commit(transaction, generation)
            except BaseException as exc:
                if transaction["phase"] in {"committing", "resuming"}:
                    with self._lock:
                        self._blocked = True
                    raise RecoveryRequired("Committed deactivation requires recovery") from exc
                self._abort(transaction, generation, exc)
                raise LifecycleError("Alias deactivation failed; prior selection retained") from exc
            return {"alias": alias, "generation": None, "revision": transaction["revision"]}

    def rollback(self, permit, alias, generation_id, *, expected_revision):
        """Rollback is another versioned activation, never an in-place edit."""
        return self.activate(permit, alias, generation_id, expected_revision=expected_revision)

    def _lease(self, gid, kind):
        _text(kind, maximum=64)
        generation = self._generation(gid)
        if not self._generations[gid]["allocated"]:
            raise GenerationBusy("Generation has no worker resources")
        counts = self._references[gid]
        counts[kind] = counts.get(kind, 0) + 1
        return Lease(self, generation, kind)

    def pin(self, alias):
        _text(alias)
        with self._lock:
            if self._closed or self._blocked:
                raise RecoveryRequired("New admission is fenced")
            row = self._aliases.get(alias)
            if row is None or row["generation"] is None:
                raise LifecycleError("Unknown profile alias")
            return self._lease(row["generation"], "request")

    def retire(self, permit, generation_id):
        self._authorize(permit)
        with self._mutation:
            with self._lock:
                if self._blocked or self._pending:
                    raise RecoveryRequired("Lifecycle recovery is required")
                generation = self._generation(generation_id)
                row = self._generations[generation_id]
                if (any(a["generation"] == generation_id for a in self._aliases.values())
                        or any(self._references[generation_id].values())):
                    raise GenerationBusy("Generation is still referenced")
            if row["allocated"]:
                count = self.backend.references(generation)
                if type(count) is not int or count < 0:
                    raise RecoveryRequired("Invalid worker reference count")
                if count:
                    raise GenerationBusy("Workers still reference this generation")
            transaction = {"id": "d-" + secrets.token_hex(24), "operation": "retire", "alias": None,
                           "generation": generation.id, "previous": None, "revision": 0,
                           "phase": "retiring", "was_allocated": row["allocated"]}
            with self._lock:
                row["state"] = "retiring"
                self._pending = transaction
                self._save()
            try:
                self._finish_retirement(transaction, generation)
            except BaseException as exc:
                with self._lock:
                    self._blocked = True
                raise RecoveryRequired("Generation retirement requires recovery") from exc

    def _finish_retirement(self, transaction, generation):
        row = self._generations[generation.id]
        if row["allocated"]:
            count = self.backend.references(generation)
            if type(count) is not int or count != 0:
                raise GenerationBusy("Worker references have not drained")
            self._acks(self.backend.retire(dict(transaction), generation), transaction, "retired", generation)
        with self._lock:
            row["allocated"] = False
            row["state"] = "retired"
            self._pending = None
            self._save()
            self._purge_retired()
            self._save()

    def _purge_retired(self):
        for gid, row in list(self._generations.items()):
            if row["state"] != "retired":
                continue
            directory = self.artifacts / gid
            if directory.exists():
                self._private_dir(directory)
                shutil.rmtree(directory)
            del self._generations[gid]
            del self._references[gid]

    def status(self):
        """Safe operational metadata; no artifact paths, keys or request data."""
        with self._lock:
            return {"schema": SCHEMA, "backend": self.identity, "workers": list(self.workers),
                    "admission_fenced": self._blocked, "recovery_required": self._blocked,
                    "aliases": json.loads(json.dumps(self._aliases)),
                    "pending_phase": self._pending["phase"] if self._pending else None,
                    "generations": [{"id": gid, **row["artifact"], "state": row["state"],
                                     "allocated": row["allocated"], "resources": dict(row["resources"]),
                                     "references": dict(self._references[gid])}
                                    for gid, row in self._generations.items()]}

    def close(self):
        self._closed = True
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
