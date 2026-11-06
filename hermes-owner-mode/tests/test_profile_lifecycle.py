# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CPU contract tests for immutable profile generations and crash recovery."""
import hashlib
import fcntl
import json
import os
import stat
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_setup import wizard
from owner_mode.profile_lifecycle import (Artifact, AdministrationDenied,
    GenerationBusy, JournalError, LifecycleError, LifecycleManager,
    RecoveryRequired, ResourceLimit, RevisionConflict, WorkerAck)
from owner_mode.profile_staging import stage_validated_profile


KEY = b"K" * 32  # Synthetic fixture only.


def stage_fixture(source, destination):
    raw = Path(source).read_bytes()
    value = json.loads(raw)
    (destination / "artifact.json").write_bytes(raw)
    return Artifact(value["name"], hashlib.sha256(raw).hexdigest(), value["model"],
                    "fixture-revision", value["operator"], destination)


class ContractBackend:
    """A strict fake worker transport; no model or GPU work is simulated."""

    identity = "test.dense.v1"
    worker_ids = ("worker-0",)
    model = "unseen/model"
    operators = {"projection", "native_matrix_override_v1"}

    def __init__(self):
        self.live = {}
        self.aliases = {}
        self.fenced = False
        self.log = []
        self.fail = None
        self.bad_ack = None
        self.refs = {}
        self.prepare_entered = None
        self.prepare_release = None
        self.journal = None

    def validate(self, artifact):
        if artifact.model_id != self.model or artifact.operator not in self.operators:
            raise LifecycleError("Unsupported test artifact")
        raw = (artifact.directory / "artifact.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != artifact.digest:
            raise LifecycleError("Artifact content changed")

    def estimate(self, artifact):
        return {"host": 16, **{worker: 64 for worker in self.worker_ids}}

    def ack(self, phase, transaction, generation=None):
        self.log.append((phase, transaction["id"], generation.id if generation else None))
        if self.fail == phase:
            raise OSError("injected transport failure")
        values = [WorkerAck(worker, transaction["id"], generation.id if generation else None,
                            generation.artifact.digest if generation else None, phase)
                  for worker in self.worker_ids]
        if self.bad_ack == phase:
            values[-1] = WorkerAck(values[0].worker_id, "stale-transaction", None, None, phase)
        return values

    def reserve(self, transaction, generation):
        self.log.append(("reserve", transaction["id"], generation.id))
        if self.fail == "reserve":
            raise OSError("injected allocation failure")

    def prepare(self, transaction, generation):
        self.validate(generation.artifact)
        if self.prepare_entered is not None:
            self.prepare_entered.set()
            if not self.prepare_release.wait(3):
                raise AssertionError("test preparation was not released")
        self.live[generation.id] = generation.artifact.digest
        return self.ack("prepared", transaction, generation)

    def fence(self, transaction):
        self.fenced = True
        return self.ack("fenced", transaction)

    def commit(self, transaction, generation):
        if not self.fenced:
            raise AssertionError("commit before admission fence")
        if self.journal:
            state = json.loads(self.journal.read_text())
            if state["pending"]["phase"] not in ("committing", "resuming"):
                raise AssertionError("commit before durable decision")
        if transaction["operation"] == "deactivate":
            self.aliases.pop(transaction["alias"], None)
        else:
            self.aliases[transaction["alias"]] = generation.id
        return self.ack("committed", transaction, generation)

    def resume(self, transaction):
        acknowledgments = self.ack("resumed", transaction)
        self.fenced = False
        return acknowledgments

    def abort(self, transaction, generation):
        if not transaction["was_allocated"]:
            self.live.pop(generation.id, None)
        return self.ack("aborted", transaction, generation)

    def references(self, generation):
        return self.refs.get(generation.id, 0)

    def retire(self, transaction, generation):
        if self.references(generation):
            raise AssertionError("free while referenced")
        acknowledgments = self.ack("retired", transaction, generation)
        self.live.pop(generation.id, None)
        return acknowledgments

    def reconcile(self, transaction, snapshot):
        self.fenced = True
        self.live = {gid: row["artifact"]["digest"] for gid, row in snapshot["generations"].items()
                     if row["allocated"]}
        self.aliases = {name: row["generation"] for name, row in snapshot["aliases"].items()
                        if row["generation"] is not None}
        return self.ack("reconciled", transaction)


class TwoWorkerBackend(ContractBackend):
    identity = "test.distributed.v1"
    worker_ids = ("worker-a", "worker-b")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.managers = []

    def tearDown(self):
        for manager in self.managers:
            manager.close()

    def manager(self, backend=None, *, limit=256, state="state", generations=64):
        backend = backend or ContractBackend()
        limits = {"host": limit, **{worker: limit for worker in backend.worker_ids}}
        manager = LifecycleManager(self.root / state, backend=backend, stager=stage_fixture,
                                   admin_key=KEY, resource_limits=limits, max_generations=generations)
        backend.journal = manager.journal
        self.managers.append(manager)
        return manager, backend, manager.admin(KEY)

    def source(self, name="arbitrary-profile", *, content=1, operator="projection", model="unseen/model"):
        path = self.root / ("source-" + str(content) + ".json")
        path.write_text(json.dumps({"name": name, "content": content, "operator": operator, "model": model}))
        return path

    def activate(self, manager, admin, *, content=1, operator="projection", revision=0, alias="owner"):
        generation = manager.stage(admin, self.source(content=content, operator=operator))
        manager.activate(admin, alias, generation.id, expected_revision=revision)
        return generation

    def test_admin_is_separate_and_instance_bound(self):
        manager, _, admin = self.manager()
        with self.assertRaises(AdministrationDenied):
            manager.admin(b"a" * 32)
        with self.assertRaises(AdministrationDenied):
            manager.stage(None, self.source())
        other, _, _ = self.manager(state="other")
        with self.assertRaises(AdministrationDenied):
            other.stage(admin, self.source())
        self.assertNotIn(KEY.decode(), json.dumps(manager.status()))

    def test_journal_presence_is_captured_under_process_lock_and_read_only(self):
        actual_open = os.open
        observed = []
        def checked_open(path, flags, *args, **kwargs):
            if Path(path) == self.root / 'state' / 'lifecycle.json':
                fd = actual_open(self.root/'state/coordinator.lock', os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    observed.append(True)
                finally:
                    os.close(fd)
            return actual_open(path, flags, *args, **kwargs)
        with patch('owner_mode.profile_lifecycle.os.open', side_effect=checked_open):
            manager, _, _ = self.manager()
            self.assertFalse(manager.journal_was_present)
            with self.assertRaises(AttributeError):
                manager.journal_was_present = True
            manager.close()
            reopened, _, _ = self.manager()
            self.assertTrue(reopened.journal_was_present)
        self.assertEqual(observed, [True, True])

    def test_presence_fact_does_not_change_after_first_mutation(self):
        manager, _, admin = self.manager()
        self.assertFalse(manager.journal_was_present)
        self.activate(manager, admin)
        self.assertFalse(manager.journal_was_present)

    def test_arbitrary_models_profile_ids_and_operator_families(self):
        for backend_type in (ContractBackend, TwoWorkerBackend):
            with self.subTest(backend=backend_type.identity):
                manager, backend, admin = self.manager(backend_type(), state=backend_type.identity)
                first = self.activate(manager, admin)
                with manager.pin("owner") as lease:
                    self.assertEqual(lease.generation, first)
                second = self.activate(manager, admin, content=2, operator="native_matrix_override_v1", revision=1)
                self.assertNotEqual(first.artifact.digest, second.artifact.digest)
                self.assertEqual(manager.status()["aliases"]["owner"]["generation"], second.id)
                self.assertEqual(backend.aliases["owner"], second.id)

    def test_inflight_queue_cache_and_gpu_leases_pin_old_generation(self):
        manager, backend, admin = self.manager()
        first = self.activate(manager, admin)
        request = manager.pin("owner")
        queued = request.retain("queue")
        cache = request.retain("cache")
        gpu = request.retain("gpu")
        second = self.activate(manager, admin, content=2, revision=1)
        with manager.pin("owner") as now:
            self.assertEqual(now.generation.id, second.id)
        self.assertEqual(request.generation.id, first.id)
        for lease in (request, queued, cache):
            lease.close()
            with self.assertRaises(GenerationBusy):
                manager.retire(admin, first.id)
        gpu.close()
        backend.refs[first.id] = 1
        with self.assertRaises(GenerationBusy):
            manager.retire(admin, first.id)
        backend.refs[first.id] = 0
        manager.retire(admin, first.id)
        self.assertNotIn(first.id, backend.live)
        with self.assertRaises(GenerationBusy):
            request.retain()

    def test_expected_revision_prevents_racing_administrators(self):
        manager, _, admin = self.manager()
        self.activate(manager, admin)
        second = manager.stage(admin, self.source(content=2))
        with self.assertRaises(RevisionConflict):
            manager.activate(admin, "owner", second.id, expected_revision=0)
        self.assertEqual(manager.status()["aliases"]["owner"]["revision"], 1)

    def test_prepare_does_not_block_old_admission(self):
        manager, backend, admin = self.manager()
        first = self.activate(manager, admin)
        second = manager.stage(admin, self.source(content=2))
        backend.prepare_entered = threading.Event()
        backend.prepare_release = threading.Event()
        errors = []
        def update():
            try:
                manager.activate(admin, "owner", second.id, expected_revision=1)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=update)
        thread.start()
        self.assertTrue(backend.prepare_entered.wait(2))
        with manager.pin("owner") as lease:
            self.assertEqual(lease.generation.id, first.id)
        backend.prepare_release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_insufficient_overlap_budget_keeps_old_profile(self):
        manager, backend, admin = self.manager(limit=100)
        first = self.activate(manager, admin)
        second = manager.stage(admin, self.source(content=2))
        with self.assertRaises(ResourceLimit):
            manager.activate(admin, "owner", second.id, expected_revision=1)
        with manager.pin("owner") as lease:
            self.assertEqual(lease.generation.id, first.id)
        self.assertNotIn(second.id, backend.live)

    def test_precommit_failures_restore_previous_generation(self):
        for phase in ("reserve", "prepared", "fenced"):
            with self.subTest(phase=phase):
                manager, backend, admin = self.manager(state=phase)
                first = self.activate(manager, admin)
                second = manager.stage(admin, self.source(content=2))
                backend.fail = phase
                with self.assertRaises(LifecycleError):
                    manager.activate(admin, "owner", second.id, expected_revision=1)
                with manager.pin("owner") as lease:
                    self.assertEqual(lease.generation.id, first.id)
                self.assertFalse(manager.status()["admission_fenced"])

    def test_partial_commit_is_fenced_until_recovery(self):
        for phase in ("committed", "resumed"):
            with self.subTest(phase=phase):
                manager, backend, admin = self.manager(TwoWorkerBackend(), state=phase)
                self.activate(manager, admin)
                second = manager.stage(admin, self.source(content=2))
                backend.fail = phase
                with self.assertRaises(RecoveryRequired):
                    manager.activate(admin, "owner", second.id, expected_revision=1)
                with self.assertRaises(RecoveryRequired):
                    manager.pin("owner")
                backend.fail = None
                manager.recover(admin)
                with manager.pin("owner") as lease:
                    self.assertEqual(lease.generation.id, second.id)
                self.assertEqual(manager.status()["aliases"]["owner"]["revision"], 2)

    def test_stale_or_missing_worker_ack_is_never_committed(self):
        manager, backend, admin = self.manager(TwoWorkerBackend())
        first = self.activate(manager, admin)
        second = manager.stage(admin, self.source(content=2))
        backend.bad_ack = "prepared"
        with self.assertRaises(LifecycleError):
            manager.activate(admin, "owner", second.id, expected_revision=1)
        self.assertEqual(manager.status()["aliases"]["owner"]["generation"], first.id)

    def test_restart_reconciles_before_admission_and_keeps_worker_refs(self):
        manager, backend, admin = self.manager(TwoWorkerBackend())
        first = self.activate(manager, admin)
        second = self.activate(manager, admin, content=2, revision=1)
        backend.refs[first.id] = 1
        manager.close()
        restarted, _, permit = self.manager(backend)
        with self.assertRaises(RecoveryRequired):
            restarted.pin("owner")
        restarted.recover(permit)
        with restarted.pin("owner") as lease:
            self.assertEqual(lease.generation.id, second.id)
        with self.assertRaises(GenerationBusy):
            restarted.retire(permit, first.id)

    def test_restart_finishes_durable_commit_instead_of_rolling_it_back(self):
        manager, backend, admin = self.manager(TwoWorkerBackend())
        self.activate(manager, admin)
        replacement = manager.stage(admin, self.source(content=2))
        backend.fail = "committed"
        with self.assertRaises(RecoveryRequired):
            manager.activate(admin, "owner", replacement.id, expected_revision=1)
        manager.close()
        backend.fail = None
        restarted, _, permit = self.manager(backend)
        restarted.recover(permit)
        with restarted.pin("owner") as lease:
            self.assertEqual(lease.generation.id, replacement.id)
        self.assertEqual(restarted.status()["aliases"]["owner"]["revision"], 2)

    def test_recovery_rejects_missing_worker_and_tampered_artifact(self):
        manager, backend, admin = self.manager(TwoWorkerBackend())
        original = self.activate(manager, admin)
        manager.close()
        restarted, _, permit = self.manager(backend)
        backend.bad_ack = "reconciled"
        with self.assertRaises(RecoveryRequired):
            restarted.recover(permit)
        self.assertTrue(restarted.status()["admission_fenced"])
        restarted.close()
        (original.artifact.directory / "artifact.json").write_text("changed")
        with self.assertRaises(JournalError):
            self.manager(backend)

    def test_rollback_is_new_revision_using_old_immutable_content(self):
        manager, _, admin = self.manager()
        first = self.activate(manager, admin)
        self.activate(manager, admin, content=2, revision=1)
        result = manager.rollback(admin, "owner", first.id, expected_revision=2)
        self.assertEqual(result["revision"], 3)
        with manager.pin("owner") as lease:
            self.assertEqual(lease.generation.artifact.digest, first.artifact.digest)

    def test_journal_failure_never_silently_resumes(self):
        manager, _, admin = self.manager()
        first = self.activate(manager, admin)
        second = manager.stage(admin, self.source(content=2))
        with patch('owner_mode.profile_lifecycle.os.replace', side_effect=OSError("disk unavailable")):
            with self.assertRaises(JournalError):
                manager.activate(admin, "owner", second.id, expected_revision=1)
        with self.assertRaises(RecoveryRequired):
            manager.pin("owner")
        with self.assertRaises(RecoveryRequired):
            manager.recover(admin)
        self.assertEqual(first.artifact.name, "arbitrary-profile")

    def test_retirement_failure_recovers_without_freeing_live_generation(self):
        manager, backend, admin = self.manager()
        first = self.activate(manager, admin)
        second = self.activate(manager, admin, content=2, revision=1)
        backend.fail = "retired"
        with self.assertRaises(RecoveryRequired):
            manager.retire(admin, first.id)
        backend.fail = None
        manager.recover(admin)
        self.assertNotIn(first.id, backend.live)
        with manager.pin("owner") as lease:
            self.assertEqual(lease.generation.id, second.id)

    def test_unknown_generation_and_alias_rejected(self):
        manager, _, admin = self.manager()
        with self.assertRaises(LifecycleError):
            manager.pin("missing")
        with self.assertRaises(LifecycleError):
            manager.activate(admin, "owner", "g-missing", expected_revision=0)
        with self.assertRaises(LifecycleError):
            manager.stage(admin, "https://example.invalid/model")

    def test_journal_identity_and_permissions_are_enforced(self):
        manager, _, _ = self.manager()
        manager.close()
        with self.assertRaises(JournalError):
            self.manager(TwoWorkerBackend())
        manager.journal.chmod(0o644)
        with self.assertRaises(JournalError):
            self.manager()

    def test_one_coordinator_per_state(self):
        self.manager()
        with self.assertRaises(JournalError):
            self.manager()

    def test_stress_new_generations_and_refs_remain_bounded(self):
        manager, backend, admin = self.manager(TwoWorkerBackend(), generations=4)
        prior = None
        for revision in range(100):
            generation = self.activate(manager, admin, content=revision + 1, revision=revision)
            for _ in range(10):
                with manager.pin("owner") as lease:
                    self.assertEqual(lease.generation.id, generation.id)
            if prior is not None:
                manager.retire(admin, prior.id)
            prior = generation
        state = manager.status()
        self.assertEqual(len(state["generations"]), 1)
        self.assertEqual(len(backend.live), 1)
        self.assertEqual(state["aliases"]["owner"]["revision"], 100)
        self.assertNotIn(str(self.root), json.dumps(state))

    def test_real_profile_stager_keeps_validated_bytes_and_excludes_freeform(self):
        from test_profile_inspector import fixture
        source = fixture(self.root / "profile")
        destination = self.root / "staged"
        destination.mkdir(mode=0o700)
        artifact = stage_validated_profile(source, destination)
        self.assertEqual((source / "profile.safetensors").read_bytes(),
                         (destination / "profile.safetensors").read_bytes())
        self.assertEqual(artifact.model_id, "fixture/model")
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(artifact.record()))
        self.assertEqual(artifact.directory, destination)

    def test_deactivation_retains_admitted_generation_and_tombstone_revision(self):
        manager, backend, admin = self.manager(TwoWorkerBackend())
        generation = self.activate(manager, admin)
        lease = manager.pin('owner')
        result = manager.deactivate(admin, 'owner', expected_revision=1)
        self.assertEqual(result, {'alias':'owner', 'generation':None, 'revision':2})
        with self.assertRaises(LifecycleError):
            manager.pin('owner')
        self.assertNotIn('owner', backend.aliases)
        self.assertIn(generation.id, backend.live)
        with self.assertRaises(GenerationBusy):
            manager.retire(admin, generation.id)
        lease.close()
        manager.retire(admin, generation.id)
        with self.assertRaises(RevisionConflict):
            self.activate(manager, admin, content=2, revision=0)
        replacement = self.activate(manager, admin, content=3, revision=2)
        self.assertEqual(manager.status()['aliases']['owner'], {'generation':replacement.id, 'revision':3})

    def test_deactivation_survives_restart_after_old_generation_retired(self):
        manager, backend, admin = self.manager()
        generation = self.activate(manager, admin)
        manager.deactivate(admin, 'owner', expected_revision=1)
        manager.retire(admin, generation.id)
        manager.close()
        reopened, _, permit = self.manager()
        self.assertEqual(reopened.status()['aliases']['owner'], {'generation':None, 'revision':2})
        with self.assertRaises(LifecycleError):
            reopened.pin('owner')
        self.activate(reopened, permit, content=2, revision=2)

    def test_deactivation_failure_before_commit_retains_active_alias(self):
        manager, backend, admin = self.manager()
        generation = self.activate(manager, admin)
        backend.fail = 'fenced'
        with self.assertRaises(LifecycleError):
            manager.deactivate(admin, 'owner', expected_revision=1)
        with manager.pin('owner') as lease:
            self.assertEqual(lease.generation.id, generation.id)

    def test_deactivation_lost_commit_ack_recovers_tombstone(self):
        manager, backend, admin = self.manager()
        generation = self.activate(manager, admin)
        backend.fail = 'committed'
        with self.assertRaises(RecoveryRequired):
            manager.deactivate(admin, 'owner', expected_revision=1)
        backend.fail = None
        manager.recover(admin)
        self.assertEqual(manager.status()['aliases']['owner'], {'generation':None, 'revision':2})
        self.assertIn(generation.id, backend.live)
        self.assertNotIn('owner', backend.aliases)

    def test_alias_cannot_replace_immutable_generation_registry_entry(self):
        manager, _, admin = self.manager()
        first = self.activate(manager, admin)
        second = manager.stage(admin, self.source(content=2))
        with self.assertRaises(LifecycleError):
            manager.activate(admin, first.id, second.id, expected_revision=0)
        with manager.pin('owner') as lease:
            self.assertEqual(lease.generation.id, first.id)

    def test_alias_limit_matches_durable_inventory_limit(self):
        manager, _, admin = self.manager(generations=2)
        generation = self.activate(manager, admin)
        manager.activate(admin, 'second', generation.id, expected_revision=0)
        with self.assertRaises(ResourceLimit):
            manager.activate(admin, 'third', generation.id, expected_revision=0)
        self.assertEqual(set(manager.status()['aliases']), {'owner','second'})

    def test_file_fsync_failure_never_publishes_generation_or_journal(self):
        manager, _, admin = self.manager()
        before = manager.journal.read_bytes()
        original = os.fsync
        def fail_file(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError('Injected artifact fsync failure')
            return original(fd)
        with patch('owner_mode.profile_lifecycle.os.fsync', side_effect=fail_file):
            with self.assertRaises(LifecycleError):
                manager.stage(admin, self.source())
        self.assertEqual(manager.journal.read_bytes(), before)
        self.assertEqual(manager.status()['generations'], [])
        self.assertEqual(list(manager.artifacts.iterdir()), [])

    def test_snapshot_rejects_links_and_special_files_without_following(self):
        for kind in ('symlink', 'hardlink', 'fifo'):
            manager, _, admin = self.manager(state=kind)
            def unsafe(source, destination, kind=kind):
                artifact = stage_fixture(source, destination)
                if kind == 'symlink': (destination/'unexpected').symlink_to(self.root/'external')
                elif kind == 'hardlink': os.link(destination/'artifact.json', destination/'unexpected')
                else: os.mkfifo(destination/'unexpected')
                return artifact
            manager.stager = unsafe
            with self.assertRaises(LifecycleError):manager.stage(admin, self.source())
            self.assertEqual(manager.status()['generations'], [])

    def test_snapshot_fsync_covers_nested_files_and_enforces_bounds(self):
        manager, _, admin = self.manager()
        def nested(source, destination):
            artifact = stage_fixture(source, destination)
            (destination/'sub').mkdir(mode=0o700)
            (destination/'sub'/'extra').write_bytes(b'extra')
            return artifact
        manager.stager = nested
        original, seen = os.fsync, []
        def observe(fd):
            seen.append((stat.S_ISREG(os.fstat(fd).st_mode), os.readlink('/proc/self/fd/'+str(fd))))
            original(fd)
        with patch('owner_mode.profile_lifecycle.os.fsync', side_effect=observe):
            manager.stage(admin, self.source())
        self.assertTrue(any(regular and name.endswith('/artifact.json') for regular, name in seen))
        self.assertTrue(any(regular and name.endswith('/sub/extra') for regular, name in seen))
        for limit in ('MAX_STAGED_ENTRIES', 'MAX_STAGED_BYTES', 'MAX_STAGED_DEPTH'):
            with patch('owner_mode.profile_lifecycle.'+limit, 0):
                with self.assertRaises(ResourceLimit):manager.stage(admin, self.source(content=2))

    def test_sglang_digest_matches_canonical_cross_engine_fingerprint(self):
        import importlib.util
        from owner_mode.profiles import profile_fingerprint
        source=Path(__file__).resolve().parents[2]/'engines/common/sglang_lifecycle_support.py'
        spec=importlib.util.spec_from_file_location('sglang_fingerprint_fixture',source)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest,payload=b'{"name":"synthetic"}',b'synthetic-only-payload'
        hashes={'profile.json':hashlib.sha256(manifest).hexdigest(),
                'profile.safetensors':hashlib.sha256(payload).hexdigest()}
        self.assertEqual(module.profile_digest(manifest,payload),profile_fingerprint({'hashes':hashes}))


if __name__ == '__main__':
    unittest.main()
