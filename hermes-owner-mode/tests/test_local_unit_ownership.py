# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real local launch/cancellation regressions; no models, GPUs or network."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode import engine_setup as engine
from owner_mode import native_remote as units
from owner_mode import engine_supervisor as supervisor
from owner_mode.setup_runtime import SetupError


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError('Timed out waiting for owned fixture')


class LocalUnitOwnership(unittest.TestCase):
    def setUp(self):
        supervisor.set_subreaper()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        state = self.root/'state'
        state.mkdir(mode=0o700)
        self.deployment = {'state': str(state), 'source': str(self.root),
            'base_url': 'http://127.0.0.1:0/v1', 'model': 'fixture',
            'profile': 'fixture', 'key_file': str(self.root/'unused'),
            'install_dependencies': False,
            'commands': [{'argv': [sys.executable, '-c', 'import time; time.sleep(30)'], 'env': {}}]}
        self.state = state
        self.threads = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        engine.stop_engine(self.deployment)
        # A regression must not leave a failed test's bounded children behind.
        roots = []
        for path in self.state.glob('launches/*/unit-*/unit-live.json'):
            value = json.loads(path.read_text())
            roots.append(value['supervisor'])
        rows = supervisor.descendants(roots)
        for row in rows:
            if row not in roots:
                supervisor.signal_identity(row, signal.SIGTERM)
        for row in roots:
            supervisor.signal_identity(row, signal.SIGTERM)
        end = time.monotonic()+8
        while time.monotonic() < end and any(supervisor.identity(row['pid']) == row for row in rows):
            time.sleep(.02)
        for thread in self.threads:
            thread.join(5)
        for row in roots:
            try:
                os.waitpid(row['pid'], os.WNOHANG)
            except ChildProcessError:
                pass
        self.temporary.cleanup()

    def start_thread(self, target):
        outcomes = []
        def run():
            try:
                outcomes.append(target())
            except BaseException as error:
                outcomes.append(error)
        thread = threading.Thread(target=run)
        self.threads.append(thread)
        thread.start()
        return thread, outcomes

    def health(self):
        return (mock.patch.object(engine, 'model_rows', return_value=[{'id': 'fixture'}]),
                mock.patch.object(engine, 'check_engine', return_value={'status': 'active'}))

    def test_concurrent_start_has_one_real_supervisor(self):
        entered = threading.Event(); release = threading.Event()
        original = units.start
        def start(request):
            entered.set()
            if not release.wait(5): raise AssertionError('Missing release')
            return original(request)
        first, second = self.health()
        with first, second, mock.patch.object(units, 'start', side_effect=start) as starts:
            thread, outcomes = self.start_thread(lambda: engine.activate_engine(self.deployment, timeout=5))
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(SetupError, 'pending'):
                engine.activate_engine(self.deployment, timeout=5)
            release.set(); thread.join(8)
        self.assertEqual(starts.call_count, 1)
        self.assertEqual(outcomes[0]['status'], 'running')
        attempt = engine._activation(self.state)
        self.assertEqual(len(attempt['units']), 1)
        self.assertEqual(len([row for row in engine._unit_states(attempt) if row['command_alive']]), 1)

    def test_stop_before_unit_start_fences_delayed_start(self):
        entered = threading.Event(); release = threading.Event()
        original = units.start
        def start(request):
            entered.set()
            if not release.wait(5): raise AssertionError('Missing release')
            return original(request)
        with mock.patch.object(units, 'start', side_effect=start):
            thread, outcomes = self.start_thread(lambda: engine.activate_engine(self.deployment, timeout=5))
            self.assertTrue(entered.wait(5))
            self.assertFalse((self.state/'running.json').exists())
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
            release.set(); thread.join(8)
        self.assertIsInstance(outcomes[0], Exception)
        attempt = engine._activation(self.state)
        self.assertEqual(attempt['phase'], 'stopped')
        self.assertTrue(all(not row.get('supervisor_alive') for row in engine._unit_states(attempt)))
        self.assertFalse(list(self.state.glob('launches/*/unit-*/unit-live.json')))

    def test_controller_loss_before_running_receipt_is_recoverable(self):
        boundary = self.root/'spawn-boundary.json'
        controller = os.fork()
        if controller == 0:
            original = subprocess.Popen
            def start(*args, **kwargs):
                process = original(*args, **kwargs)
                supervisor.save(boundary, {'pid': process.pid})
                time.sleep(30)
                return process
            try:
                with mock.patch.object(units.subprocess, 'Popen', side_effect=start):
                    engine.activate_engine(self.deployment, timeout=30)
            finally:
                os._exit(2)
        identity = wait_for(lambda: supervisor.identity(controller))
        try:
            wait_for(lambda: boundary.exists())
            self.assertFalse((self.state/'running.json').exists())
            supervisor.signal_identity(identity, signal.SIGKILL)
            os.waitpid(controller, 0)
            wait_for(lambda: bool(list(self.state.glob('launches/*/unit-*/unit-live.json'))))
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
            attempt = engine._activation(self.state)
            self.assertEqual(attempt['phase'], 'stopped')
            self.assertTrue(all(row.get('all_children_reaped') is True
                                and not row.get('supervisor_alive') for row in engine._unit_states(attempt)))
        finally:
            supervisor.signal_identity(identity, signal.SIGKILL)
            try: os.waitpid(controller, 0)
            except ChildProcessError: pass

    def test_stale_health_cannot_publish_or_stop_new_attempt(self):
        entered = threading.Event(); release = threading.Event(); calls = []
        def check(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                if not release.wait(8): raise AssertionError('Missing release')
            return {'status': 'active'}
        with (mock.patch.object(engine, 'model_rows', return_value=[{'id': 'fixture'}]),
              mock.patch.object(engine, 'check_engine', side_effect=check)):
            thread, outcomes = self.start_thread(lambda: engine.activate_engine(self.deployment, timeout=10))
            self.assertTrue(entered.wait(5))
            old = engine._activation(self.state)['nonce']
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
            fresh = engine.activate_engine({**self.deployment, 'fixture_tag': 'new'}, timeout=5)
            new = engine._activation(self.state)['nonce']
            self.assertNotEqual(old, new)
            release.set(); thread.join(8)
        self.assertIsInstance(outcomes[0], SetupError)
        self.assertEqual(fresh['fixture_tag'], 'new')
        self.assertEqual(engine._activation(self.state)['nonce'], new)
        self.assertEqual(json.loads((self.state/'running.json').read_text())['nonce'], new)
        self.assertEqual(json.loads((self.state/'deployment.json').read_text())['fixture_tag'], 'new')
        self.assertTrue(engine._units_running(engine._unit_states(engine._activation(self.state))))

    def test_healthy_repeat_reuses_the_same_real_unit(self):
        first, second = self.health()
        with first, second:
            engine.activate_engine(self.deployment, timeout=5)
            nonce = engine._activation(self.state)['nonce']
            identities = [row['supervisor'] for row in engine._unit_states(engine._activation(self.state))]
            result = engine.activate_engine(self.deployment, timeout=5)
        self.assertEqual(result['status'], 'running')
        self.assertEqual(engine._activation(self.state)['nonce'], nonce)
        self.assertEqual([row['supervisor'] for row in engine._unit_states(engine._activation(self.state))], identities)

    def test_reuse_health_failure_preserves_the_existing_owner(self):
        first, second = self.health()
        with first, second:
            engine.activate_engine(self.deployment, timeout=5)
        original = engine._activation(self.state)
        with (mock.patch.object(engine, 'model_rows', return_value=[{'id': 'fixture'}]),
              mock.patch.object(engine, 'check_engine', side_effect=SetupError('health unavailable'))):
            with self.assertRaisesRegex(SetupError, 'health unavailable'):
                engine.activate_engine(self.deployment, timeout=5)
        self.assertEqual(engine._activation(self.state)['nonce'], original['nonce'])
        self.assertTrue(engine._units_running(engine._unit_states(original)))

    def test_unit_nonce_and_birth_mismatch_cannot_publish(self):
        first, second = self.health()
        with first, second:
            engine.activate_engine(self.deployment, timeout=5)
        attempt = engine._activation(self.state)
        before = (self.state/'running.json').read_bytes()
        actual = units.state_of(attempt['units'][0])
        for replacement in ({**actual, 'nonce': '0'*32}, {**actual, 'nonce': None},
                            {**actual, 'supervisor': {**actual['supervisor'], 'start': '0'}}):
            with mock.patch.object(units, 'state_of', return_value=replacement):
                with self.assertRaisesRegex(SetupError, 'ownership differs'):
                    engine._publish_local(self.deployment, attempt['nonce'])
            self.assertEqual((self.state/'running.json').read_bytes(), before)
        self.assertTrue(engine._units_running(engine._unit_states(attempt)))

    def test_repeated_start_stop_releases_owned_popen_handles(self):
        before = set(engine._children)
        before_units = set(units._children)
        first, second = self.health()
        with first, second:
            for _ in range(4):
                engine.activate_engine(self.deployment, timeout=5)
                self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
                self.assertEqual(set(engine._children), before)
                self.assertEqual(set(units._children), before_units)

    def test_failed_popen_ack_still_cleans_real_started_unit(self):
        original = units.start
        def failed_ack(request):
            original(request)
            raise OSError('lost local acknowledgement')
        with mock.patch.object(units, 'start', side_effect=failed_ack):
            with self.assertRaisesRegex(SetupError, 'launch failed'):
                engine.activate_engine(self.deployment, timeout=5)
        attempt = engine._activation(self.state)
        self.assertEqual(attempt['phase'], 'stopped')
        self.assertTrue(all(row.get('all_children_reaped') is True and not row.get('supervisor_alive')
                            for row in engine._unit_states(attempt)))

    def test_two_units_keep_live_anchor_until_normal_stop_then_clean_orphan(self):
        self.deployment['commands'] *= 2
        first, second = self.health()
        with first, second:
            engine.activate_engine(self.deployment, timeout=5)
        attempt = engine._activation(self.state)
        rows = engine._unit_states(attempt)
        dead, live = rows[0]['supervisor'], rows[1]['supervisor']
        orphan = rows[0]['command']
        supervisor.signal_identity(dead, signal.SIGKILL)
        engine._children[dead['pid']].wait(timeout=5)
        # A later status publication must not discard the earlier orphan witness.
        engine._publish_local(self.deployment, attempt['nonce'])
        record = json.loads((self.state/'running.json').read_text())
        self.assertIn(orphan, record['owned_members'])
        original = engine._signal_identity
        observed = []
        def signal_owned(row, signum):
            observed.append((row, supervisor.identity(live['pid']) == live))
            return original(row, signum)
        with mock.patch.object(engine, '_signal_identity', side_effect=signal_owned):
            result = engine.stop_engine(self.deployment)
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(observed)
        self.assertTrue(all(not live_anchor for _, live_anchor in observed))
        self.assertIsNone(supervisor.identity(orphan['pid']))
        self.assertIsNone(supervisor.identity(live['pid']))

    def test_healthy_legacy_receipt_is_reused_without_new_spawn(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        owner = wait_for(lambda: supervisor.identity(process.pid))
        supervisor.save(self.state/'running.json', {'processes': [owner], 'owned_members': [owner]})
        first, second = self.health()
        try:
            with first, second, mock.patch.object(units, 'start') as start:
                self.assertEqual(engine.activate_engine(self.deployment, timeout=5)['status'], 'running')
            start.assert_not_called()
            self.assertFalse((self.state/'activation.json').exists())
        finally:
            supervisor.signal_identity(owner, signal.SIGTERM)
            process.wait(timeout=5)

    def test_pending_or_corrupt_ownership_never_reports_clean_stop(self):
        (self.state/'activation.json').write_text('{}')
        (self.state/'activation.json').chmod(0o600)
        self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
        with self.assertRaises(SetupError):
            engine.activate_engine(self.deployment, timeout=1)

    def test_dependency_installation_is_preparation_only(self):
        with mock.patch.object(units.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(SetupError, 'during preparation'):
                engine.activate_engine({**self.deployment, 'install_dependencies': True}, timeout=1)
        spawn.assert_not_called()
        self.assertFalse((self.state/'activation.json').exists())

    def test_stopped_phase_recovers_interrupted_marker_removal(self):
        first, second = self.health()
        with first, second:
            engine.activate_engine(self.deployment, timeout=5)
            old = json.loads((self.state/'running.json').read_text())
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
            # Simulate a crash after durable stopped publication, before unlink.
            supervisor.save(self.state/'running.json', old)
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
            self.assertFalse((self.state/'running.json').exists())
            supervisor.save(self.state/'running.json', old)
            result = engine.activate_engine(self.deployment, timeout=5)
        self.assertEqual(result['status'], 'running')
        self.assertNotEqual(engine._activation(self.state)['nonce'], old['nonce'])


if __name__ == '__main__':
    unittest.main()
