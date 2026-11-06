# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Real local child processes for launch serialization and ambiguous RPC cleanup."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

import test_native_remote
from owner_mode import native_remote, native_cluster
from owner_mode.native_cluster import archive


class NativeUnitOwnership(unittest.TestCase):
    def setUp(self):
        test_native_remote.NativeRemote.setUp(self)
        self.path = Path(native_remote.stage(self.request, archive(self.files))['state'])
        self.request = {'state': str(self.path)}
        self.addCleanup(native_remote.stop, self.request)

    def ready(self):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            state = native_remote.state_of(self.path)
            if state.get('command_alive'):
                return state
            time.sleep(.02)
        self.fail('Owned command did not start')

    def test_two_real_starts_have_exactly_one_owner(self):
        entered, release = threading.Event(), threading.Event()
        real = subprocess.Popen
        children = []
        def delayed(*args, **kwargs):
            process = real(*args, **kwargs)
            children.append(process)
            entered.set()
            self.assertTrue(release.wait(5))
            return process
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            with mock.patch.object(native_remote.subprocess, 'Popen', side_effect=delayed):
                first = pool.submit(native_remote.start, self.request)
                self.assertTrue(entered.wait(5))
                second = pool.submit(native_remote.start, self.request)
                release.set()
                first.result(5)
                with self.assertRaisesRegex(ValueError, 'unverified|still running'):
                    second.result(5)
        self.assertEqual(len(children), 1)
        self.ready()
        inode = (self.path/'unit.lock').stat().st_ino
        self.assertEqual(native_remote.stop(self.request)['status'], 'stopped')
        self.assertIsNotNone(children[0].poll())
        self.assertEqual((self.path/'unit.lock').stat().st_ino, inode)

    def test_stop_during_real_spawn_cannot_leave_a_late_command(self):
        entered, release = threading.Event(), threading.Event()
        real = subprocess.Popen
        children = []
        def delayed(*args, **kwargs):
            child = real(*args, **kwargs)
            children.append(child)
            entered.set()
            self.assertTrue(release.wait(5))
            return child
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            with mock.patch.object(native_remote.subprocess, 'Popen', side_effect=delayed):
                started = pool.submit(native_remote.start, self.request)
                self.assertTrue(entered.wait(5))
                stopped = pool.submit(native_remote.stop, self.request)
                release.set()
                started.result(5)
                self.assertEqual(stopped.result(10)['status'], 'stopped')
        self.assertIsNotNone(children[0].poll())
        self.assertFalse(native_remote.state_of(self.path).get('command_alive'))

    def test_launcher_death_after_spawn_is_adopted_before_command(self):
        # Exit inside Popen's caller before it can publish the child identity.
        code = '''import importlib.util,os,subprocess,sys
spec=importlib.util.spec_from_file_location('remote',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
real=subprocess.Popen
def spawn(*a,**kw):
 p=real(*a,**kw)
 os._exit(77)
m.subprocess.Popen=spawn
m.start({'state':sys.argv[2]})
'''
        result = subprocess.run([sys.executable, '-c', code, native_remote.__file__, str(self.path)], timeout=10)
        self.assertEqual(result.returncode, 77)
        state = self.ready()
        intent = json.loads((self.path/'unit-launch.json').read_text())
        self.assertEqual(intent['phase'], 'owned')
        self.assertEqual(intent['supervisor'], state['supervisor'])
        with self.assertRaisesRegex(ValueError, 'still running'):
            native_remote.start(self.request)
        self.assertEqual(native_remote.stop(self.request)['status'], 'stopped')

    def test_cancel_before_delayed_rpc_proves_no_start_and_fences_nonce(self):
        request = {**self.request, 'nonce': 'a'*32}
        self.assertEqual(native_remote.stop(request)['status'], 'stopped')
        with self.assertRaisesRegex(ValueError, 'consumed or cancelled'):
            native_remote.start(request)
        self.assertFalse((self.path/'unit-live.json').exists())

    def test_ambiguous_intent_is_not_clean_or_restartable(self):
        supervisor = native_remote._supervisor(self.path)
        supervisor.save(self.path/'unit-launch.json', {
            'version': 1, 'nonce': 'b'*32, 'phase': 'pending',
            'launcher': supervisor.identity(os.getpid(), True), 'supervisor': None})
        self.assertFalse(native_remote.state_of(self.path)['all_children_reaped'])
        with self.assertRaisesRegex(ValueError, 'unverified'):
            native_remote.start(self.request)
        ticks = iter([0, 1, 2, 3, 61])
        clock = SimpleNamespace(monotonic=lambda: next(ticks, 61), sleep=lambda _: None)
        with mock.patch.object(native_remote, 'time', clock):
            self.assertEqual(native_remote.stop(self.request)['status'], 'incomplete')
        # Avoid another bounded wait in cleanup; no process was ever spawned.
        self._cleanups.pop()

    def test_popen_exception_alone_does_not_prove_no_child(self):
        with mock.patch.object(native_remote.subprocess, 'Popen', side_effect=OSError('test spawn failure')):
            with self.assertRaises(OSError):
                native_remote.start(self.request)
        self.assertFalse(native_remote.state_of(self.path)['all_children_reaped'])
        with self.assertRaisesRegex(ValueError, 'unverified'):
            native_remote.start(self.request)
        self._cleanups.pop()  # This injected failure created no child to reap.

    def test_parent_error_pipe_failure_after_real_fork_is_self_adopted(self):
        with mock.patch.object(subprocess.os, 'read', side_effect=OSError('injected parent pipe read')):
            with self.assertRaisesRegex(OSError, 'parent pipe'):
                native_remote.start(self.request)
        state = self.ready()
        pid = state['supervisor']['pid']
        self.assertEqual(native_remote.stop(self.request)['status'], 'stopped')
        # Popen raised before returning its handle; reap this test's own child.
        try:os.waitpid(pid, 0)
        except ChildProcessError:pass

    def test_old_completion_cannot_complete_new_pending_intent(self):
        native_remote.start(self.request)
        self.ready()
        native_remote.stop(self.request)
        supervisor = native_remote._supervisor(self.path)
        supervisor.save(self.path/'unit-launch.json', {'version': 1, 'nonce': 'c'*32,
                                                     'phase': 'pending', 'supervisor': None})
        self.assertFalse(native_remote.state_of(self.path)['all_children_reaped'])
        self._cleanups.pop()

    def test_lock_rejects_symlink_fifo_hardlink_or_public_permissions(self):
        lock = self.path/'unit.lock'
        target = self.root/'target'
        target.write_bytes(b'')
        target.chmod(0o600)
        fixtures = [lambda: lock.symlink_to(target), lambda: os.mkfifo(lock, 0o600),
                    lambda: os.link(target, lock), lambda: (lock.touch(mode=0o644), lock.chmod(0o644))]
        for create in fixtures:
            create()
            try:
                with self.assertRaises((OSError, ValueError)):
                    native_remote.state_of(self.path)
            finally:
                lock.unlink()

    def test_lost_first_start_ack_cleans_real_remote_child(self):
        root = self.root/'cluster'
        root.mkdir(mode=0o700)
        deployment = {'state': str(root), 'remote_units': [
            {'host': 'local', 'state': str(self.path), 'rank': 0}], 'base_url': 'unused'}
        observed = []
        def rpc(host, operation, **data):
            observed.append(operation)
            if operation == 'start':
                durable = list(root.glob('cluster-attempt-*.json'))
                self.assertEqual(len(durable), 1)
                self.assertEqual(json.loads(durable[0].read_text())['nonce'], data['nonce'])
                native_remote.start(data)
                self.ready()
                raise native_cluster.SetupError('Injected lost ACK')
            if operation == 'stop':
                return native_remote.stop(data)
            self.fail('Unexpected operation')
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc):
            with self.assertRaisesRegex(native_cluster.SetupError, 'lost ACK'):
                native_cluster.activate(deployment)
        self.assertEqual(observed, ['start', 'stop'])
        state = native_remote.state_of(self.path)
        self.assertTrue(state['all_children_reaped'])
        self.assertFalse(state['supervisor_alive'])

    def test_delayed_old_nonce_cannot_restart_or_stop_new_owner(self):
        first = {**self.request, 'nonce': 'd'*32}
        native_remote.start(first)
        self.ready()
        native_remote.stop(first)
        second = {**self.request, 'nonce': 'e'*32}
        native_remote.start(second)
        current = self.ready()
        with self.assertRaisesRegex(ValueError, 'consumed or cancelled'):
            native_remote.start(first)
        self.assertEqual(native_remote.stop(first)['status'], 'stopped')
        self.assertEqual(native_remote.state_of(self.path)['supervisor'], current['supervisor'])
        self.assertTrue(native_remote.state_of(self.path)['command_alive'])

    def test_reused_pid_without_cleanup_is_never_signalled_or_clean(self):
        supervisor = native_remote._supervisor(self.path)
        wrong = {**supervisor.identity(os.getpid(), True), 'start': '0'}
        nonce = 'f'*32
        supervisor.save(self.path/'unit-launch.json', {'version': 1, 'nonce': nonce,
                         'phase': 'owned', 'supervisor': wrong})
        supervisor.save(self.path/'unit-live.json', {'nonce': nonce, 'supervisor': wrong})
        supervisor.save(self.path/'supervisor.json', {'nonce': nonce, 'supervisor': wrong,
                         'status': 'running', 'all_children_reaped': False})
        ticks = iter([0, 1, 2, 3, 61])
        clock = SimpleNamespace(monotonic=lambda: next(ticks, 61), sleep=lambda _: None)
        with mock.patch.object(native_remote, 'time', clock), \
             mock.patch.object(supervisor, 'signal_identity') as signal:
            self.assertEqual(native_remote.stop(self.request)['status'], 'incomplete')
        signal.assert_not_called()
        self._cleanups.pop()

    def test_same_size_in_place_record_change_is_rejected(self):
        supervisor = native_remote._supervisor(self.path)
        path = self.path/'record.json'
        supervisor.save(path, {'value': 1})
        real = supervisor.os.fstat
        calls = 0
        def stat(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                content = path.read_bytes().replace(b'1', b'2')
                path.write_bytes(content)
            return real(fd)
        with mock.patch.object(supervisor.os, 'fstat', side_effect=stat), \
             mock.patch.object(supervisor.time, 'sleep') as pause:
            with self.assertRaisesRegex(ValueError, 'changed'):
                supervisor.read_unit_record(path)
        pause.assert_not_called()

    def test_stable_record_needs_no_retry_delay(self):
        supervisor = native_remote._supervisor(self.path)
        path = self.path/'record.json'
        supervisor.save(path, {'value': 1})
        with mock.patch.object(supervisor.time, 'sleep') as pause:
            self.assertEqual(supervisor.read_unit_record(path), {'value': 1})
        pause.assert_not_called()

    def test_atomic_record_replacement_is_retried(self):
        supervisor = native_remote._supervisor(self.path)
        path = self.path/'record.json'
        supervisor.save(path, {'value': 1})
        real = supervisor.os.fstat
        calls = 0
        def stat(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                supervisor.save(path, {'value': 2})
            return real(fd)
        with mock.patch.object(supervisor.os, 'fstat', side_effect=stat):
            self.assertEqual(supervisor.read_unit_record(path), {'value': 2})

    def test_detached_record_is_retried_when_path_lookup_still_observes_old_inode(self):
        supervisor = native_remote._supervisor(self.path)
        path = self.path/'record.json'
        real_fstat, real_lstat = supervisor.os.fstat, Path.lstat
        for detach_at in (1, 2):
            with self.subTest(detach_at=detach_at):
                supervisor.save(path, {'value': 1})
                stale = path.lstat()
                calls = 0
                def fstat(fd):
                    nonlocal calls
                    calls += 1
                    if calls == detach_at:
                        supervisor.save(path, {'value': 2})
                    return real_fstat(fd)
                def lstat(current, *args, **kwargs):
                    return stale if current == path else real_lstat(current, *args, **kwargs)
                with mock.patch.object(supervisor.os, 'fstat', side_effect=fstat), \
                     mock.patch.object(Path, 'lstat', new=lstat):
                    self.assertEqual(supervisor.read_unit_record(path), {'value': 2})

    def test_detachment_never_hides_unsafe_record_metadata(self):
        supervisor = native_remote._supervisor(self.path)
        path = self.path/'record.json'
        real = supervisor.os.fstat
        for detach_at in (1, 2):
            for field, value in (('st_mode', 0o100644), ('st_uid', os.getuid()+1),
                                 ('st_size', 1024*1024+1), ('st_mode', 0o010600)):
                with self.subTest(detach_at=detach_at, field=field, value=value):
                    supervisor.save(path, {'value': 1})
                    calls = 0
                    def fstat(fd):
                        nonlocal calls
                        calls += 1
                        current = real(fd)
                        if calls == detach_at:
                            fields = {name: getattr(current, name) for name in dir(current)
                                      if name.startswith('st_')}
                            fields.update({'st_nlink': 0, field: value})
                            return SimpleNamespace(**fields)
                        return current
                    with mock.patch.object(supervisor.os, 'fstat', side_effect=fstat), \
                         mock.patch.object(supervisor.time, 'sleep') as pause:
                        with self.assertRaisesRegex(ValueError, 'Unsafe native unit record'):
                            supervisor.read_unit_record(path)
                    pause.assert_not_called()

    def test_detached_record_churn_has_a_fixed_retry_bound(self):
        supervisor = native_remote._supervisor(self.path)
        path = self.path/'record.json'
        supervisor.save(path, {'value': 1})
        real = supervisor.os.fstat
        def detached(fd):
            current = real(fd)
            fields = {name: getattr(current, name) for name in dir(current)
                      if name.startswith('st_')}
            return SimpleNamespace(**{**fields, 'st_nlink': 0})
        with mock.patch.object(supervisor.os, 'fstat', side_effect=detached) as calls, \
             mock.patch.object(supervisor.time, 'sleep') as pause:
            with self.assertRaisesRegex(ValueError, 'changed repeatedly'):
                supervisor.read_unit_record(path)
        self.assertEqual(calls.call_count, 4)
        self.assertEqual(pause.call_args_list, [mock.call(.001)]*3)

    def test_repeated_cluster_activation_reuses_exact_owner_without_restart(self):
        root = self.root/'cluster'
        root.mkdir(mode=0o700)
        deployment = {'state': str(root), 'remote_units': [
            {'host': 'local', 'state': str(self.path), 'rank': 0}], 'base_url': 'unused',
            'model': 'fixture', 'profile': 'fixture', 'key_file': 'unused'}
        operations = []
        def rpc(host, operation, **data):
            operations.append(operation)
            if operation == 'start':
                result = native_remote.start(data)
                self.ready()
                return result
            if operation == 'status':
                return native_remote.state_of(data['state'])
            return native_remote.stop(data)
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc), \
             mock.patch.object(native_cluster, 'model_rows', return_value=[{'id': 'fixture'}]), \
             mock.patch.object(native_cluster, 'check_engine', return_value={'status': 'active'}):
            native_cluster.activate(deployment)
            first = self.ready()
            native_cluster.activate(deployment)
            self.assertEqual(self.ready()['supervisor'], first['supervisor'])
            self.assertEqual(operations.count('start'), 1)
            # A failed recheck must not tear down the owner it merely reused.
            with mock.patch.object(native_cluster, 'check_engine', side_effect=native_cluster.SetupError('probe failed')):
                with self.assertRaisesRegex(native_cluster.SetupError, 'probe failed'):
                    native_cluster.activate(deployment)
            self.assertTrue(native_remote.state_of(self.path)['command_alive'])
            self.assertNotIn('stop', operations)

    def test_upgraded_rpc_stops_legacy_supervisor_without_importing_old_helpers(self):
        import hashlib
        source = self.files['engine_supervisor.py'][0].replace(
            b"\nif __name__=='__main__':", b"\nunit_lock=None\nread_unit_record=None\nif __name__=='__main__':")
        files = {**self.files, 'engine_supervisor.py': (source, False)}
        request = {'parent': str(self.root), 'id': 'b'*20,
                   'files': {name: {'sha256': hashlib.sha256(data).hexdigest(),
                                   'bytes': len(data), 'executable': exe}
                             for name, (data, exe) in files.items()}}
        path = Path(native_remote.stage(request, archive(files))['state'])
        helpers = native_remote._supervisor(path)
        child = subprocess.Popen([sys.executable, str(path/'engine_supervisor.py'),
                                  '--state', str(path/'supervisor.json'), '--',
                                  sys.executable, '-c', 'import time;time.sleep(60)'])
        self.addCleanup(child.wait, timeout=5)
        self.addCleanup(native_remote.stop, {'state': str(path)})
        helpers.save(path/'unit-live.json', {'supervisor': helpers.identity(child.pid, True)})
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            if native_cluster.rpc('local', 'status', state=str(path)).get('command_alive'):
                break
            time.sleep(.02)
        else:
            self.fail('Legacy fixture did not become ready')
        with self.assertRaisesRegex(ValueError, 'predates'):
            native_remote.start({'state': str(path)})
        self.assertEqual(native_cluster.rpc('local', 'stop', state=str(path))['status'], 'stopped')
        child.wait(timeout=5)

    def cluster_fixture(self):
        root = self.root/'cluster'
        root.mkdir(mode=0o700)
        deployment = {'state': str(root), 'remote_units': [
            {'host': 'local', 'state': str(self.path), 'rank': 0}], 'base_url': 'unused',
            'model': 'fixture', 'profile': 'fixture', 'key_file': 'unused'}
        def rpc(host, operation, **data):
            if operation == 'start':
                result = native_remote.start(data)
                self.ready()
                return result
            if operation == 'status':
                return native_remote.state_of(data['state'])
            return native_remote.stop(data)
        return deployment, rpc

    def test_two_actual_cluster_activations_preserve_first_owner(self):
        deployment, rpc = self.cluster_fixture()
        checking, release = threading.Event(), threading.Event()
        def models(*args, **kwargs):
            checking.set()
            self.assertTrue(release.wait(5))
            return [{'id': 'fixture'}]
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc), \
             mock.patch.object(native_cluster, 'model_rows', side_effect=models), \
             mock.patch.object(native_cluster, 'check_engine', return_value={'status': 'active'}), \
             concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(native_cluster.activate, deployment)
            self.assertTrue(checking.wait(5))
            owner = self.ready()['supervisor']
            second = pool.submit(native_cluster.activate, deployment)
            with self.assertRaisesRegex(native_cluster.SetupError, 'unresolved'):
                second.result(5)
            self.assertEqual(self.ready()['supervisor'], owner)
            release.set()
            self.assertEqual(first.result(5)['status'], 'running')
        saved = json.loads((Path(deployment['state'])/'cluster-running.json').read_text())
        self.assertEqual(saved['status'], 'running')
        self.assertEqual(saved['units'][0]['receipt']['supervisor'], owner)

    def test_cluster_stop_during_dispatch_prevents_late_running_publication(self):
        deployment, real_rpc = self.cluster_fixture()
        dispatched, release, stopped = threading.Event(), threading.Event(), threading.Event()
        def rpc(host, operation, **data):
            result = real_rpc(host, operation, **data)
            if operation == 'start':
                dispatched.set()
                self.assertTrue(release.wait(5))
            return result
        def models(*args, **kwargs):
            self.assertTrue(stopped.wait(5))
            return [{'id': 'fixture'}]
        def stop():
            result = native_cluster.stop(deployment)
            stopped.set()
            return result
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc), \
             mock.patch.object(native_cluster, 'model_rows', side_effect=models), \
             mock.patch.object(native_cluster, 'check_engine', return_value={'status': 'active'}), \
             concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(native_cluster.activate, deployment)
            self.assertTrue(dispatched.wait(5))
            last = pool.submit(stop)
            release.set()
            self.assertEqual(last.result(10)['status'], 'stopped')
            with self.assertRaises(native_cluster.SetupError):
                first.result(10)
        self.assertFalse(native_remote.state_of(self.path)['supervisor_alive'])
        self.assertEqual(json.loads((Path(deployment['state'])/'cluster-running.json').read_text())['status'], 'stopped')

    def test_old_failed_attempt_cleanup_preserves_new_cluster_owner_and_receipt(self):
        deployment, rpc = self.cluster_fixture()
        old = {**self.request, 'nonce': '1'*32}
        native_remote.start(old)
        self.ready()
        native_remote.stop(old)
        new = {**self.request, 'nonce': '2'*32}
        native_remote.start(new)
        owner = self.ready()['supervisor']
        path = Path(deployment['state'])/'cluster-running.json'
        record = {'status': 'running', 'units': [
            {**deployment['remote_units'][0], 'nonce': new['nonce']}]}
        native_cluster._durable_json(path, record)
        old_deployment = {**deployment, 'remote_units': [
            {**deployment['remote_units'][0], 'nonce': old['nonce']}]}
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc):
            self.assertEqual(native_cluster._stop(old_deployment, scoped=True)['status'], 'stopped')
        self.assertEqual(json.loads(path.read_text()), record)
        self.assertEqual(self.ready()['supervisor'], owner)

    def test_old_cleanup_preserves_entirely_disjoint_new_unit_set(self):
        deployment, rpc = self.cluster_fixture()
        old = {**self.request, 'nonce': '3'*32}
        native_remote.start(old)
        self.ready()
        path = Path(deployment['state'])/'cluster-running.json'
        record = {'status': 'running', 'units': [
            {'host': 'different', 'state': '/different/owned/unit', 'nonce': '4'*32}]}
        native_cluster._durable_json(path, record)
        old_deployment = {**deployment, 'remote_units': [
            {**deployment['remote_units'][0], 'nonce': old['nonce']}]}
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc):
            self.assertEqual(native_cluster._stop(old_deployment, scoped=True)['status'], 'stopped')
        self.assertEqual(json.loads(path.read_text()), record)
        self.assertFalse(native_remote.state_of(self.path)['supervisor_alive'])

    def test_staged_absolute_working_directory_is_used_by_real_child(self):
        cwd = self.root/'working-directory'
        cwd.mkdir(mode=0o700)
        unit = {'env': {}, 'cwd': str(cwd), 'argv': [sys.executable, '-c',
                'from pathlib import Path; import time; Path("observed").write_text(str(Path.cwd())); time.sleep(60)']}
        files = {**self.files, 'unit.json': (json.dumps(unit).encode(), False)}
        request = {'parent': str(self.root), 'id': 'c'*20,
                   'files': native_cluster.file_records(files)}
        path = Path(native_remote.stage(request, archive(files))['state'])
        self.addCleanup(native_remote.stop, {'state': str(path)})
        native_remote.start({'state': str(path)})
        deadline = time.monotonic()+5
        while not (cwd/'observed').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual((cwd/'observed').read_text(), str(cwd))

    def test_relative_working_directory_is_rejected_before_launch_intent(self):
        files = {**self.files, 'unit.json': (json.dumps({'env': {}, 'cwd': 'relative',
                 'argv': [sys.executable, '-c', 'pass']}).encode(), False)}
        request = {'parent': str(self.root), 'id': 'd'*20,
                   'files': native_cluster.file_records(files)}
        path = Path(native_remote.stage(request, archive(files))['state'])
        with self.assertRaisesRegex(ValueError, 'absolute directory'):
            native_remote.start({'state': str(path)})
        self.assertFalse((path/'unit-launch.json').exists())

    def test_native_owner_replaced_during_health_check_is_not_published(self):
        deployment, rpc = self.cluster_fixture()
        replacement = {**self.request, 'nonce': '5'*32}
        def check(*args, **kwargs):
            native_remote.stop(self.request)
            native_remote.start(replacement)
            self.ready()
            return {'status': 'active'}
        with mock.patch.object(native_cluster, 'rpc', side_effect=rpc), \
             mock.patch.object(native_cluster, 'model_rows', return_value=[{'id': 'fixture'}]), \
             mock.patch.object(native_cluster, 'check_engine', side_effect=check):
            with self.assertRaisesRegex(native_cluster.SetupError, 'ownership changed'):
                native_cluster.activate(deployment)
        # Failed old-attempt cleanup is scoped and cannot kill the replacement.
        self.assertEqual(self.ready()['nonce'], replacement['nonce'])
        self.assertNotEqual(json.loads((Path(deployment['state'])/'cluster-running.json').read_text())['status'], 'running')

    def test_delayed_real_self_publication_waits_for_same_pending_owner(self):
        import signal
        deployment, _ = self.cluster_fixture()
        real = subprocess.Popen
        children = []
        pending = []
        def spawn(*args, **kwargs):
            child = real(*args, **kwargs)
            children.append(child)
            os.kill(child.pid, signal.SIGSTOP)
            return child
        def rpc(host, operation, **data):
            if operation == 'start':
                return native_remote.start(data)
            if operation == 'status':
                status = native_remote.state_of(data['state'])
                if status.get('status') == 'launch_unresolved':
                    pending.append(status)
                return status
            return native_remote.stop(data)
        def models(*args, **kwargs):
            for child in children:
                os.kill(child.pid, signal.SIGCONT)
            return [{'id': 'fixture'}]
        try:
            with mock.patch.object(native_remote.subprocess, 'Popen', side_effect=spawn), \
                 mock.patch.object(native_cluster, 'rpc', side_effect=rpc), \
                 mock.patch.object(native_cluster, 'model_rows', side_effect=models), \
                 mock.patch.object(native_cluster, 'check_engine', return_value={'status': 'active'}):
                self.assertEqual(native_cluster.activate(deployment, timeout=5)['status'], 'running')
            self.assertTrue(pending)
            self.assertEqual(pending[0]['supervisor']['pid'], children[0].pid)
            self.assertTrue(pending[0]['supervisor_alive'])
            self.assertFalse(pending[0]['command_alive'])
        finally:
            for child in children:
                if child.poll() is None:
                    os.kill(child.pid, signal.SIGCONT)

    def test_stop_cancels_paused_unpublished_supervisor_without_early_signal(self):
        import signal
        real = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = real(*args, **kwargs)
            children.append(child)
            os.kill(child.pid, signal.SIGSTOP)
            return child
        with mock.patch.object(native_remote.subprocess, 'Popen', side_effect=spawn):
            native_remote.start(self.request)
        self.assertEqual(native_remote.state_of(self.path)['status'], 'launch_unresolved')
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            stopped = pool.submit(native_remote.stop, self.request)
            try:
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    intent = json.loads((self.path/'unit-launch.json').read_text())
                    if intent['phase'] == 'cancelled':
                        break
                    time.sleep(.02)
                else:
                    self.fail('Stop did not cancel the unpublished launch')
            finally:
                os.kill(children[0].pid, signal.SIGCONT)
            self.assertEqual(stopped.result(10)['status'], 'stopped')
        state = native_remote.state_of(self.path)
        self.assertIsNone(state['command'])
        self.assertTrue(state['all_children_reaped'])

    def test_supervisor_import_does_not_require_posix_fcntl_or_pidfds(self):
        import builtins
        import signal
        import types
        source = Path(native_remote.__file__).with_name('engine_supervisor.py').read_text()
        module = types.ModuleType('portable_supervisor_fixture')
        actual_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name == 'fcntl':
                raise ModuleNotFoundError('no fcntl on this fixture platform')
            return actual_import(name, *args, **kwargs)
        with mock.patch.object(builtins, '__import__', side_effect=importing), \
             mock.patch.dict(os.__dict__), mock.patch.dict(signal.__dict__):
            os.__dict__.pop('pidfd_open', None)
            signal.__dict__.pop('pidfd_send_signal', None)
            exec(compile(source, '<supervisor-import-fixture>', 'exec'), module.__dict__)
            self.assertTrue(callable(module.identity))
            with self.assertRaisesRegex(RuntimeError, 'require POSIX flock'):
                with module.unit_lock(self.path):
                    self.fail('Unsupported local lock was admitted')
