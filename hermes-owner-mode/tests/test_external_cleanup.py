# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""External engine shutdown checks with a CPU-only synthetic resource."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode import backend_setup as backend
from owner_mode import engine_setup as engine
from owner_mode import native_remote as units


SCHEMA = 'flashalign.external-cleanup.v1'
HELPER = '''import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--state-dir');p.add_argument('--deadline-monotonic');a=p.parse_args()
path=Path(a.state_dir)/'external.json';value=json.loads(path.read_text())
if value.get('hang'):
 ready=Path(a.state_dir)/'child-ready'
 code="import signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text('ready');time.sleep(30)"
 child=subprocess.Popen([sys.executable,'-c',code,str(ready)],start_new_session=True)
 while not ready.exists():time.sleep(.01)
 parts=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()
 (Path(a.state_dir)/'child-identity.json').write_text(json.dumps({'pid':child.pid,'start':parts[19],'uid':os.getuid()}))
 while True:time.sleep(1)
if value.get('allow_stop'):value['stopped']=True;path.write_text(json.dumps(value))
if value.get('invalid_json'):print('synthetic-private-output')
else:print(json.dumps({'schema':'flashalign.external-cleanup.v1','status':'stopped' if value['stopped'] else 'incomplete','external_resources_stopped':value['stopped']}))
'''


class ExternalCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state, self.source = self.root/'state', self.root/'integration'
        self.state.mkdir(mode=0o700); self.source.mkdir(mode=0o700)
        self.script = self.source/'cleanup.py'; self.script.write_text(HELPER)
        self.hash = hashlib.sha256(self.script.read_bytes()).hexdigest()
        self.external = self.state/'external.json'
        self.external.write_text(json.dumps({'allow_stop': True, 'stopped': False}))
        self.deployment = {'state': str(self.state), 'source': str(self.source),
            'python': sys.executable, 'container_runtime': 'synthetic-daemon',
            'integration_files': {'cleanup.py': self.hash},
            'external_cleanup': {'schema': SCHEMA, 'script': 'cleanup.py',
                                 'script_sha256': self.hash, 'timeout_seconds': 35}}
        self.nonce = 'a'*32
        self.addCleanup(self.temporary.cleanup)

    def resource(self, **values):
        current = json.loads(self.external.read_text()); current.update(values)
        self.external.write_text(json.dumps(current))

    def attempt(self, *, phase='running', marker=True):
        unit = self.state/'launches'/self.nonce/'unit-0'
        engine._unit_supervisor.save(self.state/'activation.json',
            {'version': 1, 'nonce': self.nonce, 'phase': phase, 'units': [str(unit)]})
        if marker:
            engine._unit_supervisor.save(self.state/'running.json',
                {'version': 2, 'nonce': self.nonce, 'processes': [], 'supervisors': [], 'owned_members': []})

    @contextmanager
    def os_reaped(self):
        real_stop, real_state = units.stop, units.state_of
        is_cleanup = lambda path: Path(path).is_relative_to(self.state/'external-cleanup-units')
        def stop_unit(request):
            return (real_stop(request) if is_cleanup(request['state']) else
                    {'status': 'stopped', 'all_children_reaped': True})
        def state_of(path, **kwargs):
            return real_state(path, **kwargs) if is_cleanup(path) else {}
        with (mock.patch.object(units, 'stop', side_effect=stop_unit) as stop,
              mock.patch.object(units, 'state_of', side_effect=state_of)):
            yield stop

    def test_process_reaping_cannot_publish_stop_before_external_verification(self):
        self.attempt(); self.resource(allow_stop=False)
        with self.os_reaped(): result = engine.stop_engine(self.deployment)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(engine._activation(self.state)['phase'], 'stopping')
        self.assertTrue((self.state/'running.json').exists())
        self.assertFalse(json.loads(self.external.read_text())['stopped'])

    def test_retry_confirms_external_stop_before_removing_marker(self):
        self.attempt(); self.resource(allow_stop=False)
        with self.os_reaped():
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
            self.resource(allow_stop=True)
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
        self.assertEqual(engine._activation(self.state)['phase'], 'stopped')
        self.assertFalse((self.state/'running.json').exists())
        self.assertTrue(json.loads(self.external.read_text())['stopped'])

    def test_previously_stopped_phase_does_not_skip_external_verification(self):
        self.attempt(phase='stopped'); self.resource(allow_stop=False)
        with self.os_reaped(): self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
        self.assertTrue((self.state/'running.json').exists())
        self.assertEqual(engine._activation(self.state)['phase'], 'stopping')

    def test_missing_legacy_running_marker_does_not_mean_external_resource_stopped(self):
        self.resource(allow_stop=False)
        result = engine.stop_engine(self.deployment)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(engine._activation(self.state)['phase'], 'stopping')

    def test_superseded_nonce_cannot_run_cleanup_for_a_new_attempt(self):
        self.attempt()
        with mock.patch.object(engine.subprocess, 'run') as run:
            self.assertEqual(engine._stop_local_attempt(self.deployment, 'b'*32),
                             {'status': 'stopped', 'superseded': True})
        run.assert_not_called()
        self.assertEqual(engine._activation(self.state)['phase'], 'running')

    def test_invalid_helper_output_is_incomplete_and_not_copied_to_receipts(self):
        self.attempt(); self.resource(invalid_json=True)
        with self.os_reaped(): self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
        self.assertNotIn('synthetic-private-output', (self.state/'external-cleanup.json').read_text())
        self.assertTrue((self.state/'running.json').exists())

    def test_hanging_helper_and_escaped_child_are_reaped_within_total_budget(self):
        self.attempt(); self.resource(hang=True)
        self.deployment['external_cleanup']['timeout_seconds'] = 12
        def reap_fixture():
            for path in (self.state/'external-cleanup-units').glob('*'):
                units.stop({'state': str(path), 'timeout_seconds': 15})
        self.addCleanup(reap_fixture)
        started = time.monotonic()
        with self.os_reaped(): self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
        self.assertLess(time.monotonic()-started, 15)
        self.assertTrue((self.state/'running.json').exists())
        child = json.loads((self.state/'child-identity.json').read_text())
        self.assertNotEqual(engine._unit_supervisor.identity(child['pid'], True), child)
        for path in (self.state/'external-cleanup-units').iterdir():
            row = units.state_of(path)
            self.assertTrue(row['all_children_reaped'])
            self.assertFalse(row['supervisor_alive'])

    def test_helper_has_only_remaining_budget_and_runs_after_unit_reap(self):
        self.attempt()
        with self.os_reaped() as units_stop:
            real_start = units.start
            def start(request):
                self.assertEqual(units_stop.call_count, 1)
                command = json.loads((Path(request['state'])/'unit.json').read_text())['argv']
                self.assertLessEqual(float(command[-1]), time.monotonic()+35)
                self.assertEqual(engine._activation(self.state)['phase'], 'stopping')
                return real_start(request)
            with mock.patch.object(units, 'start', side_effect=start):
                self.assertEqual(engine.stop_engine(self.deployment)['status'], 'stopped')
            self.assertLessEqual(units_stop.call_args_list[0].args[0]['timeout_seconds'], 25)

    def test_changed_helper_is_rejected_without_executing_or_reaping(self):
        self.attempt(); self.script.write_text(HELPER+'\n# Changed\n')
        with self.os_reaped() as stop, mock.patch.object(engine.subprocess, 'run') as run:
            self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
        stop.assert_not_called(); run.assert_not_called()

    def test_external_runtime_without_cleanup_contract_cannot_claim_stopped(self):
        self.attempt(); del self.deployment['external_cleanup']
        self.assertEqual(engine.stop_engine(self.deployment)['status'], 'incomplete')
        self.assertEqual(engine._activation(self.state)['phase'], 'running')
        self.assertTrue((self.state/'running.json').exists())

    def test_backend_preparation_preserves_the_declared_hook_and_staged_hash(self):
        metadata = {'external_cleanup': {'schema': SCHEMA, 'script': 'cleanup.py', 'timeout_seconds': 35}}
        result = backend._prepare_external_cleanup(metadata, self.deployment)
        self.assertEqual(result['external_cleanup'], self.deployment['external_cleanup'])

    def test_foreign_or_malformed_helper_declarations_fail_closed(self):
        for change in ({'script': '../cleanup.py'}, {'script': '/cleanup.py'},
                       {'schema': 'unknown'}, {'timeout_seconds': True}, {'script_sha256': '0'*64}):
            with self.subTest(change=change):
                deployment = {**self.deployment, 'external_cleanup': {**self.deployment['external_cleanup'], **change}}
                with self.assertRaises((ValueError, engine.SetupError)):
                    engine._external_cleanup_contract(deployment)

    def test_real_managed_cpu_unit_stops_before_external_helper_and_retry_succeeds(self):
        deployment = {**self.deployment, 'model': 'fixture', 'profile': 'fixture',
            'base_url': 'http://127.0.0.1:0/v1', 'key_file': str(self.root/'unused'),
            'install_dependencies': False,
            'commands': [{'argv': [sys.executable, '-c', 'import time; time.sleep(30)'], 'env': {}}]}
        self.addCleanup(lambda: engine.stop_engine(deployment))
        with (mock.patch.object(engine, 'model_rows', return_value=[{'id': 'fixture'}]),
              mock.patch.object(engine, 'check_engine', return_value={'status': 'active'})):
            self.assertEqual(engine.activate_engine(deployment, timeout=5)['status'], 'running')
        self.resource(allow_stop=False)
        self.assertEqual(engine.stop_engine(deployment)['status'], 'incomplete')
        attempt = engine._activation(self.state)
        self.assertTrue(all(row.get('all_children_reaped') and not row.get('supervisor_alive')
                            for row in engine._unit_states(attempt)))
        self.assertFalse(json.loads(self.external.read_text())['stopped'])
        self.resource(allow_stop=True)
        self.assertEqual(engine.stop_engine(deployment)['status'], 'stopped')
        self.assertTrue(json.loads(self.external.read_text())['stopped'])
