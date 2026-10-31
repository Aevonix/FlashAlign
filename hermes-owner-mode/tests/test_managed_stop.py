# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real subprocess/socket shutdown regressions; no model or external network."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.engine_setup import (process_identity, private_json, stop_engine,
                                     _check_available_port, _signal_identity, activate_engine)
from owner_mode.setup_runtime import SetupError


class ManagedShutdown(unittest.TestCase):
    def listener_program(self, ready, parent_exit=None):
        child_code = (
            "import json,os,signal,socket,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
            "s.bind(('127.0.0.1',0)); s.listen(2); "
            f"Path({str(ready)!r}).write_text(json.dumps({{'pid':os.getpid(),'port':s.getsockname()[1]}})); "
            "time.sleep(300)"
        )
        parent_code = (
            "import signal,subprocess,sys,time; "
            "signal.signal(signal.SIGTERM,lambda *_: sys.exit(0)); "
            f"subprocess.Popen([sys.executable,'-c',{child_code!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            f"time.sleep({parent_exit if parent_exit is not None else 300})"
        )
        return parent_code

    def listener_group(self, root):
        ready = root/'ready.json'
        parent_code = self.listener_program(ready)
        parent = subprocess.Popen([sys.executable, '-c', parent_code], start_new_session=True,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.monotonic()+5
        while not ready.exists() and time.monotonic() < end:
            time.sleep(.02)
        self.assertTrue(ready.exists())
        info = json.loads(ready.read_text())
        child = process_identity(info['pid'])
        leader = process_identity(parent.pid)
        self.assertIsNotNone(child);self.assertIsNotNone(leader)
        def cleanup():
            for record in (child, leader):
                if process_identity(record['pid']) == record:
                    os.kill(record['pid'], signal.SIGKILL)
            parent.wait(timeout=5)
        self.addCleanup(cleanup)
        state = root/'state';state.mkdir()
        private_json(state/'running.json', {'processes': [leader]})
        return parent, child, info['port'], state

    def deployment(self, root, *, parent_exit=None):
        state=root/'state';state.mkdir()
        ready=root/'ready.json'
        deployment={'state':str(state),'source':str(root),'base_url':'http://127.0.0.1:0/v1',
            'model':'fixture','profile':'fixture','key_file':str(root/'unused-key'),
            'install_dependencies':False,'commands':[{'argv':[sys.executable,'-c',
                self.listener_program(ready,parent_exit)],'env':{}}]}
        def cleanup():
            if ready.exists():
                child=json.loads(ready.read_text());record=process_identity(child['pid'])
                if record is not None:_signal_identity(record,signal.SIGKILL)
            stop_engine(deployment)
        self.addCleanup(cleanup)
        return deployment,ready

    def test_successful_start_records_members_before_later_leader_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            deployment,ready=self.deployment(Path(temporary))
            def models(*_, **_kwargs):
                end=time.monotonic()+5
                while not ready.exists() and time.monotonic()<end:time.sleep(.01)
                return [{'id':'fixture'}]
            with (mock.patch('owner_mode.engine_setup.model_rows',side_effect=models),
                  mock.patch('owner_mode.engine_setup.check_engine',return_value={'status':'active'})):
                activate_engine(deployment,timeout=5)
            state=Path(deployment['state']);record=json.loads((state/'running.json').read_text())
            child=json.loads(ready.read_text())
            self.assertIn(child['pid'],[member['pid'] for member in record['owned_members']])
            _signal_identity(record['processes'][0],signal.SIGTERM)
            end=time.monotonic()+5
            while process_identity(record['processes'][0]['pid']) and time.monotonic()<end:time.sleep(.01)
            self.assertEqual(stop_engine(deployment)['status'],'stopped')
            self.assertIsNone(process_identity(child['pid']))

    def test_failed_start_cleans_recorded_survivor_after_parent_exits(self):
        with tempfile.TemporaryDirectory() as temporary:
            deployment,ready=self.deployment(Path(temporary),parent_exit=1.2)
            with mock.patch('owner_mode.engine_setup.model_rows',side_effect=SetupError('not ready')):
                with self.assertRaisesRegex(SetupError,'exited during startup'):
                    activate_engine(deployment,timeout=5)
            self.assertTrue(ready.exists())
            child=json.loads(ready.read_text())
            self.assertIsNone(process_identity(child['pid']))
            self.assertFalse((Path(deployment['state'])/'running.json').exists())

    def test_missing_pidfd_fails_before_start_without_pid_kill_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            deployment,_=self.deployment(Path(temporary))
            with (mock.patch('owner_mode.engine_setup.os.pidfd_open',None),
                  mock.patch('owner_mode.engine_setup.subprocess.Popen') as start,
                  mock.patch('owner_mode.engine_setup.os.kill') as kill):
                with self.assertRaisesRegex(SetupError,'pidfd'):
                    activate_engine(deployment)
                self.assertFalse(_signal_identity(process_identity(os.getpid()),signal.SIGTERM))
            start.assert_not_called();kill.assert_not_called()

    def test_incomplete_stop_cli_exits_with_failure_and_keeps_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'deployment.json';private_json(path,{'state':temporary})
            with (mock.patch.object(wizard,'stop_engine',return_value={'status':'incomplete'}),
                  mock.patch('builtins.print')):
                with self.assertRaises(SystemExit) as stopped:
                    wizard.main(['--stop-engine',str(path)])
            self.assertEqual(stopped.exception.code,1)

    def test_parent_exit_does_not_leave_its_verified_listener(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, child, port, state = self.listener_group(Path(temporary))
            result = stop_engine({'state': str(state)})
            self.assertEqual(result['status'], 'stopped')
            parent.wait(timeout=5)
            self.assertIsNone(process_identity(child['pid']))
            self.assertFalse((state/'running.json').exists())
            _check_available_port(port)

    def test_missing_leader_without_member_receipt_stays_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, child, port, state = self.listener_group(Path(temporary))
            parent.terminate();parent.wait(timeout=5)
            result = stop_engine({'state': str(state)})
            self.assertEqual(result['status'], 'incomplete')
            self.assertTrue((state/'running.json').exists())
            self.assertEqual(process_identity(child['pid']), child)
            with self.assertRaises(SetupError):
                _check_available_port(port)

    def test_resumable_member_receipt_can_finish_after_leader_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, child, port, state = self.listener_group(Path(temporary))
            record=json.loads((state/'running.json').read_text())
            record['owned_members']=[child]
            private_json(state/'running.json',record)
            parent.terminate();parent.wait(timeout=5)
            result=stop_engine({'state':str(state)})
            self.assertEqual(result['status'],'stopped')
            self.assertIsNone(process_identity(child['pid']))
            _check_available_port(port)

    def test_stale_process_identity_is_not_signaled(self):
        record=process_identity(os.getpid());record['start']='stale'
        with mock.patch('owner_mode.engine_setup.os.kill') as kill:
            self.assertFalse(_signal_identity(record,signal.SIGTERM))
        kill.assert_not_called()

    def test_time_wait_is_reusable_but_an_active_listener_is_rejected(self):
        listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0));listener.listen(1)
        port=listener.getsockname()[1]
        with self.assertRaises(SetupError):
            _check_available_port(port)
        client=socket.create_connection(('127.0.0.1',port),timeout=2)
        accepted,_=listener.accept()
        accepted.shutdown(socket.SHUT_RDWR);accepted.close()
        client.recv(1);client.close();listener.close()
        with socket.socket() as plain:
            with self.assertRaises(OSError):
                plain.bind(('127.0.0.1',port))
        _check_available_port(port)


if __name__ == '__main__':
    unittest.main()
