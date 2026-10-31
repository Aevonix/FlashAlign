# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real compiler-shaped descendants may change process groups and outlive parents."""
import json
import os
from pathlib import Path
import signal
import subprocess
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.engine_setup import activate_engine,stop_engine,process_identity,_signal_identity,_children
from owner_mode import engine_supervisor
from owner_mode.setup_runtime import SetupError


class DescendantCleanup(unittest.TestCase):
    def root(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def fixture(self,root,group_kind='setpgid',crash=False):
        state=root/'state';state.mkdir();ready=root/'leaf.json'
        leaf=("import json,os,signal,socket,time; from pathlib import Path; "
              "signal.signal(signal.SIGTERM,signal.SIG_IGN); s=socket.socket(); "
              "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',0)); s.listen(); "
              f"Path({str(ready)!r}).write_text(json.dumps({{'pid':os.getpid(),'port':s.getsockname()[1],'group':os.getpgrp()}})); time.sleep(300)")
        change="os.setsid()" if group_kind=='setsid' else 'os.setpgid(0,0)'
        middle=("import os,subprocess,sys; "+change+"; "
                f"subprocess.Popen([sys.executable,'-c',{leaf!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); os._exit(0)")
        head=("import os,subprocess,sys,time; "
              f"subprocess.Popen([sys.executable,'-c',{middle!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
              +("time.sleep(.3); os._exit(7)" if crash else "time.sleep(300)"))
        if group_kind=='threaded':
            threaded_leaf='import os; os.setsid(); '+leaf
            head=("import subprocess,sys,threading,time\n"
                  "def spawn():\n"
                  f" subprocess.Popen([sys.executable,'-c',{threaded_leaf!r}],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                  " time.sleep(300)\n"
                  "threading.Thread(target=spawn).start()\n"
                  "time.sleep(300)\n")
        deployment={'state':str(state),'source':str(root),'base_url':'http://127.0.0.1:0/v1','model':'fixture','profile':'fixture','key_file':str(root/'unused'),'install_dependencies':False,'commands':[{'argv':[sys.executable,'-c',head],'env':{}}]}
        def cleanup():
            if ready.exists():
                row=json.loads(ready.read_text());record=process_identity(row['pid'])
                if record:_signal_identity(record,signal.SIGKILL)
            stop_engine(deployment)
            for p in list(_children.values()):
                try:p.wait(timeout=2)
                except Exception:pass
        self.addCleanup(cleanup)
        return deployment,ready

    def ready_models(self,ready):
        def models(*_,**__):
            until=time.monotonic()+5
            while not ready.exists() and time.monotonic()<until:time.sleep(.01)
            self.assertTrue(ready.exists());return [{'id':'fixture'}]
        return models

    def test_changed_group_or_session_and_orphan_are_recorded_and_stopped(self):
        for kind in ('setpgid','setsid'):
            with self.subTest(kind=kind):
                root=self.root();deployment,ready=self.fixture(root,kind)
                with mock.patch('owner_mode.engine_setup.model_rows',side_effect=self.ready_models(ready)),mock.patch('owner_mode.engine_setup.check_engine',return_value={'status':'active'}):
                    activate_engine(deployment,timeout=8)
                record=json.loads((root/'state/running.json').read_text());leaf=json.loads(ready.read_text())
                self.assertNotEqual(leaf['group'],record['processes'][0]['pid'])
                self.assertIn(leaf['pid'],[r['pid'] for r in record['owned_members']])
                self.assertEqual(stop_engine(deployment)['status'],'stopped')
                self.assertIsNone(process_identity(leaf['pid']))
                supervisor=json.loads((root/'state/supervisor-0.json').read_text())
                self.assertTrue(supervisor['all_children_reaped'])
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);probe.bind(('127.0.0.1',leaf['port']))

    def test_parent_failure_cleans_new_group_orphan_before_reporting_failure(self):
        root=self.root();deployment,ready=self.fixture(root,crash=True)
        with mock.patch('owner_mode.engine_setup.model_rows',side_effect=SetupError('not ready')):
            with self.assertRaises(SetupError):activate_engine(deployment,timeout=3)
        self.assertTrue(ready.exists());leaf=json.loads(ready.read_text())
        self.assertIsNone(process_identity(leaf['pid']))
        self.assertFalse((root/'state/running.json').exists())
        self.assertTrue(json.loads((root/'state/supervisor-0.json').read_text())['all_children_reaped'])

    def test_thread_spawned_new_session_is_tracked_without_global_scan(self):
        root=self.root();deployment,ready=self.fixture(root,'threaded')
        with (mock.patch('owner_mode.engine_setup.model_rows',side_effect=self.ready_models(ready)),
              mock.patch('owner_mode.engine_setup.check_engine',return_value={'status':'active'}),
              mock.patch.object(engine_supervisor,'_scan_descendants',side_effect=AssertionError('Unexpected host-wide scan'))):
            activate_engine(deployment,timeout=8)
            leaf=json.loads(ready.read_text())
            record=json.loads((root/'state/running.json').read_text())
            self.assertNotEqual(leaf['group'],record['processes'][0]['pid'])
            self.assertIn(leaf['pid'],[row['pid'] for row in record['owned_members']])
            self.assertEqual(stop_engine(deployment)['status'],'stopped')
        self.assertIsNone(process_identity(leaf['pid']))
        self.assertTrue(json.loads((root/'state/supervisor-0.json').read_text())['all_children_reaped'])

    def test_unavailable_task_children_uses_verified_ancestry_fallback(self):
        record=engine_supervisor.identity(os.getpid(),True)
        with (mock.patch.object(engine_supervisor,'_child_ids',return_value=None),
              mock.patch.object(engine_supervisor,'_scan_descendants',wraps=engine_supervisor._scan_descendants) as fallback):
            found=engine_supervisor.descendants([record])
        self.assertIn(record,found)
        fallback.assert_called_once()

    def test_killed_supervisor_never_claims_unobserved_orphans_are_reaped(self):
        root=self.root();deployment,ready=self.fixture(root)
        with mock.patch('owner_mode.engine_setup.model_rows',side_effect=self.ready_models(ready)),mock.patch('owner_mode.engine_setup.check_engine',return_value={'status':'active'}):
            activate_engine(deployment,timeout=8)
        record=json.loads((root/'state/running.json').read_text());anchor=record['processes'][0]
        self.assertTrue(_signal_identity(anchor,signal.SIGKILL));_children[anchor['pid']].wait(timeout=5)
        result=stop_engine(deployment)
        self.assertEqual(result['status'],'incomplete');self.assertTrue((root/'state/running.json').exists())
        self.assertIsNone(process_identity(json.loads(ready.read_text())['pid']))

    def test_stop_from_later_process_retains_orphan_supervision(self):
        root=self.root();deployment,ready=self.fixture(root,'setsid')
        config=root/'deployment.json';config.write_text(json.dumps(deployment))
        code="""import json,time,sys
from pathlib import Path
from unittest import mock
from test_setup import wizard
from owner_mode.engine_setup import activate_engine
config=Path(sys.argv[1]);deployment=json.loads(config.read_text());ready=config.parent/'leaf.json'
def models(*a,**k):
    until=time.monotonic()+5
    while not ready.exists() and time.monotonic()<until:time.sleep(.01)
    return [{'id':'fixture'}]
with mock.patch('owner_mode.engine_setup.model_rows',side_effect=models),mock.patch('owner_mode.engine_setup.check_engine',return_value={'status':'active'}):
    activate_engine(deployment,timeout=8)
"""
        env={**os.environ,'PYTHONPATH':str(Path(__file__).parent),'PYTHONDONTWRITEBYTECODE':'1'}
        result=subprocess.run([sys.executable,'-c',code,str(config)],env=env,capture_output=True,text=True,timeout=12)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIsNotNone(process_identity(json.loads(ready.read_text())['pid']))
        self.assertEqual(stop_engine(deployment)['status'],'stopped')
        self.assertIsNone(process_identity(json.loads(ready.read_text())['pid']))

    def test_cleanup_write_failure_still_reaps_real_changed_group_child(self):
        root=self.root();deployment,ready=self.fixture(root,'setsid')
        state=root/'state/supervisor-denied.json'
        program="""import json,sys
from pathlib import Path
import engine_supervisor as module
original=module.save
def denied(path,value):
    if value['status'] in ('stopping','cleanup_incomplete','stopped'):
        raise OSError('simulated receipt write denial')
    return original(path,value)
module.save=denied
raise SystemExit(module.supervise(json.loads(sys.argv[1]),Path(sys.argv[2])))
"""
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]),'PYTHONDONTWRITEBYTECODE':'1'}
        proc=subprocess.Popen([sys.executable,'-c',program,json.dumps(deployment['commands'][0]['argv']),str(state)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        birth=process_identity(proc.pid)
        def finish():
            if proc.poll() is None:
                _signal_identity(birth,signal.SIGKILL);proc.wait(timeout=5)
        self.addCleanup(finish)
        until=time.monotonic()+5
        while not ready.exists() and time.monotonic()<until:time.sleep(.01)
        self.assertTrue(ready.exists())
        proc.terminate();self.assertNotEqual(proc.wait(timeout=15),0)
        leaf=json.loads(ready.read_text())
        self.assertIsNone(process_identity(leaf['pid']))
        self.assertFalse(json.loads(state.read_text())['all_children_reaped'])
        with socket.socket() as port:
            port.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);port.bind(('127.0.0.1',leaf['port']))

    def test_malformed_supervisor_receipt_does_not_raise_or_authorize(self):
        from owner_mode.engine_setup import _supervisor_state
        root=self.root();record=process_identity(os.getpid())
        for value in ([],{'supervisor':record,'command':[]},{'supervisor':record,'command':{'pid':1}}):
            (root/'supervisor-0.json').write_text(json.dumps(value))
            self.assertIsNone(_supervisor_state(root,{'receipt':'supervisor-0.json','process':record}))
        self.assertIsNone(_supervisor_state(root,None))

    def test_malformed_supervisor_descriptor_preserves_incomplete_state(self):
        root=self.root();state=root/'state';state.mkdir()
        (state/'running.json').write_text(json.dumps({'processes':[],'owned_members':[],'supervisors':[None]}))
        with mock.patch('owner_mode.engine_setup._signal_identity') as send:
            self.assertEqual(stop_engine({'state':str(state)})['status'],'incomplete')
        send.assert_not_called();self.assertTrue((state/'running.json').exists())

if __name__=='__main__':unittest.main()
