# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded shutdown and real late supervisor completion without model hardware."""
import hashlib
import json
import threading
import os
import time
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import test_native_remote
from owner_mode import native_remote, native_cluster, engine_setup


class ShutdownBudget(unittest.TestCase):
    def test_invalid_budget_fails_before_files_or_processes(self):
        for value in (False, True, None, -1, 0, .5, 301, float('nan'), float('inf'), '60'):
            with self.subTest(value=value), mock.patch.object(native_remote, 'owned_directory') as owned:
                with self.assertRaises(ValueError):
                    native_remote.stop({'state':'unused','timeout_seconds':value})
                owned.assert_not_called()
        self.assertEqual(native_remote.stop_timeout(), 60)

    def test_controllers_finish_before_workers_share_one_remaining_budget(self):
        barrier=threading.Barrier(3);controller_done=threading.Event();seen=[]
        def rpc(host,operation,**kwargs):
            self.assertEqual(operation,'stop')
            self.assertGreater(kwargs['transport_timeout'],0)
            self.assertLessEqual(kwargs['transport_timeout'],3)
            self.assertLessEqual(kwargs['timeout_seconds'],kwargs['transport_timeout'])
            self.assertEqual(kwargs['nonce'],'a'*32)
            seen.append(host)
            if host=='controller':
                controller_done.set()
            else:
                self.assertTrue(controller_done.is_set());barrier.wait(1)
            return {'status':'stopped','all_children_reaped':True}
        units=[{'host':f'worker{i}','state':f'/unit{i}','nonce':'a'*32} for i in range(3)]
        deployment={'state':'/unused','remote_units':units,'controller_units':[{'host':'controller','state':'/controller','nonce':'a'*32}],'stop_timeout':3}
        with mock.patch.object(native_cluster,'rpc',side_effect=rpc), \
             mock.patch('owner_mode.engine_supervisor.read_unit_record',return_value={}), \
             mock.patch.object(native_cluster,'_durable_json'):
            self.assertEqual(native_cluster._stop(deployment)['status'],'stopped')
        self.assertEqual(seen[0],'controller');self.assertEqual(set(seen),{'controller','worker0','worker1','worker2'})

    def test_rpc_stop_timeout_is_not_the_ordinary_45_second_limit(self):
        result=SimpleNamespace(stdout=b'{"status":"stopped"}',returncode=0)
        with mock.patch.object(native_cluster.subprocess,'run',return_value=result) as run:
            native_cluster.rpc('local','stop',state='/unit',timeout_seconds=60)
        self.assertEqual(run.call_args.kwargs['timeout'],65)

    def test_final_receipt_published_during_exit_is_refreshed(self):
        test_native_remote.NativeRemote.setUp(self)
        path=Path(native_remote.stage(self.request,native_cluster.archive(self.files))['state'])
        helper=native_remote._supervisor(path);owner=helper.identity(os.getpid(),True);nonce='a'*32
        helper.save(path/'unit-launch.json',{'version':1,'nonce':nonce,'phase':'owned','supervisor':owner})
        before={'status':'stopping','supervisor':owner,'nonce':nonce,'supervisor_alive':False,'all_children_reaped':False}
        after={**before,'status':'stopped','all_children_reaped':True}
        with mock.patch.object(native_remote,'_state_of',side_effect=[before,after]) as state:
            self.assertEqual(native_remote.stop({'state':str(path),'nonce':nonce}),
                             {'status':'stopped','all_children_reaped':True})
        self.assertEqual(state.call_count,2)

    def test_local_units_share_budget_and_remain_nonce_scoped(self):
        barrier=threading.Barrier(2);seen=[]
        attempt={'version':1,'nonce':'a'*32,'phase':'running','units':['/unit0','/unit1']}
        def stop(request):
            seen.append(request);barrier.wait(1);return {'status':'stopped','all_children_reaped':True}
        with mock.patch.object(engine_setup._unit_supervisor,'unit_lock',return_value=__import__('contextlib').nullcontext()), \
             mock.patch.object(engine_setup,'_activation',return_value=attempt), \
             mock.patch.object(engine_setup._unit_supervisor,'read_unit_record',return_value=None), \
             mock.patch.object(engine_setup._unit_supervisor,'save'), \
             mock.patch.object(engine_setup._native_units,'stop',side_effect=stop), \
             mock.patch.object(engine_setup._native_units,'state_of',return_value={}), \
             mock.patch.object(engine_setup,'_remove_stopped_marker'):
            self.assertEqual(engine_setup._stop_local_attempt({'state':'/unused','stop_timeout':3})['status'],'stopped')
        self.assertEqual(len(seen),2)
        self.assertTrue(all(r['nonce']=='a'*32 and 1<=r['timeout_seconds']<=3 for r in seen))

    def test_real_supervisor_can_attest_completion_after_old_fifteen_second_limit(self):
        test_native_remote.NativeRemote.setUp(self)
        source=Path(native_remote.__file__).with_name('engine_supervisor.py').read_text()
        old="        save(path,{**base,'status':'stopped','all_children_reaped':True,'exit_code':code})"
        self.assertEqual(source.count(old),1)
        delayed=source.replace(old,"        time.sleep(16)\n"+old)
        self.files['engine_supervisor.py']=(delayed.encode(),False)
        self.request['files']['engine_supervisor.py'].update(sha256=hashlib.sha256(delayed.encode()).hexdigest(),bytes=len(delayed.encode()))
        original=native_remote._supervisor_module
        try:
            with mock.patch.object(native_remote,'_NATIVE_SUPERVISOR_SOURCE',delayed,create=True):
                native_remote._supervisor_module=None
                path=native_remote.stage(self.request,native_cluster.archive(self.files))['state']
                request={'state':path};native_remote.start(request)
                deadline=time.monotonic()+5
                while not native_remote.state_of(path).get('command_alive'):
                    self.assertLess(time.monotonic(),deadline);time.sleep(.02)
                start=time.monotonic()
                try:
                    result=native_remote.stop(request)
                    self.assertEqual(result,{'status':'stopped','all_children_reaped':True})
                    self.assertGreater(time.monotonic()-start,15)
                    self.assertLess(time.monotonic()-start,30)
                finally:
                    native_remote.stop(request)
        finally:
            native_remote._supervisor_module=original

    def test_upgrade_stops_retained_supervisor_without_restaging_or_new_signature(self):
        test_native_remote.NativeRemote.setUp(self)
        source=Path(native_remote.__file__).with_name('engine_supervisor.py').read_text()
        old='def unit_lock(directory, timeout=20):'
        self.assertEqual(source.count(old),1)
        retained=source.replace(old,'def unit_lock(directory):\n    timeout=20')+'\n# retained deployment helper\n'
        self.files['engine_supervisor.py']=(retained.encode(),False)
        self.request['files']['engine_supervisor.py'].update(sha256=hashlib.sha256(retained.encode()).hexdigest(),bytes=len(retained.encode()))
        original=native_remote._supervisor_module;path=None
        try:
            with mock.patch.object(native_remote,'_NATIVE_SUPERVISOR_SOURCE',retained,create=True):
                native_remote._supervisor_module=None
                path=Path(native_remote.stage(self.request,native_cluster.archive(self.files))['state'])
                request={'state':str(path)};native_remote.start(request)
                helper=native_remote._supervisor(path);until=time.monotonic()+5
                while not native_remote._state_of(path,helper).get('command_alive'):
                    self.assertLess(time.monotonic(),until);time.sleep(.02)
            # A new SDK uses its own checked helper for control; the running
            # supervisor and its immutable on-disk fingerprint stay untouched.
            native_remote._supervisor_module=None
            before=(path/'stage-receipt.json').read_bytes()
            self.assertEqual(native_remote.stop(request)['status'],'stopped')
            self.assertEqual((path/'engine_supervisor.py').read_bytes(),retained.encode())
            self.assertEqual((path/'stage-receipt.json').read_bytes(),before)
        finally:
            if path is not None:
                native_remote._supervisor_module=None
                native_remote.stop({'state':str(path)})
            native_remote._supervisor_module=original


if __name__=='__main__':unittest.main()
