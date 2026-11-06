# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
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

from test_native_setup import make_bundle
from test_setup import make_profile
from owner_mode.native_bundle import check_bundle
from owner_mode.native_lifecycle import SCHEMA, declaration, addresses, prepare_runner
from owner_mode.native_setup import prepare, command
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError
from owner_mode.native_socket_forward import prepare as prepare_socket
from owner_mode import native_control_runner
from owner_mode import native_remote
from owner_mode.native_bundle import host_platform


def controlled_bundle(root):
    bundle = make_bundle(root)
    path = bundle/'bin/flashalign_admin.py'
    path.write_text('# fixture bridge\n')
    import hashlib
    manifest = json.loads((bundle/'manifest.json').read_text())
    manifest.update(runtime_version='1.2.0', profile_control={
        'schema': SCHEMA, 'protocol': 'flashalign.tensorfold-control.v2', 'bridge': 'bin/flashalign_admin.py'})
    manifest['files']['bin/flashalign_admin.py'] = {'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    return bundle, manifest


class NativeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_only_explicit_matching_bundle_can_enable_control(self):
        bundle, manifest = controlled_bundle(self.root/'bundle')
        profile = validate_profile(make_profile(self.root/'profile'))
        check_bundle(bundle, {'model_type': 'glm_moe_dsa'}, profile)
        self.assertEqual(declaration(manifest)['protocol'], 'flashalign.tensorfold-control.v2')
        for value in ({}, {**manifest['profile_control'], 'protocol': []},
                      {**manifest['profile_control'], 'bridge': 'bin/missing.py'}):
            with self.assertRaises(SetupError):declaration({**manifest, 'profile_control': value})
        self.assertIsNone(declaration({}))

    def test_local_setup_stages_one_stdlib_bridge_and_reports_admin_socket(self):
        bundle, _ = controlled_bundle(self.root/'bundle')
        profile = make_profile(self.root/'profile')
        model = self.root/'model';model.mkdir()
        (model/'config.json').write_text(json.dumps({'model_type': 'glm_moe_dsa', 'hidden_size': 2}))
        home = self.root/'home';home.mkdir()
        result = prepare(home, {'kind':'tensorfold','bundle':str(bundle),'model_path':str(model),'devices':['0']},
                         profile, model_id='unseen-label')
        argv = result['commands'][0]['argv']
        self.assertIn('native_control_runner.py', argv[-2])
        config = json.loads(Path(argv[-1]).read_text())
        self.assertIn('--owner-control-socket', config['engine_argv'])
        self.assertNotIn('--owner-profile', config['engine_argv'])
        self.assertEqual(config['admin_socket'], result['admin_socket'])
        package = Path(config['package'])
        proof = subprocess.run([sys.executable,'-I','-S','-c',
            'import sys;sys.path.insert(0,sys.argv[1]);import owner_mode.profiles,owner_mode.profile_lifecycle,owner_mode.profile_admin_server',
            str(package)],capture_output=True,text=True)
        self.assertEqual(proof.returncode,0,proof.stderr)

    def test_controlled_followers_use_the_same_serve_entry_without_signing_key(self):
        _, manifest = controlled_bundle(self.root/'bundle')
        manifest.update(rank_transport='dsv41_env', assets_directory='assets/runtime', follower_binary='bin/obsolete')
        checked = {'manifest':manifest,'drafting':{'method':'dspark'},'hosts':[{},{}],
                   'arguments':['--context','4096','--parallel','2']}
        for rank in (0,1):
            argv,env = command(self.root/'runtime',self.root/'profile',self.root/'key',
                {'rank':rank,'device':str(rank),'host':'local','model_path':'/models/test'},checked,
                {'engram_dir':'/models/engram'},'any-model',8000,'127.0.0.1',29500,
                replay_directory=self.root/'replay' if rank==0 else None)
            self.assertEqual(argv[1],'serve')
            self.assertEqual(argv[argv.index('--rank')+1],str(rank))
            self.assertIn('--owner-control-socket',argv)
            self.assertNotIn('--owner-profile',argv)
            self.assertEqual('--owner-key-file' in argv,rank==0)
            self.assertEqual(env['TF_TP_RANK'],str(rank))

    def test_control_runtime_upgrade_uses_captured_files_and_preserves_replay(self):
        from owner_mode import native_lifecycle
        bundle, _ = controlled_bundle(self.root/'bundle')
        profile = make_profile(self.root/'profile')
        model = self.root/'model'; model.mkdir()
        (model/'config.json').write_text(json.dumps({'model_type':'glm_moe_dsa','hidden_size':2}))
        home = self.root/'home'; home.mkdir()
        spec = {'kind':'tensorfold','bundle':str(bundle),'model_path':str(model),'devices':['0']}
        first = prepare(home, spec, profile, model_id='unseen-label')
        source = Path(native_lifecycle.__file__).with_name('profile_lifecycle.py')
        read_bytes = Path.read_bytes
        changed = read_bytes(source) + b'\n# changed control runtime fixture\n'
        reads = []
        def replacement(path):
            if path == source:
                reads.append(path)
                return changed
            return read_bytes(path)
        with mock.patch.object(Path, 'read_bytes', replacement):
            upgraded = prepare(home, spec, profile, model_id='unseen-label')
        self.assertEqual(len(reads), 1)
        self.assertNotEqual(upgraded['state'], first['state'])
        self.assertEqual(upgraded['key_file'], first['key_file'])
        self.assertEqual(upgraded['owner_replay'], first['owner_replay'])
        copied = Path(upgraded['state'])/'rank-0/control-python/owner_mode/profile_lifecycle.py'
        self.assertEqual(copied.read_bytes(), changed)
        self.assertEqual(prepare(home, spec, profile, model_id='unseen-label')['state'], first['state'])

    def test_remote_setup_has_private_admin_forward_and_only_leader_bridge(self):
        bundle, manifest=controlled_bundle(self.root/'bundle')
        profile=make_profile(self.root/'profile');home=self.root/'home';home.mkdir()
        stages=[]
        def rpc(host,operation,**kw):
            if operation=='inventory':
                return {'platform':host_platform(),'home':'/home/demo','model_path':'/models/demo',
                        'config':{'model_type':'glm_moe_dsa','hidden_size':2},'config_sha256':'a'*64,
                        'index_sha256':None,'devices':['0'],'active_drafting':[],
                        'python':'/usr/bin/python3','python_version':[3,12,1]}
            self.assertEqual(host,'local')
            return native_remote.stage(kw,kw['payload'])
        def stage(host,files,**kw):
            stages.append((host,files))
            return {'status':'prepared'}
        spec={'kind':'tensorfold','bundle':str(bundle),'master_address':'10.0.0.21','api_transport':'direct',
              'hosts':[{'host':'node'+str(n),'model_path':'/models/demo','devices':['0']} for n in range(4)]}
        with mock.patch('owner_mode.native_cluster.rpc',side_effect=rpc), \
             mock.patch('owner_mode.native_cluster.stage_files',side_effect=stage):
            result=prepare(home,spec,profile,model_id='custom-model')
        self.assertEqual(result['admin_remote']['host'],'node0')
        self.assertEqual(result['controller_units'][0]['kind'],'ssh_admin_forward')
        tunnel = Path(result['controller_units'][0]['state'])/'unit.json'
        argv = json.loads(tunnel.read_text())['argv']
        for option in ('ControlMaster=no', 'ControlPath=none', 'ControlPersist=no', 'ForkAfterAuthentication=no'):
            self.assertIn(option, argv)
        self.assertEqual(result['profile_lifecycle']['status'],'prepared')
        self.assertEqual(len(stages),4)
        for n,(_,files) in enumerate(stages):
            self.assertEqual('engine.key' in files,n==0)
            self.assertEqual('native-control.json' in files,n==0)
            if n==0:
                config=json.loads(files['native-control.json'][0])
                self.assertIn('--owner-control-socket',config['engine_argv'])
                self.assertNotIn('--owner-profile',config['engine_argv'])
        # No process was started; this empty, deterministic client socket directory is test-owned.
        Path(result['admin_socket']).parent.rmdir()

    def test_local_activation_waits_for_control_before_signed_probe(self):
        from owner_mode.engine_setup import activate_engine,stop_engine
        state=self.root/'state';state.mkdir()
        deployment={'python':sys.executable,'state':str(state),'source':str(self.root),
                    'base_url':'http://127.0.0.1:0/v1','model':'fixture','profile':'fixture',
                    'key_file':str(self.root/'unused'),'install_dependencies':False,
                    'profile_lifecycle':{'schema':'flashalign.lifecycle-launch.v1'},
                    'commands':[{'argv':[sys.executable,'-c','import time;time.sleep(60)'],'env':{}}]}
        self.addCleanup(lambda:stop_engine(deployment))
        calls=[]
        def readiness(*args,**kw):
            calls.append('control')
            if len(calls)<3:raise SetupError('Bridge is loading')
            return {'profile_lifecycle':{'status':'ready'}}
        def signed(*args,**kw):
            self.assertGreaterEqual(calls.count('control'),3)
            calls.append('signed');return {'status':'active'}
        with mock.patch('owner_mode.engine_setup.model_rows',return_value=[{'id':'fixture'}]), \
             mock.patch('owner_mode.lifecycle_setup.check_lifecycle',side_effect=readiness), \
             mock.patch('owner_mode.engine_setup.check_engine',side_effect=signed):
            result=activate_engine(deployment,timeout=5)
        self.assertEqual(calls[-1],'signed')
        self.assertEqual(result['profile_lifecycle']['status'],'ready')
        self.assertEqual(stop_engine(deployment)['status'],'stopped')

    def test_active_listener_is_refused_and_stale_socket_can_be_replaced(self):
        parent=self.root/'private';parent.mkdir(mode=0o700);path=parent/'admin.sock'
        listener=socket.socket(socket.AF_UNIX);self.addCleanup(listener.close)
        listener.bind(str(path));path.chmod(0o600);listener.listen()
        with self.assertRaisesRegex(ValueError,'already active'):prepare_socket(path)
        self.assertTrue(path.exists())
        listener.close();prepare_socket(path);self.assertFalse(path.exists())
        path.write_text('not a socket');path.chmod(0o600)
        with self.assertRaises(ValueError):prepare_socket(path)

    def test_runner_stops_engine_and_bridge_and_refuses_live_previous_engine(self):
        source=Path(native_control_runner.__file__).parent
        runtime=self.root/'runtime';runtime.mkdir()
        control=self.root/'control';control.mkdir(mode=0o700)
        # Both subprocesses expose owned sockets and record their PIDs for the cleanup assertion.
        engine=self.root/'engine.py';bridge=self.root/'bridge.py'
        script='import os,socket,sys,time\np=sys.argv[1];s=socket.socket(socket.AF_UNIX);s.bind(p);os.chmod(p,0o600);s.listen();open(p+".pid","w").write(str(os.getpid()))\nwhile True:time.sleep(.1)\n'
        engine.write_text(script)
        bridge.write_text(script.replace('p=sys.argv[1]', 'p=sys.argv[sys.argv.index("--admin-socket")+1]'))
        manifest={'runtime_version':'1.2.0','rank_transport':'cli','files':{'bin/bridge.py':{}},
            'profile_control':{'schema':SCHEMA,'protocol':'flashalign.tensorfold-control.v2','bridge':'bin/bridge.py'}}
        argv,files,paths=prepare_runner(manifest,runtime,[sys.executable,str(engine),str(control/'native.sock')],
                                      self.root/'profile',python=sys.executable,timeout=5)
        for name,(data,_) in files.items():
            path=runtime/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data if isinstance(data,bytes) else Path(data).read_bytes());path.chmod(0o600)
        config=json.loads((runtime/'native-control.json').read_text())
        config.update(control_directory=str(control),control_socket=str(control/'native.sock'),
                      admin_socket=str(control/'admin.sock'),bridge=str(bridge))
        (runtime/'native-control.json').write_text(json.dumps(config))
        proc=subprocess.Popen(argv,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            deadline=time.monotonic()+6
            while time.monotonic()<deadline and not (control/'admin.sock.pid').exists() and proc.poll() is None:time.sleep(.02)
            self.assertTrue((control/'admin.sock.pid').exists())
            pids=[int((control/(name+'.pid')).read_text()) for name in ('native.sock','admin.sock')]
            duplicate=subprocess.run(argv,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=3)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertIsNone(proc.poll())
            proc.terminate();self.assertEqual(proc.wait(timeout=5),0,proc.stderr.read().decode())
            for pid in pids:
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            proc.stderr.close()
