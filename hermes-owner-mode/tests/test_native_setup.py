# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Native-only packaging, rank staging, key ownership and activation boundaries."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile
from owner_mode.native_bundle import check_bundle, host_platform, PLATFORMS
from owner_mode.native_setup import prepare, check, command, architecture_matches
from owner_mode.native_cluster import activate
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError


def make_bundle(root, *, platform_id=None):
    root.mkdir()
    platform_id = platform_id or host_platform()
    elf = bytearray(64)
    elf[:6] = b'\x7fELF\x02\x01'
    elf[18:20] = PLATFORMS[platform_id].to_bytes(2, 'little')
    files = {'bin/tensorfold-native': bytes(elf), 'lib/libnccl.so.2': bytes(elf), 'lib/libtfgrammar.so': bytes(elf),
             'licenses/libtfgrammar.so.txt': b'grammar license', 'licenses/libnccl.so.2.txt': b'NCCL license',
             'LICENSE': b'fixture license', 'NOTICE.md': b'fixture notice',
             'sources/tensorfold.tar.gz': b'fixture source archive'}
    for name, data in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {'version': 2, 'engine': 'tensorfold-native', 'runtime_version': '1.1.9',
                'source_revision': 'a'*40, 'platform': platform_id, 'binary': 'bin/tensorfold-native',
                'cuda_architectures': ['sm_89', 'sm_120', 'sm_121'],
                'source_archive': 'sources/tensorfold.tar.gz', 'rank_transport': 'cli',
                'durable_owner_replay': True,
                'model_types': ['glm_moe_dsa'], 'transforms': ['projection'],
                'tensor_parallel_sizes': [1, 4], 'speculative_methods': ['mtp'],
                'default_speculative_method': 'mtp', 'default_mtp_drafts': 4,
                'arguments': ['--context', '65536', '--parallel', '2'],
                'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                 **({'executable': True} if name.startswith('bin/') else {})}
                          for name, data in files.items()}}
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


class NativeSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.bundle = make_bundle(self.root/'bundle')
        self.profile = make_profile(self.root/'profile')
        self.model = self.root/'model'
        self.model.mkdir()
        self.config = {'model_type': 'glm_moe_dsa', 'hidden_size': 2}
        (self.model/'config.json').write_text(json.dumps(self.config))
        self.spec = {'kind': 'tensorfold', 'bundle': str(self.bundle), 'model_path': str(self.model), 'devices': ['0']}

    def test_native_only_contract_and_file_tampering(self):
        profile = validate_profile(self.profile)
        check_bundle(self.bundle, self.config, profile)
        path = self.bundle/'manifest.json'
        data = json.loads(path.read_text())
        data.update(version=1, engine='tensorfold-python')
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(SetupError, 'TensorFold 1.1.9'):
            check_bundle(self.bundle, self.config, profile)
        data.update(version=2, engine='tensorfold-native')
        path.write_text(json.dumps(data))
        (self.bundle/'bin/tensorfold-native').write_bytes(b'changed')
        with self.assertRaisesRegex(SetupError, 'checksum'):
            check_bundle(self.bundle, self.config, profile)

    def test_prepare_is_repeatable_uses_unique_key_and_keeps_mtp(self):
        one = prepare(self.home, self.spec, self.profile, model_id='fixture')
        two = prepare(self.home, self.spec, self.profile, model_id='fixture')
        self.assertEqual(one['key_file'], two['key_file'])
        key = Path(one['key_file'])
        self.assertEqual(len(key.read_bytes()), 32)
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        elsewhere = self.root/'elsewhere'
        elsewhere.mkdir()
        third = prepare(elsewhere, self.spec, self.profile, model_id='fixture')
        self.assertNotEqual(key.read_bytes(), Path(third['key_file']).read_bytes())
        argv = one['commands'][0]['argv']
        self.assertTrue(argv[0].endswith('/bin/tensorfold-native'))
        self.assertIn('--owner-key-file', argv)
        self.assertIn('--owner-profile', argv)
        self.assertIn('--owner-replay-dir', argv)
        self.assertNotIn('--no-drafts', argv)
        self.assertEqual(argv[argv.index('--mtp-drafts')+1], '4')
        self.assertFalse((Path(one['state'])/'running.json').exists())

    def test_same_key_upgrade_preserves_replay_directory_and_new_keys_are_isolated(self):
        path = self.bundle/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['durable_owner_replay'] = True
        path.write_text(json.dumps(manifest))
        key = self.home/'persistent-engine.key'
        first = prepare(self.home, self.spec, self.profile, model_id='fixture', key_file=key)
        replay = Path(first['owner_replay']['directory'])
        self.assertTrue(replay.parent.is_dir())
        self.assertFalse(replay.exists())  # only the engine may initialize the journal leaf
        self.assertNotIn(Path(first['state']), replay.parents)
        replay.mkdir(mode=0o700)
        (replay/'state.bin').write_bytes(b'preserved journal')
        manifest['source_revision'] = 'd'*40
        path.write_text(json.dumps(manifest))
        upgraded = prepare(self.home, self.spec, self.profile, model_id='fixture', key_file=key)
        self.assertNotEqual(first['state'], upgraded['state'])
        self.assertEqual(first['owner_replay'], upgraded['owner_replay'])
        self.assertEqual((replay/'state.bin').read_bytes(), b'preserved journal')
        moved_key = self.home/'same-key-new-path'
        moved_key.write_bytes(key.read_bytes())
        moved_key.chmod(0o600)
        next_profile = make_profile(self.root/'profile-v2', name='second-owner-profile')
        changed = prepare(self.home, self.spec, next_profile, model_id='fixture', key_file=moved_key)
        self.assertNotEqual(upgraded['state'], changed['state'])
        self.assertEqual(first['owner_replay'], changed['owner_replay'])
        other = prepare(self.home, {**self.spec, 'port': 18443}, self.profile, model_id='fixture')
        self.assertNotEqual(first['owner_replay']['directory'], other['owner_replay']['directory'])
        self.assertTrue(first['owner_replay']['durable'])
        argv = upgraded['commands'][0]['argv']
        self.assertEqual(argv[argv.index('--owner-replay-dir')+1], str(replay))

    def test_durable_replay_is_required_before_preparation(self):
        path = self.bundle/'manifest.json'
        original = json.loads(path.read_text())
        for invalid in (False, 'true', 1, None, 'missing'):
            manifest = {**original, 'durable_owner_replay': invalid}
            if invalid == 'missing':
                manifest.pop('durable_owner_replay')
            path.write_text(json.dumps(manifest))
            with self.subTest(capability=invalid):
                with self.assertRaisesRegex(SetupError, 'durable_owner_replay: true'):
                    prepare(self.home, self.spec, self.profile, model_id='fixture')
        self.assertFalse((self.home/'.config').exists())

    def test_known_ram_only_key_is_not_silently_reused_or_rotated(self):
        key = self.home/'old-engine.key'
        key.write_bytes(b'previous prototype signing key!!')
        key.chmod(0o600)
        old_state = self.home/'.config/owner-mode/engines/old-prototype'
        old_state.mkdir(parents=True)
        old = {'runtime': 'native', 'key_file': str(key)}
        (old_state/'deployment.json').write_text(json.dumps(old))
        original = key.read_bytes()
        path = self.bundle/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['durable_owner_replay'] = True
        manifest['source_revision'] = 'd'*40
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SetupError, 'fresh signing-key file'):
            prepare(self.home, self.spec, self.profile, model_id='fixture', key_file=key)
        self.assertEqual(key.read_bytes(), original)
        self.assertNotIn('owner_replay', old)
        fresh = prepare(self.home, self.spec, self.profile, model_id='fixture', key_file=self.home/'fresh-engine.key')
        self.assertNotEqual(Path(fresh['key_file']).read_bytes(), original)
        self.assertTrue(fresh['owner_replay']['durable'])

    def test_durable_remote_leader_passes_stable_state_without_follower_auth_flags(self):
        arm = make_bundle(self.root/'replay-arm', platform_id='linux-aarch64')
        manifest_path = arm/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['durable_owner_replay'] = True
        manifest_path.write_text(json.dumps(manifest))
        spec = {**self.spec, 'bundle': str(arm), 'master_address': '10.0.0.1', 'api_transport': 'direct',
                'hosts': [{'host': 'spark'+str(i), 'model_path': '/models/glm', 'devices': ['0'],
                           'state_root': '/persistent/flashalign'} for i in range(4)]}
        stages = []
        def rpc(host, operation, **kwargs):
            if operation == 'inventory':
                return {'platform': 'linux-aarch64', 'home': '/home/test', 'model_path': '/models/glm',
                        'config': self.config, 'config_sha256': 'b'*64, 'index_sha256': None,
                        'devices': ['0'], 'active_drafting': []}
            with tarfile.open(fileobj=io.BytesIO(kwargs['payload'])) as archive:
                unit = json.loads(archive.extractfile('unit.json').read())
            stages.append((host, kwargs, unit))
            return {'status': 'prepared'}
        key = self.home/'remote-key'
        with mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc):
            first = prepare(self.home, spec, self.profile, model_id='fixture', key_file=key)
            first_stages = list(stages)
            stages.clear()
            manifest['source_revision'] = 'd'*40
            manifest_path.write_text(json.dumps(manifest))
            upgraded = prepare(self.home, spec, self.profile, model_id='fixture', key_file=key)
        self.assertEqual(first['owner_replay'], upgraded['owner_replay'])
        for index, (_, request, unit) in enumerate(first_stages):
            argv = unit['argv']
            self.assertEqual('--owner-key-file' in argv, index == 0)
            self.assertEqual('--owner-replay-dir' in argv, index == 0)
            self.assertEqual('owner_replay_dir' in request, index == 0)
            if index == 0:
                directory = request['owner_replay_dir']
                self.assertTrue(directory.startswith('/persistent/flashalign/native-replay/'))
                self.assertEqual(argv[argv.index('--owner-replay-dir')+1], directory)
                self.assertEqual(stages[0][1]['owner_replay_dir'], directory)

    def test_deepseek_replay_flag_reaches_only_its_http_leader(self):
        manifest = {'binary':'bin/tensorfold-dsv41', 'follower_binary':'bin/tf-dsv41',
                    'rank_transport':'dsv41_env', 'assets_directory':'assets/runtime',
                    'durable_owner_replay':True}
        hosts = [{'rank':i, 'host':'local', 'device':str(i), 'model_path':'/models/deepseek'} for i in range(4)]
        checked = {'manifest':manifest, 'hosts':hosts, 'drafting':{'method':'dspark'},
                   'arguments':['--context','65536']}
        for node in hosts:
            argv, _ = command(Path('/runtime'), Path('/profile'), Path('/key'), node, checked,
                              {'engram_source':'/original'}, 'deepseek', 8000, '127.0.0.1', 29000,
                              replay_directory=Path('/persistent/replay') if node['rank']==0 else None)
            self.assertEqual('--owner-key-file' in argv, node['rank']==0)
            self.assertEqual('--owner-replay-dir' in argv, node['rank']==0)
            if node['rank']:
                self.assertEqual(argv[:2], ['/runtime/bin/tf-dsv41', 'follow'])

    def test_custom_launch_and_speculative_disable_are_rejected(self):
        for change in ({'command': ['python3', '-m', 'tensorfold']}, {'arguments': ['--no-drafts']}):
            with self.subTest(change=change), self.assertRaises(SetupError):
                prepare(self.home, {**self.spec, **change}, self.profile, model_id='fixture')
        self.assertFalse((self.home/'.config').exists())

    def test_four_arm_ranks_stage_identical_profile_and_leader_key_without_activation(self):
        arm = make_bundle(self.root/'arm', platform_id='linux-aarch64')
        spec = {**self.spec, 'bundle': str(arm), 'master_address': '10.0.0.1', 'api_transport': 'direct',
                'hosts': [{'host': 'spark'+str(i), 'model_path': '/models/glm', 'devices': ['0']} for i in range(4)]}
        stages = []
        def rpc(host, operation, **kwargs):
            if operation == 'inventory':
                return {'platform': 'linux-aarch64', 'home': '/home/test', 'model_path': '/models/glm',
                        'config': self.config, 'config_sha256': 'b'*64, 'index_sha256': None,
                        'devices': ['0'], 'active_drafting': []}
            self.assertEqual(operation, 'stage')
            with tarfile.open(fileobj=io.BytesIO(kwargs['payload'])) as archive:
                names = ('engine.key', 'profile/profile.json', 'unit.json')
                stages.append({name: archive.extractfile(name).read() for name in names if name in archive.getnames()})
            return {'status': 'prepared'}
        with mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc):
            result = prepare(self.home, spec, self.profile, model_id='fixture')
        self.assertEqual(len(result['remote_units']), 4)
        self.assertIn('engine.key', stages[0])
        self.assertEqual(len(stages[0]['engine.key']), 32)
        self.assertTrue(all('engine.key' not in row for row in stages[1:]))
        self.assertEqual(len({row['profile/profile.json'] for row in stages}), 1)
        for rank, row in enumerate(stages):
            argv = json.loads(row['unit.json'])['argv']
            self.assertEqual(argv[argv.index('--rank')+1], str(rank))
            self.assertEqual('--owner-key-file' in argv, rank == 0)

    def test_glm_environment_ranks_keep_keys_required_by_their_command_contract(self):
        path = self.bundle/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['rank_transport'] = 'glm_env'
        path.write_text(json.dumps(manifest))
        spec = {**self.spec, 'devices':['0','1','2','3']}
        result = prepare(self.home, spec, self.profile, model_id='fixture')
        for command_row in result['commands']:
            argv = command_row['argv']
            self.assertIn('--owner-key-file', argv)
            key = Path(argv[argv.index('--owner-key-file')+1])
            self.assertTrue(key.is_file())
            self.assertEqual(len(key.read_bytes()), 32)

    def test_cluster_activation_starts_followers_first_and_cleans_owned_failure(self):
        events = []
        deployment = {'state': str(self.root), 'remote_units': [{'host': 'n'+str(i), 'state': '/owned/'+str(i), 'rank': i} for i in range(4)],
                      'base_url': 'http://10.0.0.1:8000/v1', 'model': 'fixture', 'profile': 'owner', 'key_file': '/private/key'}
        def rpc(host, operation, **kwargs):
            events.append((operation, host))
            if operation == 'status':
                return {'status': 'stopped'}
            return {'status': 'stopped' if operation == 'stop' else 'starting'}
        with mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc):
            with self.assertRaisesRegex(SetupError, 'exited'):
                activate(deployment, timeout=1)
        self.assertEqual([host for action, host in events if action == 'start'], ['n1', 'n2', 'n3', 'n0'])
        self.assertEqual(set(host for action, host in events if action == 'stop'), {'n0', 'n1', 'n2', 'n3'})

    def test_remote_endpoint_is_a_persistent_owned_forward(self):
        arm = make_bundle(self.root/'forward-arm', platform_id='linux-aarch64')
        spec = {**self.spec, 'bundle': str(arm), 'master_address': '10.0.0.1',
                'hosts': [{'host': 'spark', 'model_path': '/models/glm', 'devices': ['0'],
                           'state_root': '/persistent/flashalign'}]}
        stages = []
        def rpc(host, operation, **kwargs):
            if operation == 'inventory':
                return {'platform': 'linux-aarch64', 'home': '/home/test', 'model_path': '/models/glm',
                        'config': self.config, 'config_sha256': 'b'*64, 'index_sha256': None,
                        'devices': ['0'], 'active_drafting': []}
            with tarfile.open(fileobj=io.BytesIO(kwargs['payload'])) as archive:
                unit = json.loads(archive.extractfile('unit.json').read())
            stages.append((host, kwargs['parent'], unit))
            return {'status': 'prepared'}
        with mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc):
            result = prepare(self.home, spec, self.profile, model_id='fixture')
        self.assertTrue(result['base_url'].startswith('http://127.0.0.1:'))
        self.assertTrue(result['remote_base_url'].startswith('http://10.0.0.1:'))
        self.assertEqual(stages[0][1], '/persistent/flashalign')
        self.assertEqual(result['controller_units'][0]['host'], 'local')
        self.assertEqual(stages[-1][2]['argv'][0], 'ssh')
        self.assertIn('StrictHostKeyChecking=yes', stages[-1][2]['argv'])
        self.assertIn('-N', stages[-1][2]['argv'])
        self.assertNotIn('engine.key', ' '.join(stages[-1][2]['argv']))

    def test_cluster_forward_is_started_last_and_stopped_with_ranks(self):
        from owner_mode.native_cluster import stop
        units = [{'host': 'n'+str(i), 'state': '/owned/'+str(i), 'rank': i} for i in range(2)]
        local = {'host': 'local', 'state': '/owned/tunnel', 'kind': 'ssh_forward'}
        deployment = {'state': str(self.root), 'remote_units': units, 'controller_units': [local],
                      'base_url': 'http://127.0.0.1:8000/v1', 'model': 'fixture', 'profile': 'owner', 'key_file': '/key'}
        events = []
        def rpc(host, operation, **kwargs):
            events.append((operation,host))
            return {'status': 'stopped' if operation == 'stop' else 'running', 'command_alive': True}
        with (mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc),
              mock.patch('owner_mode.native_cluster.model_rows', return_value=[{'id':'fixture'}]),
              mock.patch('owner_mode.native_cluster.check_engine', return_value={'status':'active'})):
            result = activate(deployment, timeout=1)
            self.assertEqual(result['status'], 'running')
            self.assertEqual(stop(deployment)['status'], 'stopped')
        self.assertEqual([host for action,host in events if action=='start'], ['n1','n0','local'])
        self.assertEqual([host for action,host in events if action=='stop'], ['local','n1','n0'])

    def test_runtime_deadline_and_network_interface_are_checked(self):
        path = self.bundle/'manifest.json'
        original = json.loads(path.read_text())
        for bad in (0, 59, 7201, '1800'):
            path.write_text(json.dumps({**original,'startup_timeout':bad}))
            with self.assertRaisesRegex(SetupError, 'startup_timeout'):
                check_bundle(self.bundle,self.config,validate_profile(self.profile))
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(SetupError, 'network interface'):
            check(self.home,{**self.spec,'network_interface':'nonexistent-interface'},validate_profile(self.profile),model_id='fixture')

    def test_fixed_native_mtp_preserves_drafts_without_unsupported_flag(self):
        path = self.bundle/'manifest.json'
        data = json.loads(path.read_text())
        data.update(mtp_drafts_tunable=False, default_mtp_drafts=3)
        path.write_text(json.dumps(data))
        result = prepare(self.home,self.spec,self.profile,model_id='fixture')
        self.assertEqual(result['drafting'], {'method':'mtp','mtp_drafts':3,'tunable':False})
        self.assertNotIn('--mtp-drafts', result['commands'][0]['argv'])
        self.assertNotIn('--no-drafts', result['commands'][0]['argv'])
        with self.assertRaisesRegex(SetupError,'fixed MTP depth'):
            prepare(self.home,{**self.spec,'mtp_drafts':4},self.profile,model_id='fixture')

    def test_deepseek_auxiliary_source_is_checked_before_key_or_staging(self):
        import os
        path = self.bundle/'manifest.json'
        manifest = json.loads(path.read_text())
        extra = {'bin/tf-dsv41':(self.bundle/'bin/tensorfold-native').read_bytes(), 'assets/runtime/rope.bin':b'fixture'}
        for name, data in extra.items():
            target=self.bundle/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            manifest['files'][name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                                     **({'executable':True} if name.startswith('bin/') else {})}
        manifest.update(rank_transport='dsv41_env',follower_binary='bin/tf-dsv41',assets_directory='assets/runtime',
            auxiliary_engram={'source':{'id':'Missing/Fixture','revision':'a'*40,'config_sha256':'b'*64,'index_sha256':'c'*64}},
            asset_geometry={'context':65536,'pool_tokens':65536,'slots':1,'tensor_parallel':1},
            environment={'TF_DSV41_CONTEXT':'65536','TF_DSV41_POOL_TOKENS':'65536','TF_DSV41_SLOTS':'1'})
        path.write_text(json.dumps(manifest))
        with mock.patch.dict(os.environ,{'HF_HOME':str(self.root/'empty')}):
            with self.assertRaisesRegex(SetupError,'requires original Engram'):
                prepare(self.home,self.spec,self.profile,model_id='fixture')
        self.assertFalse((self.home/'.config').exists())

    def test_cpu_architecture_does_not_imply_gpu_compatibility(self):
        self.assertFalse(architecture_matches('sm_89', ['sm_120']))
        self.assertFalse(architecture_matches('sm_120', ['sm_121a']))
        self.assertTrue(architecture_matches('sm_121', ['sm_121a']))
        self.assertTrue(architecture_matches('sm_121', ['compute_120']))

    def test_grammar_and_corresponding_source_are_required(self):
        path = self.bundle/'manifest.json'
        original = json.loads(path.read_text())
        for key in ('source_archive',):
            modified = {k: v for k, v in original.items() if k != key}
            path.write_text(json.dumps(modified))
            with self.assertRaisesRegex(SetupError, 'source archive'):
                check_bundle(self.bundle, self.config, validate_profile(self.profile))
        modified = {**original, 'files': {k: v for k, v in original['files'].items() if k != 'lib/libtfgrammar.so'}}
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(SetupError, 'libtfgrammar'):
            check_bundle(self.bundle, self.config, validate_profile(self.profile))

    def test_environment_cannot_disable_speculation_before_preparation(self):
        with self.assertRaisesRegex(SetupError, 'cannot disable'):
            prepare(self.home, {**self.spec, 'environment': {'TF_DSV41_DRAFTS': '0'}}, self.profile, model_id='fixture')
        self.assertFalse((self.home/'.config').exists())

    def test_remote_dflash_is_checked_on_model_host(self):
        arm = make_bundle(self.root/'arm-dflash', platform_id='linux-aarch64')
        path = arm/'manifest.json'
        manifest = json.loads(path.read_text())
        manifest.update(speculative_methods=['dflash'], default_speculative_method='dflash')
        path.write_text(json.dumps(manifest))
        spec = {**self.spec, 'bundle': str(arm), 'hosts': [{'host': 'spark', 'model_path': '/remote/model', 'devices': ['0']}],
                'draft_model': '/remote/dflash'}
        def rpc(host, operation, **kwargs):
            self.assertEqual(operation, 'inventory')
            result = {'platform': 'linux-aarch64', 'home': '/home/test', 'model_path': '/remote/model', 'config': self.config,
                      'config_sha256': 'b'*64, 'index_sha256': None, 'devices': ['0'], 'active_drafting': []}
            if kwargs.get('draft_model'):
                result.update(draft_model='/remote/dflash', draft_config={'architectures': ['DFlash2DraftModel']})
            return result
        with mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc):
            result = check(self.home, spec, validate_profile(self.profile), model_id='fixture')
        self.assertEqual(result['drafting']['draft_model'], '/remote/dflash')


if __name__ == '__main__':
    unittest.main()
