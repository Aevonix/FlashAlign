# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Profile validation and setup integration using synthetic files and services."""
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from support import Engine, load_cli

ROOT = Path(__file__).resolve().parents[1]
wizard = load_cli('setup')
from owner_mode.profiles import ProfileError, discover_profiles, match_profile, validate_profile
from owner_mode.setup_runtime import Console, SetupError, check_engine, engine_profiles, restart_harnesses
from owner_mode.auth import read_api_key
from owner_mode.policy import Policy
from owner_mode.proxy import ProviderProxy
import httpx


def make_profile(root, *, model='bench', name='custom-owner', values=(1., 0.), transform=None):
    root.mkdir(parents=True, exist_ok=True)
    profile = {'name': name, 'model': {'id': model, 'revision': 'test-revision'},
               'hidden_size': 2, 'sites': [{'layer': 0, 'site': 'attn_out',
                   'tensor': 'layers.0.output.weight', 'rank': 1, 'scale': .75}]}
    if transform:
        profile['transform'] = transform
    (root/'profile.json').write_text(json.dumps(profile))
    header = json.dumps({'L0.attn_out.Q': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [0, 8]}}).encode()
    (root/'profile.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<ff', *values))
    return root


def write_tensors(root, tensors):
    header, body = {}, b''
    for name, (shape, values) in tensors.items():
        packed = struct.pack('<'+'f'*len(values), *values)
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [len(body), len(body)+len(packed)]}
        body += packed
    encoded = json.dumps(header).encode()
    (root/'profile.safetensors').write_bytes(struct.pack('<Q', len(encoded))+encoded+body)


class ProfileValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_valid_profile_and_matching_revision_and_format(self):
        result = validate_profile(make_profile(self.root/'valid'))
        self.assertTrue(result['artifact_valid'])
        self.assertTrue(match_profile(result, {'id': 'bench'})[0])
        self.assertFalse(match_profile(result, {'id': 'different'})[0])
        self.assertFalse(match_profile(result, {'id': 'bench', 'revision': 'wrong'})[0])
        result['model']['format'] = 'FP8'
        self.assertFalse(match_profile(result, {'id': 'bench', 'format': 'INT4'})[0])

    def test_dense_mlp_and_embedding_output_sites(self):
        for kind in ('mlp_out','embedding_out'):
            path=make_profile(self.root/kind)
            metadata=json.loads((path/'profile.json').read_text())
            metadata['sites'][0]['site']=kind
            (path/'profile.json').write_text(json.dumps(metadata))
            write_tensors(path,{f'L0.{kind}.Q':([1,2],[1.,0.])})
            self.assertTrue(validate_profile(path)['artifact_valid'])
            if kind=='embedding_out':
                metadata['sites'][0]['layer']=1
                (path/'profile.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ProfileError,'layer zero'):
                    validate_profile(path)

    def test_missing_tensor_nonfinite_and_nonorthogonal_profiles_are_rejected(self):
        for label, values in [('nan', (math.nan, 0.)), ('wrong-norm', (2., 0.))]:
            with self.subTest(label=label), self.assertRaises(ProfileError):
                validate_profile(make_profile(self.root/label, values=values))
        path = make_profile(self.root/'missing')
        (path/'profile.safetensors').unlink()
        with self.assertRaises(ProfileError):
            validate_profile(path)

    def test_unsupported_transform_and_duplicate_sites_are_rejected(self):
        with self.assertRaises(ProfileError):
            validate_profile(make_profile(self.root/'unsupported', transform='unknown'))
        path = make_profile(self.root/'duplicate')
        data = json.loads((path/'profile.json').read_text())
        data['sites'] *= 2
        (path/'profile.json').write_text(json.dumps(data))
        with self.assertRaises(ProfileError):
            validate_profile(path)

    def test_trailing_payload_and_wrong_hash_are_rejected(self):
        path = make_profile(self.root/'extra')
        with (path/'profile.safetensors').open('ab') as stream:
            stream.write(b'JUNK')
        with self.assertRaises(ProfileError):
            validate_profile(path)
        path = make_profile(self.root/'hash')
        (path/'sha256.json').write_text(json.dumps({'profile.safetensors': '0'*64}))
        with self.assertRaisesRegex(ProfileError, 'checksum'):
            validate_profile(path)

    def test_catalog_alias_and_invalid_profiles_do_not_block_valid_ones(self):
        make_profile(self.root/'good', model='publisher/bench')
        make_profile(self.root/'bad', values=(0., 0.))
        (self.root/'catalog.json').write_text(json.dumps({'models': [
            {'path': 'good', 'profile': 'custom-owner', 'model_ids': ['bench']}]}))
        rows, errors = discover_profiles([self.root])
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(errors), 1)
        self.assertTrue(match_profile(rows[0], {'id': 'bench'})[0])

    def test_duplicate_json_fields_are_rejected(self):
        path = make_profile(self.root/'duplicate-json')
        (path/'profile.json').write_text('{"name":"one","name":"two"}')
        with self.assertRaises(ProfileError):
            validate_profile(path)

    def test_extension_tensors_are_validated_and_positive_norms_required(self):
        path = make_profile(self.root/'normalized')
        data = json.loads((path/'profile.json').read_text())
        data.update(transform='row_normalized_projection_v1', requires=['row_gain', 'row_norm_preconditioning'])
        (path/'profile.json').write_text(json.dumps(data))
        tensors = {'L0.attn_out.Q': ([1, 2], (1., 0.)),
                   'L0.attn_out.g': ([2], (1., 2.)), 'L0.attn_out.m': ([2], (3., 4.))}
        write_tensors(path, tensors)
        self.assertTrue(validate_profile(path)['artifact_valid'])
        tensors['L0.attn_out.m'] = ([2], (0., 4.))
        write_tensors(path, tensors)
        with self.assertRaisesRegex(ProfileError, 'positive'):
            validate_profile(path)

    def test_input_delta_requires_the_correct_input_shape(self):
        path = make_profile(self.root/'delta')
        data = json.loads((path/'profile.json').read_text())
        data.update(transform='input_low_rank_delta_v1', requires=['matrix_input'])
        data['sites'][0]['input_size'] = 3
        (path/'profile.json').write_text(json.dumps(data))
        tensors = {'L0.attn_out.Q': ([1, 2], (1., 0.)), 'L0.attn_out.B': ([1, 3], (1., 2., 3.))}
        write_tensors(path, tensors)
        self.assertTrue(validate_profile(path)['artifact_valid'])
        tensors['L0.attn_out.B'] = ([3, 1], (1., 2., 3.))
        write_tensors(path, tensors)
        with self.assertRaisesRegex(ProfileError, 'shapes'):
            validate_profile(path)


class SetupFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)/'home'
        self.home.mkdir()
        self.key = Path(self.tmp.name)/'engine.key'
        self.key.write_bytes(os.urandom(32))
        self.key.chmod(0o600)
        self.engine = Engine(self.key.read_bytes(), profiles=('custom-owner',))
        self.addCleanup(self.engine.close)
        self.profile = make_profile(Path(self.tmp.name)/'profile')
        self.output = io.StringIO()
        self.executed = []
        self.found = {name: {'detected': True, 'executable': '/fake/'+name,
                    'runtime_compatible': True, 'model_urls': [], 'models': []}
                    for name in ('hermes', 'openclaw')}

    def answers(self, *, harness='both', profile='1'):
        return [harness, self.engine.base_url, '1', profile, str(self.key),
                'verified', 'rcs', '+15550000011', 'default', 'off', 'yes']

    def run_flow(self, answers=None, extra=(), restart=None):
        with mock.patch.object(wizard, 'discover', return_value=self.found), \
             mock.patch('builtins.input', side_effect=answers or self.answers()), \
             mock.patch.object(wizard, 'restart_harnesses', side_effect=restart or
                              (lambda home, selected, found: {name: {'status': 'restarted'} for name in selected})), \
             redirect_stdout(self.output):
            return wizard.main(['--home', str(self.home), '--profiles', str(self.profile), '--no-color', *extra])

    def test_custom_model_profile_is_verified_applied_and_restarts_selected_harnesses(self):
        calls = []
        def restart(home, selected, found):
            calls.append((home, selected))
            return {name: {'status': 'restarted'} for name in selected}
        result = self.run_flow(restart=restart)
        self.assertEqual(calls, [(self.home, ['hermes', 'openclaw'])])
        self.assertEqual(result['profile'], 'custom-owner')
        self.assertEqual(result['engine_check']['status'], 'active')
        self.assertEqual(result['engine_check']['standard_mode'], 'off')
        self.assertEqual(validate_profile(result['profile_path'])['name'], 'custom-owner')
        hermes = json.loads((self.home/'.hermes/config.yaml').read_text())
        openclaw = json.loads((self.home/'.openclaw/openclaw.json').read_text())
        self.assertEqual(hermes['plugins']['entries']['owner-mode']['settings']['routes']['head']['profile'], 'custom-owner')
        self.assertEqual(openclaw['agents']['defaults']['model']['primary'], 'owner-mode/bench')
        self.assertIn('FLASHALIGN', self.output.getvalue())
        self.assertIn('AEVONIX RESEARCH', self.output.getvalue())
        self.assertNotIn(self.key.read_bytes().hex(), self.output.getvalue())

    def test_wrong_key_prevents_configuration_and_restart(self):
        self.key.write_bytes(os.urandom(32))
        with self.assertRaises(SetupError):
            self.run_flow()
        self.assertFalse((self.home/'.hermes/config.yaml').exists())
        self.assertFalse((self.home/'.openclaw/openclaw.json').exists())

    def test_arbitrary_installed_profile_name_without_local_artifact(self):
        self.profile.rename(self.profile.with_name('hidden'))
        result = self.run_flow(self.answers(profile='custom-owner'), extra=('--no-restart',))
        self.assertEqual(result['profile'], 'custom-owner')
        self.assertNotIn('profile_path', result)
        self.assertEqual(result['restart']['hermes']['status'], 'skipped')

    def test_rejected_profile_does_not_change_existing_configuration(self):
        path = self.home/'.hermes/config.yaml'
        path.parent.mkdir()
        path.write_text('{"unchanged":true}')
        previous = path.read_bytes()
        self.engine.profiles.clear()
        with self.assertRaises(SetupError):
            self.run_flow()
        self.assertEqual(path.read_bytes(), previous)
        self.assertNotIn('restarted', self.output.getvalue())

    def test_restart_failure_is_reported_and_configuration_is_retained(self):
        result = self.run_flow(restart=lambda *_: {'hermes': {'status': 'failed'}, 'openclaw': {'status': 'restarted'}})
        self.assertEqual(result['status'], 'restart_failed')
        self.assertEqual(result['activation'], 'gateway restart required')
        self.assertTrue((self.home/'.hermes/config.yaml').is_file())
        self.assertIn('ACTION NEEDED', self.output.getvalue())

    def test_optional_inventory_filters_unloaded_profiles(self):
        with mock.patch('owner_mode.setup_runtime.request', return_value=(200, {}, {'data': [
            {'name': 'yes', 'model': {'id': 'bench'}, 'loaded': True},
            {'name': 'no', 'model': {'id': 'bench'}, 'loaded': False}]})):
            self.assertEqual([p['name'] for p in engine_profiles(self.engine.base_url)], ['yes'])

    def test_restart_uses_argument_lists_and_scopes_each_harness(self):
        calls = []
        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout='DO NOT PRINT', stderr='DO NOT PRINT')
        result = restart_harnesses(self.home, ['hermes', 'openclaw'], self.found, runner=runner)
        self.assertTrue(all(row['status'] == 'restarted' for row in result.values()))
        self.assertEqual(calls[0][0], ['/fake/hermes', 'gateway', 'restart'])
        self.assertEqual(calls[1][0], ['/fake/openclaw', 'gateway', 'restart'])
        self.assertEqual(calls[0][1]['env']['HERMES_HOME'], str(self.home/'.hermes'))
        self.assertEqual(calls[1][1]['env']['OPENCLAW_CONFIG_PATH'], str(self.home/'.openclaw/openclaw.json'))
        self.assertNotIn('shell', calls[0][1])

    def test_changed_profile_is_rejected_before_adapter_install(self):
        details = validate_profile(self.profile)
        data = json.loads((self.profile/'profile.json').read_text())
        data['sites'][0]['scale'] = .5
        (self.profile/'profile.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'changed'):
            wizard.provision(self.home, ['hermes'], self.engine.base_url, 'bench', 'custom-owner', [], False,
                             self.key, profile_details=details)
        self.assertFalse((self.home/'.hermes/config.yaml').exists())

    def test_missing_effective_header_is_not_success(self):
        with mock.patch('owner_mode.setup_runtime.request', return_value=(200, {}, {})):
            with self.assertRaises(SetupError):
                check_engine(self.engine.base_url, 'bench', 'custom-owner', self.key)

    def test_protected_upstream_token_is_used_for_runtime_requests(self):
        token = 'synthetic-test-token'
        result = wizard.provision(self.home, ['hermes'], self.engine.base_url, 'bench', 'custom-owner', [], False,
                                  self.key, api_key=token)
        cfg = json.loads((self.home/'.hermes/config.yaml').read_text())
        routes = cfg['plugins']['entries']['owner-mode']['settings']['routes']
        self.assertNotIn(token, json.dumps(cfg))
        self.assertEqual(read_api_key(routes['head']['api_key_file']), token)
        self.engine.required_authorization = 'Bearer ' + token
        proxy = ProviderProxy(Policy(), routes, self.key)
        proxy.start()
        self.addCleanup(proxy.close)
        with httpx.Client() as client:
            response = client.post(proxy.base_url('head')+'/chat/completions',
                                   json={'model': 'bench', 'messages': []}, headers={'Authorization': 'Bearer incorrect'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'off')

    def test_branding_honors_no_color(self):
        with mock.patch.dict(os.environ, {'NO_COLOR': '1'}):
            console = Console(output=self.output)
            console.banner('1.0.0')
        self.assertNotIn('\x1b', self.output.getvalue())
        self.assertIn('ReAlign models instantly, on demand.', self.output.getvalue())

    def plan(self, **changes):
        path = Path(self.tmp.name)/'setup-plan.json'
        data = {'harnesses': ['hermes', 'openclaw'], 'base_url': self.engine.base_url,
                'model': 'bench', 'profile': str(self.profile), 'engine_key_file': str(self.key),
                'contacts': [{'channel': 'rcs', 'account': 'default', 'contact': '+15550000011'}],
                'tools_default': False}
        data.update(changes)
        path.write_text(json.dumps(data))
        path.chmod(0o600)
        return path

    def run_plan(self, path, action):
        with mock.patch.object(wizard, 'discover', return_value=self.found), \
             mock.patch('builtins.input', side_effect=AssertionError('Agent setup must not prompt')), \
             mock.patch.object(wizard, 'restart_harnesses', side_effect=AssertionError('Agent setup must not restart')), \
             redirect_stdout(self.output):
            return wizard.main(['--home', str(self.home), '--config', str(path), action])

    def test_agent_preflight_does_not_write_configuration_or_restart(self):
        result = self.run_plan(self.plan(), '--check')
        self.assertEqual(result['status'], 'verified')
        self.assertFalse(result['applied'])
        self.assertFalse((self.home/'.config').exists())
        self.assertFalse((self.home/'.hermes').exists())

    def test_agent_apply_stops_at_restart_with_commands_and_context(self):
        result = self.run_plan(self.plan(), '--apply')
        self.assertEqual(result['status'], 'awaiting_restart')
        self.assertTrue(result['applied'])
        self.assertTrue((self.home/'.hermes/config.yaml').is_file())
        for item in result['restart'].values():
            self.assertEqual(item['status'], 'pending')
            self.assertEqual(item['command'][1:], ['gateway', 'restart'])
        self.assertEqual(result['restart']['hermes']['env']['HERMES_HOME'], str(self.home/'.hermes'))

    def test_agent_rejects_public_plan_and_embedded_restart_commands(self):
        path = self.plan()
        path.chmod(0o644)
        with self.assertRaises(SetupError):
            self.run_plan(path, '--apply')
        path = self.plan(restart=['arbitrary', 'command'])
        with self.assertRaisesRegex(SetupError, 'unknown fields'):
            self.run_plan(path, '--apply')

    def test_agent_rejected_profile_preserves_configuration(self):
        self.engine.profiles.clear()
        with self.assertRaises(SetupError):
            self.run_plan(self.plan(), '--apply')
        self.assertFalse((self.home/'.hermes').exists())

    def test_new_deployment_keys_are_unique_and_reruns_reuse_them(self):
        first, second = Path(self.tmp.name)/'one/key', Path(self.tmp.name)/'two/key'
        self.assertEqual(wizard.init_signing_key(first)['status'], 'created')
        saved = first.read_bytes()
        self.assertEqual(len(saved), 32)
        self.assertEqual(first.stat().st_mode & 0o777, 0o600)
        self.assertEqual(wizard.init_signing_key(first)['status'], 'reused')
        self.assertEqual(first.read_bytes(), saved)
        wizard.init_signing_key(second)
        self.assertNotEqual(first.read_bytes(), second.read_bytes())


if __name__ == '__main__':
    unittest.main()
