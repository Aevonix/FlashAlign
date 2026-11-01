# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Small encoded fixtures exercise native artifact and public asset contracts."""
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.native_profile import FORMAT, REQUIRES, NativePayload
from owner_mode.profile_assets import (download_record, materialize_profile, _download,
                                       _public_url, _ReleaseRedirects)
from owner_mode.profile_inspector import inspect_profile, render
from owner_mode.profiles import (ProfileError, discover_profiles, stage_profile,
                                 validate_profile, profile_fingerprint)
from owner_mode.native_setup import check as check_tensorfold
from owner_mode.setup_runtime import SetupError
from owner_mode.backend_setup import (KIT_ROOT, check_backend, prepare_backend, active_speculation)
from owner_mode.quickstart import make_default_plan


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def native_fixture(root, *, missing=False):
    root.mkdir()
    tensors, header, data = {}, {}, bytearray()
    for suffix, shape, dtype, byte in [('weight', [32, 64], 'F8_E4M3', 1),
                                     ('scale', [1, 2], 'F8_E8M0', 127)]:
        name = 'layers.0.attn.wo_b.'+suffix
        size = shape[0]*shape[1]
        raw = bytes([byte])*size
        begin = len(data);data.extend(raw)
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [begin, len(data)]}
        partition = bytes([byte])*(size//2)
        tensors[name] = {**header[name], 'stock_sha256': digest(raw), 'edited_sha256': digest(raw),
                         'stock_partitions': [digest(partition)]*2,
                         'edited_partitions': [digest(partition)]*2}
    encoded = json.dumps(header, separators=(',', ':')).encode()
    payload = struct.pack('<Q', len(encoded))+encoded+data
    filename = 'native.safetensors'
    doc = {'format': FORMAT, 'name': 'native-test',
           'model': {'id': 'fixture/model', 'revision': '1'*40},
           'source': {'id': 'fixture/edited', 'revision': '2'*40},
           'hidden_size': 32, 'input_size': 64, 'layers': [0], 'tp_world_size': 2,
           'transform': 'native_matrix_override_v1', 'requires': list(REQUIRES), 'strengths': [1.0],
           'payload': {'file': filename, 'size_bytes': len(payload), 'sha256': digest(payload),
                       'header_size_bytes': len(encoded), 'header_sha256': digest(encoded)},
           'tensors': tensors}
    (root/'profile.json').write_text(json.dumps(doc))
    record = {'version': 1, 'file': filename, 'url': 'https://github.com/example/repo/releases/download/v1/'+filename,
              'bytes': len(payload), 'sha256': digest(payload), 'terms': 'RESPONSIBLE_USE.md'}
    (root/'payload-download.json').write_text(json.dumps(record))
    (root/'RESPONSIBLE_USE.md').write_text('Fixture source terms.\n')
    (root/'LICENSE').write_text('Fixture license.\n')
    if not missing:
        (root/filename).write_bytes(payload)
    seal(root, doc)
    return doc, payload


def seal(root, doc):
    (root/'profile.json').write_text(json.dumps(doc))
    checks = {path.name: digest(path.read_bytes()) for path in root.iterdir()
              if path.name not in ('sha256.json', doc['payload']['file'])}
    checks[doc['payload']['file']] = doc['payload']['sha256']
    (root/'sha256.json').write_text(json.dumps(checks))


class NativeProfiles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_all_hashes_and_partitions_are_checked_without_dense_allocation(self):
        path = self.root/'profile';doc, _ = native_fixture(path)
        with mock.patch.object(NativePayload, 'partition', side_effect=AssertionError('no dense allocation')):
            checked = validate_profile(path, include_audit=True)
        self.assertTrue(checked['artifact_valid'])
        self.assertEqual(checked['schema'], FORMAT)
        self.assertEqual(checked['audit']['site_count'], 1)
        self.assertEqual(len(checked['audit']['tensors']), 2)
        self.assertEqual(checked['payload_status'], 'verified')
        doc['tensors']['layers.0.attn.wo_b.weight']['edited_partitions'][1] = '0'*64
        seal(path, doc)
        with self.assertRaisesRegex(ProfileError, 'partition'):
            validate_profile(path)

    def test_missing_payload_is_discoverable_but_not_file_validated(self):
        path = self.root/'profile';native_fixture(path, missing=True)
        with mock.patch('socket.socket', side_effect=AssertionError('offline')):
            profiles, errors = discover_profiles([path])
        self.assertEqual(errors, [])
        self.assertEqual(len(profiles), 1)
        self.assertIsNone(profiles[0]['artifact_valid'])
        self.assertTrue(profiles[0]['metadata_valid'])
        with self.assertRaisesRegex(ProfileError, 'payload is missing'):
            validate_profile(path)
        with self.assertRaisesRegex(ProfileError, 'payload is missing'):
            inspect_profile(path)

    def test_header_tensor_and_finite_encoding_fail_closed(self):
        for mutation in ('bytes', 'nan', 'overlap', 'extension', 'strength', 'filename', 'boolean_shape'):
            with self.subTest(mutation=mutation):
                path = self.root/mutation;doc, payload = native_fixture(path)
                if mutation in ('bytes', 'nan'):
                    data = bytearray(payload)
                    data[-3] = 127 if mutation == 'nan' else 2
                    (path/doc['payload']['file']).write_bytes(data)
                elif mutation == 'overlap':
                    doc['tensors']['layers.0.attn.wo_b.scale']['data_offsets'] = [0, 2]
                    seal(path, doc)
                elif mutation == 'extension':
                    doc['unrecognized'] = True;seal(path, doc)
                elif mutation == 'strength':
                    doc['strengths'] = [1.1];seal(path, doc)
                elif mutation == 'filename':
                    doc['payload']['file'] = '\x1b[0m.safetensors';seal(path, doc)
                else:
                    doc['tensors']['layers.0.attn.wo_b.scale']['shape'] = [True, 2];seal(path, doc)
                with self.assertRaises(ProfileError):
                    validate_profile(path)

    def test_payload_symlink_is_rejected(self):
        path = self.root/'profile';doc, payload = native_fixture(path, missing=True)
        target = self.root/'payload';target.write_bytes(payload)
        (path/doc['payload']['file']).symlink_to(target)
        with self.assertRaises(ProfileError):validate_profile(path)

    def test_inspector_and_stage_keep_native_names_and_source_terms(self):
        path = self.root/'profile';native_fixture(path)
        with mock.patch('socket.socket', side_effect=AssertionError('offline')):
            report = inspect_profile(path)
        self.assertEqual(report['basis_count'], 0)
        self.assertEqual(report['maximum_rank'], 0)
        text = render(report, details=True)
        self.assertIn('Native tensors: 2', text)
        self.assertIn('SHA-256 native.safetensors:', text)
        self.assertNotIn('rank None', text)
        self.assertEqual(report['assessment']['model_quality'], 'not_assessed')
        stage = self.root/'staged';stage_profile(path, stage)
        self.assertEqual(profile_fingerprint(validate_profile(path)), profile_fingerprint(validate_profile(stage)))
        self.assertEqual((stage/'RESPONSIBLE_USE.md').read_bytes(), (path/'RESPONSIBLE_USE.md').read_bytes())

    def test_tensorfold_rejects_native_before_inventory_or_download(self):
        path = self.root/'profile';native_fixture(path, missing=True)
        profile = validate_profile(path, allow_missing_payload=True)
        with mock.patch('owner_mode.native_setup.nodes', side_effect=AssertionError('inventory')):
            with self.assertRaisesRegex(SetupError, 'vLLM native integration'):
                check_tensorfold(self.root, {}, profile, model_id='fixture/model')


class NativeAssets(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile = self.root/'profile'
        self.doc, self.payload = native_fixture(self.profile, missing=True)

    def test_apply_downloads_to_external_cache_and_reuses_only_valid_artifact(self):
        def fetch(record, path):path.write_bytes(self.payload)
        with mock.patch('owner_mode.profile_assets._download', side_effect=fetch) as download:
            ready = materialize_profile(self.profile, cache_root=self.root/'cache')
            again = materialize_profile(self.profile, cache_root=self.root/'cache')
        self.assertEqual(download.call_count, 1)
        self.assertEqual(ready['path'], again['path'])
        self.assertTrue(ready['artifact_valid'])
        self.assertFalse((self.profile/'native.safetensors').exists())
        self.assertEqual((Path(ready['path'])/'RESPONSIBLE_USE.md').read_bytes(),
                         (self.profile/'RESPONSIBLE_USE.md').read_bytes())

    def test_failed_download_never_publishes_a_cache_entry(self):
        with mock.patch('owner_mode.profile_assets._download', side_effect=lambda r,p:p.write_bytes(b'bad')):
            with self.assertRaises(ProfileError):materialize_profile(self.profile, cache_root=self.root/'cache')
        self.assertEqual(list((self.root/'cache').iterdir()), [])

    def test_changed_descriptor_or_notice_is_rejected_offline(self):
        checked = validate_profile(self.profile, allow_missing_payload=True)
        for name in ('payload-download.json', 'RESPONSIBLE_USE.md'):
            old = (self.profile/name).read_bytes()
            (self.profile/name).write_bytes(old+b' ')
            with self.assertRaisesRegex(ProfileError, 'changed'):download_record(checked)
            (self.profile/name).write_bytes(old)

    def test_asset_urls_and_redirects_cannot_reach_other_services(self):
        for url in ('http://github.com/a/b/releases/download/v1/x',
                    'https://github.com:444/a/b/releases/download/v1/x',
                    'https://user:secret@github.com/a/b/releases/download/v1/x',
                    'https://127.0.0.1/a', 'https://example.com/a',
                    'https://github.com/a/b/releases/download/v1/x?token=secret'):
            with self.subTest(url=url), self.assertRaises(ProfileError):_public_url(url)
        with self.assertRaises(ProfileError):
            _ReleaseRedirects().redirect_request(None, None, 302, '', {}, 'http://127.0.0.1/')

    def test_stream_download_checks_size_hash_and_no_authorization(self):
        record = download_record(validate_profile(self.profile, allow_missing_payload=True))
        class Response(io.BytesIO):
            status = 200
            def geturl(self):return 'https://release-assets.githubusercontent.com/asset?signature=test'
        for body, expected_error in ((self.payload, False), (self.payload+b'x', True), (b'bad', True)):
            response = Response(body);response.headers = {}
            opener = mock.Mock();opener.open.return_value = response
            target = self.root/('download-'+str(len(body)))
            with mock.patch('owner_mode.profile_assets.build_opener', return_value=opener):
                if expected_error:
                    with self.assertRaises(ProfileError):_download(record, target)
                else:_download(record, target)
            request = opener.open.call_args.args[0]
            self.assertNotIn('Authorization', request.headers)
            self.assertNotIn('Cookie', request.headers)


class NativeSetup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root/'home';self.home.mkdir()
        self.path = self.root/'profile'
        native_fixture(self.path)
        self.profile = validate_profile(self.path)
        self.model = self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type': 'deepseek_v41',
            'hidden_size': 32, 'num_hidden_layers': 1}))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        self.kit = KIT_ROOT/'vllm-deepseek-native'
        self.metadata = json.loads((self.kit/'backend.json').read_text())
        self.metadata['compatible_models'] = [{**self.profile['model'], 'model_type': 'deepseek_v41',
            'config_sha256': digest((self.model/'config.json').read_bytes()),
            'index_sha256': digest((self.model/'model.safetensors.index.json').read_bytes())}]
        self.spec = {'kind': 'vllm', 'model_path': str(self.model), 'devices': ['0','1','2','3']}

    def patches(self):
        return mock.patch('owner_mode.backend_setup._kits',
                          side_effect=lambda:iter([(self.kit, self.metadata, 'vllm')]))

    def check(self, spec=None):
        with self.patches(), mock.patch('owner_mode.backend_setup.active_speculation', return_value=None):
            return check_backend(self.home, spec or self.spec, self.profile, model_id='fixture/model')

    def test_native_backend_selects_exact_speculation_without_rank_or_model_substitution(self):
        checked = self.check()
        self.assertEqual(checked['metadata']['variant'], 'deepseek-v41-native')
        self.assertNotIn('maximum_rank', checked['metadata'])
        self.assertEqual(checked['drafting']['speculative_config'], self.metadata['default_speculative_config'])
        self.assertNotIn('model', checked['drafting']['speculative_config'])
        for field, value in [('num_speculative_tokens', 5), ('attention_backend', 'other'),
                             ('enable_adaptive_verification', 1), ('adaptive_verification_cost_scale', True)]:
            config = {**self.metadata['default_speculative_config'], field: value}
            with self.subTest(field=field), self.assertRaises(SetupError):
                self.check({**self.spec, 'speculative_config': config})
        config = {**self.metadata['default_speculative_config'], 'model': str(self.model)}
        self.assertEqual(self.check({**self.spec, 'speculative_config': config})['drafting']['speculative_config'], config)

    def test_incompatible_backend_rejects_before_any_asset_download(self):
        spec = {**self.spec, 'variant': 'deepseek-v41'}
        with self.patches(), mock.patch('owner_mode.profile_assets.materialize_profile', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(SetupError, 'No unique'):
                prepare_backend(self.home, spec, self.path, model_id='fixture/model')

    def test_docker_preparation_wires_isolated_native_launcher(self):
        runtime = {'python': sys.executable, 'container_runtime': 'docker', 'image': 'fixture:image', 'image_id': 'sha256:test'}
        with (self.patches(), mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value=runtime),
              mock.patch('owner_mode.backend_setup._run')):
            result = prepare_backend(self.home, self.spec, self.path, model_id='fixture/model')
        argv = result['commands'][0]['argv']
        self.assertEqual(result['container_runtime'], 'docker')
        self.assertTrue(argv[1].endswith('/integration/launch.py'))
        self.assertIn('--model-receipt', argv)
        self.assertIn('--replay-file', argv)
        self.assertIn('--state-dir', argv)
        self.assertEqual(result['startup_timeout'], 3600)
        self.assertEqual(argv[argv.index('--startup-timeout')+1], '3600')
        self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1]), self.metadata['default_speculative_config'])
        self.assertNotIn('PYTHONPATH', result['commands'][0]['env'])
        self.assertFalse((Path(result['state'])/'running.json').exists())

    def test_default_wizard_selects_native_variant_without_downloading(self):
        (self.path/'native.safetensors').unlink()
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=self.path,
            engine_bundle=None, backend='auto', profiles=[], engine_python=sys.executable,
            devices='0,1,2,3', tools_default=None, engine_only=True)
        out = io.StringIO()
        with (self.patches(), mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.profile_assets._download', side_effect=AssertionError('network')),
              mock.patch('builtins.input', side_effect=AssertionError('no prompt')), redirect_stdout(out)):
            path = make_default_plan(args, {}, wizard.Console(color=False))
        plan = json.loads(path.read_text())
        self.assertEqual(plan['engine']['variant'], 'deepseek-v41-native')
        self.assertEqual(plan['engine']['speculative_config'], self.metadata['default_speculative_config'])
        self.assertIn('public profile payload', out.getvalue())
        self.assertIn('Source terms:', out.getvalue())
        self.assertFalse((self.path/'native.safetensors').exists())

    def test_active_host_launcher_preserves_native_speculation(self):
        process = self.root/'proc/123';process.mkdir(parents=True)
        config = self.metadata['default_speculative_config']
        argv = [sys.executable, str(self.kit/'launch.py'), '--model', str(self.model),
                '--speculative-config', json.dumps(config)]
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in argv)+b'\0')
        (process/'cwd').symlink_to(self.root)
        active = active_speculation(self.model, 'fixture/model', self.root/'proc')
        self.assertEqual(active['kind'], 'vllm')
        self.assertEqual(active['speculative_config'], config)

    def test_auto_selection_never_downloads_tensorfold_as_native_fallback(self):
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=self.path,
            engine_bundle=None, backend='auto', profiles=[], engine_python=sys.executable,
            devices='0,1,2,3', tools_default=None, engine_only=True)
        with (self.patches(), mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.quickstart.backend_matches_checkpoint', return_value=False),
              mock.patch('owner_mode.quickstart.download_engine_bundle', side_effect=AssertionError('fallback download')),
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(SetupError, 'No native vLLM integration'):
                make_default_plan(args, {}, wizard.Console(color=False))


if __name__ == '__main__':
    unittest.main()
