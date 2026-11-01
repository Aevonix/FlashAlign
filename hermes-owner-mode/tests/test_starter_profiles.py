# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Starter discovery is offline and preserves each artifact's stated limits."""
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_setup import wizard, make_profile
from test_engine_setup import make_bundle
from owner_mode.engine_setup import prepare_engine, private_json
from owner_mode.profiles import ProfileError, bundled_profile_for, bundled_profile_root, discover_profiles
from owner_mode.quickstart import make_default_plan


class StarterProfiles(unittest.TestCase):
    def test_profile_suggestions_follow_the_catalog_without_name_guessing(self):
        catalog = {'models': [{'model_ids': ['Example/Model'], 'profile': 'example-profile'}]}
        self.assertEqual(wizard.profile_for('example/model', catalog), 'example-profile')
        self.assertIsNone(wizard.profile_for('unknown-deepseek-1017', catalog))
        catalog['models'].append({'model_ids': ['Example/Model'], 'profile': 'ambiguous'})
        self.assertIsNone(wizard.profile_for('Example/Model', catalog))

    def test_starter_profiles_have_valid_payloads_and_experimental_status(self):
        profiles, errors = discover_profiles([bundled_profile_root()])
        self.assertEqual(errors, [])
        self.assertEqual({p['name'] for p in profiles},
                         {'glm53-full-v1', 'deepseek41-flash1017-v1',
                          'glm53-flash-v1', 'qwen38-27b-v1', 'gemma4-26b-a4b-v1'})
        for profile in profiles:
            if profile.get('format') == 'flashalign.native_wo_b.v1':
                self.assertEqual(profile['payload_status'], 'missing')
                self.assertIsNone(profile['artifact_valid'])
                self.assertTrue(profile['metadata_valid'])
                manifest = json.loads(Path(profile['manifest']).read_text())
                self.assertEqual(manifest['source']['id'], 'drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated')
                self.assertRegex(manifest['source']['revision'], r'^[0-9a-f]{40}$')
                self.assertEqual(manifest['strengths'], [1.0])
                continue
            self.assertEqual(profile['status'], 'experimental_runtime_checked')
            manifest = json.loads(Path(profile['manifest']).read_text())
            self.assertLessEqual(set(manifest), {'name', 'model', 'hidden_size', 'sites', 'transform',
                                               'requires', 'source', 'status', 'schema', 'operators',
                                               'compatible_models', 'author', 'license_notice'})
            self.assertEqual(manifest['author'], 'Aevonix Inc')
            edited = manifest['source']['edited']
            self.assertRegex(edited['revision'], r'^[0-9a-f]{40}$')
            self.assertEqual(edited['url'], 'https://huggingface.co/' + edited['id'] + '/tree/' + edited['revision'])
            self.assertIn('Fitting and runtime packaging by Aevonix', manifest['source']['attribution'])
            self.assertNotIn('derivation', manifest['source'])
            self.assertNotIn('selection', manifest['source'])

    def test_source_declarations_and_retained_notices_are_separate(self):
        root = bundled_profile_root()
        expected = {'deepseek41-flash1017-v1': 'MIT',
                    'glm53-full-v1': 'LicenseRef-GLM-5.3',
                    'glm53-flash-v1': 'MIT',
                    'qwen38-27b-v1': 'Apache-2.0',
                    'gemma4-26b-a4b-v1': 'Apache-2.0'}
        for name, license_id in expected.items():
            with self.subTest(profile=name):
                manifest = json.loads((root/name/'profile.json').read_text())
                if manifest.get('format') == 'flashalign.native_wo_b.v1':
                    self.assertEqual(hashlib.sha256((root/name/'LICENSE').read_bytes()).hexdigest(),
                                     json.loads((root/name/'sha256.json').read_text())['LICENSE'])
                    self.assertTrue((root/name/'RESPONSIBLE_USE.md').is_file())
                    self.assertEqual(manifest['source']['revision'], 'de412d2249632903dace23399526e00a145aada5')
                    continue
                self.assertNotIn('license', manifest['source'])
                notice = manifest['license_notice']
                self.assertEqual(notice['license'], license_id)
                self.assertEqual(notice['scope'], 'retained_model_source_terms')
                self.assertEqual(notice['file'], 'LICENSE')
                self.assertEqual(notice['sha256'], hashlib.sha256((root/name/'LICENSE').read_bytes()).hexdigest())
                for source in ('stock', 'edited'):
                    declaration = manifest['source'][source]
                    self.assertEqual(declaration['license_declaration_url'],
                                     'https://huggingface.co/' + declaration['id'] + '/blob/' +
                                     declaration['revision'] + '/README.md')
                if name == 'gemma4-26b-a4b-v1':
                    self.assertEqual(manifest['source']['base']['license'], 'Apache-2.0')
                    self.assertEqual(manifest['source']['stock']['license'], 'Apache-2.0')
                    self.assertEqual(manifest['source']['edited']['license'], 'gemma')
                elif name == 'glm53-full-v1':
                    for source in ('stock', 'edited'):
                        declaration = manifest['source'][source]
                        self.assertEqual(declaration['license'], 'LicenseRef-GLM-5.3')
                        self.assertEqual(declaration['license_name'], 'GLM-5.3')
                        self.assertEqual(declaration['license_metadata_tag'], 'other')
                        self.assertTrue(declaration['license_url'].endswith('/LICENSE'))

    def test_starter_tensors_keep_their_frozen_payloads(self):
        expected = {
            'glm53-full-v1': (33559576, '997a245d0f824648afb9a10b4270f5698538e26dafcc79622ea1432e5b1135d5'),
            'glm53-flash-v1': (2964392, '797e47c0c0d7d19f151b64a5ac514eefab73b370ecbe911bd6da2644955be40a'),
            'qwen38-27b-v1': (2652448, 'd7a9b49364c51e313e74b58ea7ac244d470c38609550bf0bd96129da0c222a87'),
            'gemma4-26b-a4b-v1': (567208, 'ca290c628c644091ad366c5cc16b6d6d167ce59b6e2b123386e68db992d1e451'),
        }
        for name, (size, digest) in expected.items():
            with self.subTest(profile=name):
                payload = (bundled_profile_root()/name/'profile.safetensors').read_bytes()
                self.assertEqual(len(payload), size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
        native = bundled_profile_root()/'deepseek41-flash1017-v1'
        descriptor = json.loads((native/'payload-download.json').read_text())
        self.assertEqual(descriptor['bytes'], 1091589280)
        self.assertEqual(descriptor['sha256'], '88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864')
        self.assertEqual(descriptor['url'], 'https://github.com/Aevonix/FlashAlign/releases/download/v1.1.0/wo_b_l10_35_1017.safetensors')
        self.assertFalse((native/descriptor['file']).exists())

    def test_family_selection_handles_flat_and_nested_model_configurations(self):
        full = bundled_profile_for({'model_type': 'glm_moe_dsa', 'hidden_size': 6144})
        flash = bundled_profile_for({'model_type': 'glm5_next',
                                    'text_config': {'model_type': 'glm5_next_text', 'hidden_size': 4096}})
        self.assertEqual(full['name'], 'glm53-full-v1')
        self.assertEqual(flash['name'], 'glm53-flash-v1')
        deepseek = bundled_profile_for({'model_type': 'deepseek_v41',
                                       'text_config': {'model_type': 'deepseek_v41_text', 'hidden_size': 5120}})
        qwen = bundled_profile_for({'model_type': 'qwen3_5',
                                   'text_config': {'model_type': 'qwen3_5_text', 'hidden_size': 5120}})
        self.assertEqual(deepseek['name'], 'deepseek41-flash1017-v1')
        self.assertEqual(qwen['name'], 'qwen38-27b-v1')
        gemma = bundled_profile_for({'model_type': 'gemma4',
                                    'text_config': {'model_type': 'gemma4_text', 'hidden_size': 2816}})
        self.assertEqual(gemma['name'], 'gemma4-26b-a4b-v1')
        self.assertEqual(gemma['model'], {
            'id': 'RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic',
            'revision': 'ed35d7abe5d940da41b4ff06eb482feb0be8cb44'})
        vllm_models = json.loads((bundled_profile_root().parent/'engines/vllm-gemma4/backend.json').read_text())['compatible_models']
        for model in vllm_models:
            self.assertIn(model, gemma['compatible_models'])
        self.assertEqual(wizard.profile_for(gemma['model']['id']), 'gemma4-26b-a4b-v1')
        self.assertEqual(bundled_profile_for({'model_type': 'gemma4_text', 'hidden_size': 2816})['name'],
                         'gemma4-26b-a4b-v1')
        self.assertIsNone(bundled_profile_for({'model_type': 'another_model', 'hidden_size': 6144}))
        self.assertIsNone(bundled_profile_for({'model_type': 'glm_moe_dsa', 'hidden_size': 4096}))
        with self.assertRaisesRegex(ProfileError, 'revision'):
            bundled_profile_for({'model_type': 'glm_moe_dsa', 'hidden_size': 6144, '_commit_hash': 'different'})

    def test_default_setup_rejects_known_failed_starter_before_network_or_contact_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);home = root/'home';home.mkdir()
            model = root/'model';model.mkdir()
            (model/'config.json').write_text('{"model_type":"fixture_model","hidden_size":2}')
            profiles = root/'profiles'; profiles.mkdir()
            failed = make_profile(profiles/'failed')
            doc = json.loads((failed/'profile.json').read_text()); doc['status'] = 'failed_reproduction_gate'
            (failed/'profile.json').write_text(json.dumps(doc))
            (profiles/'catalog.json').write_text(json.dumps({'models': [{'path': 'failed',
                'model_types': ['fixture_model'], 'hidden_size': 2}]}))
            bundle = make_bundle(root/'bundle')
            args = SimpleNamespace(home=home, model_path=model, profile=None, engine_bundle=bundle,
                                   profiles=[], engine_python=None, devices='0', tools_default='off')
            found = {'hermes': {'detected': True, 'runtime_compatible': True},
                     'openclaw': {'detected': False, 'runtime_compatible': True}}
            with (mock.patch('owner_mode.profiles.bundled_profile_root', return_value=profiles),
                  mock.patch('owner_mode.quickstart.download_engine_bundle', side_effect=AssertionError('network')),
                  mock.patch('builtins.input', side_effect=AssertionError('contact prompt')),
                  redirect_stdout(io.StringIO())):
                with self.assertRaisesRegex(ProfileError, 'failed validation'):
                    make_default_plan(args, found, wizard.Console(color=False))
            self.assertFalse((home/'.config').exists())

    def test_agent_can_select_a_bundled_profile_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);home = root/'home';home.mkdir()
            model = root/'model';model.mkdir()
            (model/'config.json').write_text('{"model_type":"qwen3_5","hidden_size":5120}')
            bundle = make_bundle(root/'bundle')
            profiles, errors = discover_profiles([bundled_profile_root()])
            self.assertEqual(errors, [])
            profile = next(p for p in profiles if p['name'] == 'qwen38-27b-v1')
            metadata = json.loads((bundle/'manifest.json').read_text())
            metadata['model_types'] = ['qwen3_5']
            (bundle/'manifest.json').write_text(json.dumps(metadata))
            plan = root/'plan.json'
            private_json(plan, {'harnesses': ['hermes'], 'model': profile['model']['id'],
                'profile': profile['name'], 'contacts': [{'channel': 'rcs', 'account': 'default',
                'contact': '+15550000011'}], 'tools_default': False,
                'engine': {'model_path': str(model), 'bundle': str(bundle),
                           'arguments': ['--context', '65536']}})
            found = {'hermes': {'detected': True, 'runtime_compatible': True}}
            # This case checks profile-name resolution. Native checkpoint binding
            # has independent tests and cannot use this deliberately tiny model.
            with (mock.patch.object(wizard, 'discover', return_value=found),
                  mock.patch('owner_mode.automation.check_engine_setup', return_value={
                      'kind': 'tensorfold', 'input_modalities': ['text'],
                      'context_window': {'tokens': 65536}}) as engine_check,
                  redirect_stdout(io.StringIO())):
                result = wizard.main(['--home', str(home), '--config', str(plan), '--check'])
            self.assertEqual(engine_check.call_args.args[2]['name'], profile['name'])
            self.assertTrue(result['artifact_validated'])
            self.assertFalse((home/'.config').exists())

    def test_managed_install_preserves_the_deepseek_source_license(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);model = root/'model';model.mkdir()
            # The tiny test engine accepts the profile's dimensions; no model is loaded.
            (model/'config.json').write_text('{"model_type":"deepseek_v41","hidden_size":5120}')
            bundle = make_bundle(root/'bundle')
            manifest = json.loads((bundle/'manifest.json').read_text())
            manifest['model_types'] = ['deepseek_v41']
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            from test_deepseek_setup import legacy_profile
            source = legacy_profile(root/'legacy')
            original = bundled_profile_root()/'deepseek41-flash1017-v1'
            (source/'LICENSE').write_bytes((original/'LICENSE').read_bytes())
            # This tiny synthetic checkpoint exercises license copying; exact
            # native checkpoint binding is covered separately.
            with mock.patch('owner_mode.native_setup.checkpoint_matches'):
                deployment = prepare_engine(root/'home', {'model_path': str(model), 'bundle': str(bundle),
                                            'devices': ['0']}, source,
                                            model_id='deepseek-ai/DeepSeek-V4.1-Flash-1017')
            installed = Path(deployment['profile_path'])/'LICENSE'
            self.assertEqual(installed.read_bytes(), (source/'LICENSE').read_bytes())


if __name__ == '__main__':
    unittest.main()
