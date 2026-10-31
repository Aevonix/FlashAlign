# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Context compatibility checks use synthetic configuration and no model calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import wizard, make_profile
from test_engine_setup import make_bundle
from owner_mode.automation import run_plan
from owner_mode.engine_setup import check_engine_setup, private_json
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import (SetupError, check_harness_contexts,
    harness_context_requirements, managed_context_window, model_rows)


class ContextSetup(unittest.TestCase):
    def test_implicit_hermes_minimum_is_checked_for_managed_engines(self):
        found = {'hermes': {'context_requirements': []}}
        for tokens in (None, 4096, 16384):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(SetupError, 'runtime minimum context=64000'):
                check_harness_contexts(['hermes'], found, {'tokens': tokens}, managed=True)
        result = check_harness_contexts(['hermes'], found, {'tokens': 65536}, managed=True)
        self.assertEqual(result['harnesses']['hermes'][0]['tokens'], 64000)
        self.assertEqual(found['hermes']['context_requirements'], [])
        self.assertEqual(check_harness_contexts([], found, {'tokens': 4096}, managed=True)['harnesses'], {})

    def test_unknown_remote_context_is_reported_without_guessing_it(self):
        found = {'hermes': {'context_requirements': []}}
        result = check_harness_contexts(['hermes'], found, {'tokens': None})
        self.assertIn('unverified', result['status'])
        self.assertIsNone(result['engine']['tokens'])
        with self.assertRaisesRegex(SetupError, 'runtime minimum context=64000'):
            check_harness_contexts(['hermes'], found, {'tokens': 8192})

    def test_managed_plan_checks_the_implicit_floor_before_any_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root/'home'
            home.mkdir()
            profile = make_profile(root/'profile')
            plan = root/'plan.json'
            private_json(plan, {'model': 'bench', 'profile': str(profile), 'harnesses': ['hermes'],
                'contacts': [{'channel': 'sms', 'account': 'default', 'contact': '+15550000101'}],
                'tools_default': False, 'engine': {'kind': 'vllm', 'model_path': str(root/'model')}})
            found = {'hermes': {'detected': True, 'runtime_compatible': True, 'context_requirements': []}}
            provision = mock.Mock()
            for tokens in (None, 8192):
                with (self.subTest(tokens=tokens),
                      mock.patch('owner_mode.automation.check_engine_setup', return_value={
                          'kind': 'vllm', 'context_window': {'tokens': tokens}}),
                      mock.patch('owner_mode.automation.prepare_engine') as prepare):
                    with self.assertRaisesRegex(SetupError, 'runtime minimum context=64000'):
                        run_plan(plan, home, found, provision, apply=True)
                    prepare.assert_not_called()
                    provision.assert_not_called()
                    self.assertEqual(list(home.iterdir()), [])

    def test_hermes_explicit_limits_are_read_without_rewriting_them(self):
        config = {'model': {'context_length': 737280, 'max_tokens': 32000},
                  'agent': {'reasoning_effort': 'low'}}
        original = json.dumps(config)
        result = harness_context_requirements('hermes', config)
        self.assertEqual([(row['kind'], row['tokens']) for row in result],
                         [('context', 737280), ('output', 32000)])
        self.assertEqual(json.dumps(config), original)

    def test_only_selected_openclaw_model_limits_are_checked(self):
        config = {'agents': {'defaults': {'model': {'primary': 'provider/org/model'}}},
                  'models': {'providers': {'provider': {'models': [
                      {'id': 'unused', 'contextWindow': 999999},
                      {'id': 'org/model', 'contextWindow': 65536, 'maxTokens': 32000}]}}}}
        rows = harness_context_requirements('openclaw', config)
        self.assertEqual([row['tokens'] for row in rows], [65536, 32000])

    def test_unknown_or_smaller_engine_cannot_replace_large_harness(self):
        found = {'hermes': {'context_requirements': harness_context_requirements('hermes',
            {'model': {'context_length': 737280, 'max_tokens': 32000}})}}
        for limit in (None, 65536):
            with self.assertRaisesRegex(SetupError, '737280'):
                check_harness_contexts(['hermes'], found, {'tokens': limit, 'source': 'fixture'})

    def test_large_output_budget_is_checked_separately(self):
        found = {'hermes': {'context_requirements': harness_context_requirements('hermes',
            {'model': {'context_length': 32768, 'max_tokens': 70000}})}}
        with self.assertRaisesRegex(SetupError, 'max_tokens=70000'):
            check_harness_contexts(['hermes'], found, {'tokens': 65536})

    def test_compatible_limits_and_direct_mode_are_reported(self):
        found = {'hermes': {'context_requirements': harness_context_requirements('hermes',
            {'model': {'context_length': 65536, 'max_tokens': 32000}})}}
        result = check_harness_contexts(['hermes'], found, {'tokens': 65536})
        self.assertEqual(result['status'], 'compatible')
        self.assertEqual(check_harness_contexts([], found, {'tokens': 4096})['harnesses'], {})

    def test_invalid_limit_does_not_disable_the_check(self):
        with self.assertRaisesRegex(SetupError, 'positive token count'):
            harness_context_requirements('hermes', {'model': {'context_length': 'unlimited'}})

    def test_discovery_reports_context_without_any_endpoint_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary);config = home/'.hermes/config.yaml';config.parent.mkdir()
            config.write_text(json.dumps({'model': {'context_length': 737280, 'max_tokens': 32000}}))
            found = wizard.discover(home, which=lambda _: None)
            self.assertEqual(found['hermes']['context_requirements'][0]['tokens'], 737280)

    def test_managed_deepseek_default_and_explicit_override_are_visible(self):
        metadata = {'default_arguments': ['--max-model-len', '65536']}
        self.assertEqual(managed_context_window('vllm', metadata)['tokens'], 65536)
        for arguments in (['--max-model-len=8192'], ['--max_model_len', '8192'],
                          ['--max_model_len=8192']):
            with self.subTest(arguments=arguments):
                self.assertEqual(managed_context_window('vllm', metadata, arguments)['tokens'], 8192)
        self.assertEqual(managed_context_window('vllm', {
            'default_arguments': ['--max_model_len=65536']})['tokens'], 65536)
        with self.assertRaisesRegex(SetupError, 'complete vLLM option'):
            managed_context_window('vllm', metadata, ['--max-model-l', '8192'])

    def test_sglang_fallback_uses_the_tested_variant_context(self):
        self.assertEqual(managed_context_window('sglang', {'qualification': {'tested_context_length': 2048}})['tokens'], 2048)

    def test_llamacpp_context_is_divided_between_parallel_slots(self):
        self.assertEqual(managed_context_window('llamacpp', {})['tokens'], 2048)
        self.assertEqual(managed_context_window('llamacpp', {}, ['--ctx-size', '16384', '--parallel', '4'])['tokens'], 4096)
        self.assertEqual(managed_context_window('llamacpp', {}, ['-c', '1024', '-np', '4'])['tokens'], 256)

    def test_vllm_endpoint_context_field_is_normalized(self):
        with mock.patch('owner_mode.setup_runtime.request', return_value=(200, {},
                        {'data': [{'id': 'fixture', 'max_model_len': 65536}]})):
            self.assertEqual(model_rows('http://127.0.0.1:1/v1')[0]['contextWindow'], 65536)

    def test_checkpoint_position_limit_does_not_prove_usable_tensorfold_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model=root/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps({'model_type':'glm_moe_dsa','hidden_size':2,
                                                        'max_position_embeddings':1000000}))
            profile=make_profile(root/'profile');bundle=make_bundle(root/'bundle')
            data=json.loads((bundle/'manifest.json').read_text());data['arguments']=[]
            (bundle/'manifest.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(SetupError,'tested context'):
                check_engine_setup(root,{'model_path':str(model),'bundle':str(bundle),'devices':['0']},
                                   validate_profile(profile),model_id='bench')

    def test_openclaw_output_budget_is_not_raised_to_hermes_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);home=root/'home';home.mkdir()
            profile=make_profile(root/'profile');plan=root/'plan.json'
            private_json(plan,{'model':'bench','profile':str(profile),'harnesses':['hermes','openclaw'],
                'contacts':[{'channel':'sms','account':'default','contact':'+15550000101'}],
                'tools_default':False,'engine':{'kind':'vllm','model_path':str(root/'model')}})
            found={name:{'detected':True,'runtime_compatible':True,'context_requirements':[
                {'field':'output','kind':'output','tokens':value}]} for name,value in [('hermes',32000),('openclaw',8192)]}
            deployment={'base_url':'http://127.0.0.1:1/v1','key_file':str(root/'unused-key')}
            provision=mock.Mock(return_value={})
            with (mock.patch('owner_mode.automation.check_engine_setup',return_value={
                    'kind':'vllm','context_window':{'tokens':65536}}),
                  mock.patch('owner_mode.automation.prepare_engine',return_value=deployment),
                  mock.patch('owner_mode.automation.activate_engine',return_value=deployment),
                  mock.patch('owner_mode.automation.check_engine',return_value={'status':'active'}),
                  mock.patch('owner_mode.automation.register_openclaw')):
                run_plan(plan,home,found,provision,activate=True)
            self.assertEqual(provision.call_args.kwargs['model_info']['maxTokens'],8192)


if __name__ == '__main__':
    unittest.main()
