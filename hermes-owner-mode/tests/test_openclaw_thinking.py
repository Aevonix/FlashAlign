# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Thinking wire formats come from a matching model, never an engine-wide guess."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import wizard
import test_backend_modalities as modalities
import test_tensorfold_metadata as tensorfold
from owner_mode.backend_setup import backend_catalog
from owner_mode.model_metadata import configure_openclaw_compat, declared_openclaw_compat
from owner_mode.setup_runtime import SetupError


FORMAT = {'thinkingFormat': 'qwen-chat-template'}


class ThinkingCompatibility(unittest.TestCase):
    def test_declarations_select_exact_supported_type(self):
        metadata = {'model_types': ['qwen', 'other'],
                    'openclaw_compat_by_model_type': {'qwen': FORMAT}}
        self.assertEqual(declared_openclaw_compat(metadata, 'qwen'), FORMAT)
        self.assertIsNone(declared_openclaw_compat(metadata, 'other'))
        self.assertIsNone(declared_openclaw_compat({'thinkingFormat': 'qwen'}, 'qwen'))
        self.assertEqual(declared_openclaw_compat({'model_architectures': ['qwen'],
            'openclaw_compat_by_model_type': {'qwen': FORMAT}}, 'qwen'), FORMAT)

    def test_invalid_or_unlisted_declarations_fail(self):
        for value in (None, [], {'unlisted': FORMAT}, {'qwen': None},
                      {'qwen': {'thinkingFormat': 'unknown'}},
                      {'qwen': {'thinkingFormat': ['qwen']}},
                      {'qwen': {'thinkingFormat': 'qwen', 'headers': {}}}):
            with self.subTest(value=value), self.assertRaises(SetupError):
                declared_openclaw_compat({'model_types': ['qwen'],
                    'openclaw_compat_by_model_type': value}, 'qwen')

    def test_reasoning_efforts_are_bounded_and_copied(self):
        value = {'thinkingFormat': 'openai', 'supportedReasoningEfforts': ['none', 'low', 'high']}
        descriptor, _ = configure_openclaw_compat({}, 'model', {'id': 'model'}, {'openclaw_compat': value})
        self.assertEqual(descriptor['compat'], value)
        descriptor['compat']['supportedReasoningEfforts'].append('max')
        self.assertEqual(value['supportedReasoningEfforts'], ['none', 'low', 'high'])
        for efforts in (None, [], 'none', ['none', 'none'], ['unexpected'], [1], [{}]):
            with self.subTest(efforts=efforts), self.assertRaises(SetupError):
                configure_openclaw_compat({}, 'model', {'id': 'model'}, {'openclaw_compat': {
                    'thinkingFormat': 'openai', 'supportedReasoningEfforts': efforts}})

    def test_explicit_selected_model_compat_wins_and_is_copied(self):
        config = {'agents': {'defaults': {'model': {'primary': 'selected/org/model'}}},
                  'models': {'providers': {
                      'selected': {'models': [{'id': 'org/model', 'compat': {
                          'thinkingFormat': 'qwen', 'supportsDeveloperRole': False}}]},
                      'other': {'models': [{'id': 'org/model', 'compat': FORMAT}]}}}}
        before = copy.deepcopy(config)
        descriptor, report = configure_openclaw_compat(config, 'org/model', {'id': 'org/model'},
                                                     {'openclaw_compat': FORMAT})
        self.assertEqual(descriptor['compat'], before['models']['providers']['selected']['models'][0]['compat'])
        self.assertEqual(report['source'], 'explicit canonical model')
        descriptor['compat']['thinkingFormat'] = 'changed'
        self.assertEqual(config, before)

    def test_explicit_empty_compat_is_not_overwritten(self):
        config = {'models': {'providers': {'old': {'models': [{'id': 'model', 'compat': {}}]}}}}
        descriptor, _ = configure_openclaw_compat(config, 'model', {'id': 'model'},
                                                  {'openclaw_compat': FORMAT})
        self.assertEqual(descriptor['compat'], {})

    def test_different_model_does_not_supply_compat(self):
        config = {'models': {'providers': {'old': {'models': [{'id': 'other', 'compat': FORMAT}]}}}}
        descriptor, report = configure_openclaw_compat(config, 'model', {'id': 'model'})
        self.assertNotIn('compat', descriptor)
        self.assertFalse(report['classified'])

    def test_conflicting_unselected_matching_models_fail(self):
        config = {'models': {'providers': {
            'a': {'models': [{'id': 'model', 'compat': FORMAT}]},
            'b': {'models': [{'id': 'model', 'compat': {'thinkingFormat': 'qwen'}}]}}}}
        with self.assertRaisesRegex(SetupError, 'Conflicting'):
            configure_openclaw_compat(config, 'model', {'id': 'model'})

    def test_no_engine_wide_family_inference(self):
        for info in ({}, {'backend': 'tensorfold'}, {'backend': 'vllm'},
                     {'backend': 'sglang'}, {'backend': 'llamacpp'},
                     {'model_type': 'qwen4_exp'}, {'openclaw_compat': None}):
            with self.subTest(info=info):
                value, report = configure_openclaw_compat({}, 'model', {'id': 'model'}, info)
                self.assertNotIn('compat', value)
                self.assertFalse(report['classified'])

    def test_actual_writer_retains_override_and_derived_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            path = home/'.openclaw/openclaw.json'
            path.parent.mkdir()
            original = {'models': {'providers': {'old': {'models': [
                {'id': 'model', 'compat': {'thinkingFormat': 'qwen'}}]}}}}
            path.write_text(json.dumps(original))
            with mock.patch.object(wizard, 'free_port', return_value=48123):
                first = wizard.provision(home, ['openclaw'], 'http://127.0.0.1:48124/v1',
                    'model', 'synthetic-profile', [], True, model_info={'openclaw_compat': FORMAT})
                again = wizard.provision(home, ['openclaw'], 'http://127.0.0.1:48124/v1',
                    'model', 'synthetic-profile', [], True, model_info={'openclaw_compat': FORMAT})
            config = json.loads(path.read_text())
            self.assertEqual(config['models']['providers']['old'], original['models']['providers']['old'])
            model = config['models']['providers']['owner-mode']['models'][0]
            plugin = config['plugins']['entries']['owner-mode']['config']['model']
            self.assertEqual(model['compat'], {'thinkingFormat': 'qwen'})
            self.assertEqual(model, plugin)
            self.assertEqual(first['openclaw_model_compatibility']['source'], 'explicit canonical model')
            self.assertEqual(again['openclaw_model_compatibility']['source'], 'explicit canonical model')


class BackendThinking(unittest.TestCase):
    setUp = modalities.BackendModalities.setUp
    family = modalities.BackendModalities.family
    check = modalities.BackendModalities.check
    automatic_setup = modalities.BackendModalities.automatic_setup
    def test_validated_type_reaches_automatic_writer(self):
        self.assertEqual(self.check()['openclaw_compat'], FORMAT)
        _, _, config, _ = self.automatic_setup()
        self.assertEqual(config['models']['providers']['owner-mode']['models'][0]['compat'], FORMAT)
        self.assertEqual(config['plugins']['entries']['owner-mode']['config']['model']['compat'], FORMAT)

    def test_unknown_type_does_not_inherit_a_declared_format(self):
        self.family('future_model')
        metadata = copy.deepcopy(self.metadata)
        metadata['model_types'].append('future_model')
        with mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit, metadata, 'sglang')])):
            self.assertIsNone(self.check()['openclaw_compat'])

    def test_catalog_exposes_only_model_scoped_formats(self):
        rows = backend_catalog()
        for row in rows:
            declarations = row.get('openclaw_compat_by_model_type', {})
            self.assertTrue(set(declarations) <= set(row['model_types']))
            for value in declarations.values():
                self.assertTrue(set(value) <= {'thinkingFormat', 'supportedReasoningEfforts'})


class TensorFoldThinking(unittest.TestCase):
    setUp = tensorfold.TensorFoldMetadata.setUp
    manifest = tensorfold.TensorFoldMetadata.manifest
    check = tensorfold.TensorFoldMetadata.check
    def test_tensorfold_matching_declaration_passes_without_operator_changes(self):
        self.manifest(openclaw_compat_by_model_type={'glm_moe_dsa': FORMAT})
        self.assertEqual(self.check()['openclaw_compat'], FORMAT)

    def test_tensorfold_absent_declaration_remains_unclassified(self):
        self.assertIsNone(self.check()['openclaw_compat'])

    def test_tensorfold_bad_declaration_rejected_before_staging(self):
        self.manifest(openclaw_compat_by_model_type={'unlisted': FORMAT})
        with self.assertRaisesRegex(SetupError, 'model-scoped'):
            self.check()
        self.assertFalse((self.home/'.config').exists())


if __name__ == '__main__':
    unittest.main()
