# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Model rules are registered independently of setup and request policy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_setup import wizard
from owner_mode.model_adapters import (ModelAdapter, ModelAdapterRegistry, DEFAULT_MODEL_ADAPTERS,
                                      model_adapters, register_model_adapter, using_model_adapters)
from owner_mode.model_metadata import model_request_compatibility, configure_hermes_reasoning, compatible_request


class ModelAdaptersTests(unittest.TestCase):
    def test_new_model_uses_existing_codec_without_central_branch(self):
        registry = DEFAULT_MODEL_ADAPTERS.with_adapter(ModelAdapter(
            'fixture', ('organization/new-model',), request_kind='chat_template_reasoning_v1',
            request_efforts=('low', 'max'), hermes_efforts=('low', 'max'), hermes_default_effort='max'))
        rule = model_request_compatibility('organization/new-model', registry=registry)
        self.assertEqual(rule, {'kind': 'chat_template_reasoning_v1', 'efforts': ['low', 'max'],
                                'model': 'organization/new-model'})
        body = {'model': 'organization/new-model', 'reasoning_effort': 'max'}
        self.assertEqual(compatible_request(body, rule)['chat_template_kwargs'],
                         {'enable_thinking': True, 'reasoning_effort': 'max'})
        config = {'agent': {'reasoning_effort': 'medium'}}
        self.assertEqual(configure_hermes_reasoning(config, 'organization/new-model', registry=registry)['effort'], 'max')
        self.assertEqual(config['agent']['reasoning_effort'], 'medium')

    def test_explicit_rule_overrides_deepseek_legacy_default(self):
        model = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
        declaration = {'kind': 'chat_template_reasoning_v1', 'efforts': ['low', 'high', 'max']}
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            with patch.object(wizard, 'free_port', return_value=48123):
                wizard.provision(home, ['openclaw'], 'http://127.0.0.1:48124/v1', model,
                                 'profile', [], True, model_info={'request_compatibility': declaration})
            config = json.loads((home/'.config/owner-mode/openclaw-bridge.json').read_text())
            self.assertEqual(config['routes']['head']['request_compatibility'], {**declaration, 'model': model})

    def test_legacy_default_is_typed_and_disables_only_explicit_off(self):
        model = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
        rule = model_request_compatibility(model)
        self.assertEqual(rule, {'kind': 'boolean_thinking_v1', 'model': model})
        self.assertEqual(compatible_request({'model': model, 'reasoning_effort': 'none'}, rule),
                         {'model': model, 'chat_template_kwargs': {'enable_thinking': False}})
        high = {'model': model, 'reasoning_effort': 'max'}
        self.assertIs(compatible_request(high, rule), high)

    def test_registry_rejects_ambiguous_models(self):
        first = ModelAdapter('first', ('organization/model',))
        with self.assertRaises(ValueError):
            ModelAdapterRegistry((first, ModelAdapter('second', ('organization/model',))))
        self.assertIsNone(DEFAULT_MODEL_ADAPTERS.get(['invalid']))

    def test_registered_model_reaches_normal_provisioning_and_requests(self):
        adapter = ModelAdapter('installed', ('organization/installed-model',),
                               request_kind='boolean_thinking_v1')
        with using_model_adapters(DEFAULT_MODEL_ADAPTERS), tempfile.TemporaryDirectory() as temp:
            register_model_adapter(adapter)
            with self.assertRaises(ValueError):
                register_model_adapter(adapter)
            with patch.object(wizard, 'free_port', return_value=48123):
                wizard.provision(Path(temp), ['openclaw'], 'http://127.0.0.1:48124/v1',
                                 adapter.model_ids[0], 'profile', [], True)
            config = json.loads((Path(temp)/'.config/owner-mode/openclaw-bridge.json').read_text())
            rule = config['routes']['head']['request_compatibility']
            self.assertEqual(compatible_request({'model': adapter.model_ids[0], 'reasoning_effort': 'none'}, rule),
                             {'model': adapter.model_ids[0], 'chat_template_kwargs': {'enable_thinking': False}})
        self.assertIsNone(model_adapters().get(adapter.model_ids[0]))


if __name__ == '__main__':
    unittest.main()
