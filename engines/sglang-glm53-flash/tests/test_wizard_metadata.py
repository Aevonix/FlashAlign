# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The GLM declarations reach both real configuration writers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
REPO = KIT.parents[1]
sys.path.insert(0, str(REPO / 'hermes-owner-mode/tests'))
from test_setup import wizard
from owner_mode.model_metadata import (compatible_request, declared_openclaw_compat,
                                       declared_openclaw_request_policy,
                                       declared_request_compatibility)
from owner_mode.setup_runtime import SetupError


class WizardMetadata(unittest.TestCase):
    def test_max_effort_survives_both_writers_and_request_mapping(self):
        metadata = json.loads((KIT / 'backend.json').read_text())
        model = 'zai-org/GLM-5.3-Flash'
        for model_type in metadata['model_types']:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as temporary:
                home = Path(temporary)
                hp = home / '.hermes/config.yaml'
                hp.parent.mkdir()
                hp.write_text(json.dumps({'model': {'default': model},
                    'agent': {'reasoning_overrides': {model: 'max'}}}))
                op = home / '.openclaw/openclaw.json'
                op.parent.mkdir()
                op.write_text(json.dumps({'agents': {'defaults': {'thinkingDefault': 'max'}}}))
                info = {'input': ['text', 'image'], 'contextWindow': 65536,
                        'openclaw_compat': declared_openclaw_compat(metadata, model_type),
                        'openclaw_request_policy': declared_openclaw_request_policy(metadata, model_type),
                        'request_compatibility': declared_request_compatibility(metadata, model_type)}
                with patch.object(wizard, 'free_port', side_effect=(18134, 18135)):
                    wizard.provision(home, ['hermes', 'openclaw'], 'http://127.0.0.1:18136/v1',
                                     model, 'synthetic-profile', [], True, model_info=info)
                hermes = json.loads(hp.read_text())
                openclaw = json.loads(op.read_text())
                hroute = hermes['plugins']['entries']['owner-mode']['settings']['routes']['head']
                oroute = json.loads((home / '.config/owner-mode/openclaw-bridge.json').read_text())['routes']['head']
                self.assertEqual(hermes['agent']['reasoning_overrides'][model], 'max')
                self.assertEqual(openclaw['agents']['defaults']['thinkingDefault'], 'max')
                compat = openclaw['models']['providers']['owner-mode']['models'][0]['compat']
                self.assertEqual(compat['thinkingFormat'], 'openai')
                self.assertIn('max', compat['supportedReasoningEfforts'])
                self.assertNotIn('none', compat['supportedReasoningEfforts'])
                for route in (hroute, oroute):
                    request = compatible_request({'model': model, 'reasoning_effort': 'max'},
                                                 route['request_compatibility'])
                    self.assertEqual(request['chat_template_kwargs'],
                                     {'enable_thinking': True, 'reasoning_effort': 'max'})

    def test_saved_openclaw_thinking_off_is_rejected_before_writes(self):
        metadata = json.loads((KIT / 'backend.json').read_text())
        for model_type in metadata['model_types']:
            for choice in ('off', 'none', False):
                with self.subTest(model_type=model_type, choice=choice), tempfile.TemporaryDirectory() as temporary:
                    home = Path(temporary)
                    path = home / '.openclaw/openclaw.json'
                    path.parent.mkdir()
                    original = json.dumps({'agents': {'defaults': {'thinkingDefault': choice}}})
                    path.write_text(original)
                    info = {'openclaw_request_policy': declared_openclaw_request_policy(metadata, model_type)}
                    with self.assertRaisesRegex(SetupError, 'does not support thinking off'):
                        wizard.provision(home, ['openclaw'], 'http://127.0.0.1:18136/v1',
                                         'zai-org/GLM-5.3-Flash', 'synthetic-profile', [], True, model_info=info)
                    self.assertEqual(path.read_text(), original)


if __name__ == '__main__':
    unittest.main()
