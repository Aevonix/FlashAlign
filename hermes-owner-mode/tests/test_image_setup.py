# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Serving-lane image capabilities reach native harness configuration."""
import json
from pathlib import Path
import tempfile
import unittest

from test_setup import wizard
from owner_mode.model_metadata import configure_hermes_images
from owner_mode.setup_runtime import SetupError


class ImageSetup(unittest.TestCase):
    def test_declared_multimodal_lane_configures_native_capability(self):
        config={}
        result=configure_hermes_images(config,['text','image'])
        self.assertIs(config['model']['supports_vision'],True)
        self.assertEqual(config['agent']['image_input_mode'],'auto')
        self.assertTrue(result['native_images'])

    def test_text_only_lane_does_not_inherit_old_model_vision(self):
        config={'model':{'supports_vision':True}}
        result=configure_hermes_images(config,['text'])
        self.assertIs(config['model']['supports_vision'],False)
        self.assertFalse(result['native_images'])

    def test_explicit_text_and_auxiliary_choices_are_preserved(self):
        for config in ({'agent':{'image_input_mode':'text'}},
                       {'auxiliary':{'vision':{'provider':'fixture','model':'aux'}}}):
            before=json.dumps(config,sort_keys=True)
            result=configure_hermes_images(config,['text','image'])
            self.assertFalse(result['native_images'])
            self.assertEqual(result['status'],'preserved explicit image route')
            if 'auxiliary' in config:
                self.assertEqual(config['auxiliary'],json.loads(before)['auxiliary'])
            else:
                self.assertEqual(config['agent']['image_input_mode'],'text')

    def test_missing_endpoint_modalities_are_not_invented(self):
        config={'model':{'default':'fixture'}}
        before=json.dumps(config)
        result=configure_hermes_images(config,None)
        self.assertEqual(result['status'],'unverified')
        self.assertEqual(json.dumps(config),before)

    def test_forced_native_on_text_lane_reports_a_conflict(self):
        config={'agent':{'image_input_mode':'native'}}
        with self.assertRaisesRegex(SetupError,'text-only'):
            configure_hermes_images(config,['text'])
        self.assertEqual(config['agent']['image_input_mode'],'native')

    def test_invalid_modalities_are_rejected(self):
        with self.assertRaises(SetupError):
            configure_hermes_images({},['image'])

    def test_actual_writer_sets_hermes_and_openclaw_from_declared_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary)
            result=wizard.provision(home,['hermes','openclaw'],'http://127.0.0.1:12345/v1',
                'fixture-model','fixture-profile',[],False,
                model_info={'input':['text','image'],'contextWindow':65536,'maxTokens':32000})
            hermes=json.loads((home/'.hermes/config.yaml').read_text())
            openclaw=json.loads((home/'.openclaw/openclaw.json').read_text())
            self.assertTrue(hermes['model']['supports_vision'])
            self.assertTrue(result['hermes_images']['native_images'])
            descriptor=openclaw['models']['providers']['owner-mode']['models'][0]
            self.assertEqual(descriptor['input'],['text','image'])
            self.assertEqual(descriptor['maxTokens'],32000)


if __name__=='__main__':
    unittest.main()
