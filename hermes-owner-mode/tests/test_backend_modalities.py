# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Explicit model-type capabilities reach automatic native harness setup."""
import copy
import json
from pathlib import Path
import secrets
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile, wizard
from owner_mode.automation import run_plan
from owner_mode.backend_setup import KIT_ROOT, backend_catalog, _input_modalities
from owner_mode.engine_setup import check_engine_setup, private_json
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError


class BackendModalities(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.model, self.draft = self.root/'model', self.root/'draft'
        self.model.mkdir()
        self.draft.mkdir()
        self.family('qwen3_5')
        (self.draft/'config.json').write_text(json.dumps({'architectures':['DFlash2DraftModel']}))
        self.profile = make_profile(self.root/'profile')
        profile = json.loads((self.profile/'profile.json').read_text())
        profile['sites'][0]['tensor'] = 'model.language_model.layers.0.self_attn.o_proj.weight'
        (self.profile/'profile.json').write_text(json.dumps(profile))
        self.spec = {'kind':'sglang', 'model_path':str(self.model),
                     'draft_model':str(self.draft), 'devices':['0']}
        self.kit = KIT_ROOT/'sglang'
        self.metadata = json.loads((self.kit/'backend.json').read_text())

    def family(self, name):
        (self.model/'config.json').write_text(json.dumps({'model_type':name,'hidden_size':2}))

    def check(self):
        with mock.patch('owner_mode.backend_setup.active_speculation', return_value=None):
            return check_engine_setup(self.home, self.spec, validate_profile(self.profile), model_id='bench')

    def automatic_setup(self, *, auxiliary=False):
        before = {'model':{'default':'previous-model','supports_vision':True}}
        if auxiliary:
            before['auxiliary'] = {'vision':{'provider':'existing-vision','model':'image-model'}}
        private_json(self.home/'.hermes/config.yaml', before)
        key = self.root/'engine-key'
        key.write_bytes(secrets.token_bytes(32))
        key.chmod(0o600)
        plan = self.root/'plan.json'
        private_json(plan, {'model':'bench','profile':str(self.profile),'engine':self.spec,
            'harnesses':['hermes','openclaw'],'tools_default':False,
            'contacts':[{'channel':'sms','account':'default','contact':'+15550000011'}]})
        found = {name:{'detected':True,'runtime_compatible':True,'context_requirements':[]}
                 for name in ('hermes','openclaw')}
        deployment = {'base_url':'http://127.0.0.1:12345/v1','key_file':str(key)}
        # Only engine installation/network calls are fixtures. Capability lookup,
        # automatic propagation, and both configuration writers are real.
        with (mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.automation.prepare_engine', return_value=deployment),
              mock.patch('owner_mode.automation.activate_engine', return_value=deployment),
              mock.patch('owner_mode.automation.check_engine', return_value={'status':'active'}),
              mock.patch('owner_mode.automation.register_openclaw')):
            result = run_plan(plan, self.home, found, wizard.provision, activate=True)
        hermes = json.loads((self.home/'.hermes/config.yaml').read_text())
        openclaw = json.loads((self.home/'.openclaw/openclaw.json').read_text())
        return result, hermes, openclaw, before

    def test_multimodal_type_reaches_automatic_setup_and_native_writers(self):
        self.assertEqual(self.check()['input_modalities'], ['text','image'])
        result, hermes, openclaw, _ = self.automatic_setup()
        self.assertEqual(result['input_modalities'], ['text','image'])
        self.assertIs(hermes['model']['supports_vision'], True)
        self.assertEqual(hermes['agent']['image_input_mode'], 'auto')
        self.assertTrue(result['hermes_images']['native_images'])
        provider = hermes['providers'][hermes['model']['provider']]
        self.assertEqual(provider['models'], {'bench':{'supports_vision':True}})
        self.assertEqual(openclaw['models']['providers']['owner-mode']['models'][0]['input'], ['text','image'])

    def test_text_type_stays_text_only_and_preserves_auxiliary_route(self):
        self.family('qwen3_5_text')
        self.assertEqual(self.check()['input_modalities'], ['text'])
        result, hermes, openclaw, before = self.automatic_setup(auxiliary=True)
        self.assertIs(hermes['model']['supports_vision'], False)
        self.assertEqual(hermes['auxiliary'], before['auxiliary'])
        self.assertFalse(result['hermes_images']['native_images'])
        self.assertEqual(result['hermes_images']['status'], 'preserved explicit image route')
        self.assertEqual(openclaw['models']['providers']['owner-mode']['models'][0]['input'], ['text'])

    def test_multimodal_type_does_not_replace_explicit_auxiliary_route(self):
        result, hermes, _, before = self.automatic_setup(auxiliary=True)
        self.assertIs(hermes['model']['supports_vision'], True)
        self.assertEqual(hermes['auxiliary'], before['auxiliary'])
        self.assertFalse(result['hermes_images']['native_images'])

    def test_missing_typed_entry_never_inherits_general_vision(self):
        self.family('future_model')
        metadata = copy.deepcopy(self.metadata)
        metadata['model_types'].append('future_model')
        metadata['input_modalities'] = ['text','image']
        with mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit,metadata,'sglang')])):
            self.assertIsNone(self.check()['input_modalities'])

    def test_invalid_typed_declarations_fail_before_writes(self):
        for value in (None, [], {}, {'unlisted':['text']}, {'qwen3_5':['image']},
                      {'qwen3_5':['text','text']}, {'qwen3_5':['text','video']},
                      {'qwen3_5_text':None}, {'qwen3_5':[{'text':True}]}):
            with self.subTest(value=value):
                metadata = copy.deepcopy(self.metadata)
                metadata['input_modalities_by_model_type'] = value
                with mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit,metadata,'sglang')])):
                    with self.assertRaisesRegex(SetupError, 'input.modalit'):
                        self.check()
                self.assertFalse((self.home/'.config').exists())

    def test_catalog_exposes_typed_capabilities(self):
        entry = next(row for row in backend_catalog() if row['kit'] == str(self.kit))
        self.assertEqual(entry['input_modalities_by_model_type'],
                         {'qwen3_5':['text','image'],'qwen3_5_text':['text']})

    def test_gemma_capabilities_keep_the_text_variant_text_only(self):
        metadata = json.loads((KIT_ROOT/'vllm-gemma4/backend.json').read_text())
        entry = next(row for row in backend_catalog()
                     if row['kit'] == str(KIT_ROOT/'vllm-gemma4'))
        self.assertEqual(entry['input_modalities_by_model_type'],
                         {'gemma4':['text','image'],'gemma4_text':['text']})
        self.assertEqual(_input_modalities(metadata, 'gemma4'), ['text','image'])
        self.assertEqual(_input_modalities(metadata, 'gemma4_text'), ['text'])
        self.assertIsNone(_input_modalities(metadata, 'unqualified_variant'))


if __name__ == '__main__':
    unittest.main()
