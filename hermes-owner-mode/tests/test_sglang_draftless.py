# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Draftless NGRAM capability does not weaken existing draft-model checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_setup import wizard
from owner_mode.backend_setup import KIT_ROOT, _native_profile_layers, _speculation, prepare_backend
from owner_mode.model_metadata import declared_request_compatibility
from owner_mode.setup_runtime import SetupError


class DraftlessSGLang(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.spec = {'model_path': str(self.home/'model')}
        self.metadata = {'speculative_algorithms': ['NGRAM'],
                         'draftless_speculative_algorithms': ['NGRAM']}

    def check(self, spec=None, metadata=None, active=None):
        with patch('owner_mode.backend_setup.active_speculation', return_value=active):
            return _speculation(self.home, spec or self.spec, 'sglang', 'test/model',
                                metadata=self.metadata if metadata is None else metadata)

    def test_ngram_needs_no_draft_model(self):
        self.assertEqual(self.check(), {'draft_algorithm': 'NGRAM'})

    def test_active_ngram_settings_are_preserved(self):
        active = {'kind': 'sglang', 'draft_algorithm': 'NGRAM',
                  'speculative_arguments': ['--speculative-num-draft-tokens', '3']}
        self.assertEqual(self.check(active=active), {k:v for k,v in active.items() if k!='kind'})

    def test_external_draft_and_active_method_change_are_rejected(self):
        with self.assertRaisesRegex(SetupError, 'external draft'):
            self.check({**self.spec, 'draft_model': str(self.home/'draft')})
        with self.assertRaisesRegex(SetupError, 'unsupported'):
            self.check(active={'kind': 'sglang', 'draft_algorithm': 'DFLASH'})

    def test_capability_is_explicit_and_limited_to_ngram(self):
        for value in ('NGRAM', ['DFLASH'], ['NGRAM', 'NGRAM'], ['OTHER'], [{}]):
            with self.subTest(value=value), self.assertRaises(SetupError):
                self.check(metadata={'speculative_algorithms':['NGRAM','DFLASH'],
                                     'draftless_speculative_algorithms':value})

    def test_dflash_still_requires_draft_selection(self):
        with (patch('owner_mode.backend_setup.draft_settings', return_value={'draft_model':'unchanged'}) as draft,
              patch('owner_mode.backend_setup.active_speculation', return_value=None)):
            result=_speculation(self.home,self.spec,'sglang','test/model',
                                metadata={'speculative_algorithms':['DFLASH']})
        draft.assert_called_once()
        self.assertEqual(result, {'draft_model':'unchanged','draft_algorithm':'DFLASH'})

    def test_managed_launch_keeps_ngram_three_without_external_draft(self):
        kit=KIT_ROOT/'sglang-gemma4'
        profile=KIT_ROOT.parent/'profiles/gemma4-26b-a4b-v1'
        metadata=json.loads((kit/'backend.json').read_text())
        checked={'kind':'sglang','kit':str(kit),'metadata':metadata,
                 'python':{'executable':sys.executable},'model':{'path':str(self.home/'model')},
                 'drafting':{'draft_algorithm':'NGRAM'},'arguments':[],'devices':['0']}
        with (patch('owner_mode.backend_setup.check_backend',return_value=checked),
              patch('owner_mode.backend_setup._prepare_runtime',return_value={'python':sys.executable,'source':str(self.home/'source')})):
            result=prepare_backend(self.home,self.spec,profile,model_id='fixture')
        argv=result['commands'][0]['argv']
        self.assertNotIn('--draft-model',argv)
        self.assertEqual(argv[argv.index('--draft-algorithm')+1],'NGRAM')
        self.assertEqual(argv.count('--speculative-num-draft-tokens'),1)
        self.assertEqual(argv[argv.index('--speculative-num-draft-tokens')+1],'3')
        self.assertIn('--disable-overlap-schedule',argv)

    def test_wizard_checks_exact_gemma_site_inventory(self):
        kit=KIT_ROOT/'sglang-gemma4'
        metadata=json.loads((kit/'backend.json').read_text())
        profile=json.loads((KIT_ROOT.parent/'profiles/gemma4-26b-a4b-v1/profile.json').read_text())
        sites=profile['sites']
        self.assertEqual(_native_profile_layers(kit,metadata,{},sites),[])
        with self.assertRaisesRegex(SetupError,'exact declared'):
            _native_profile_layers(kit,metadata,{},sites[:-1])
        self.assertEqual(json.loads((kit/'expected-sites.json').read_text()),
                         json.loads((kit/'flashalign_sglang/expected-sites.json').read_text()))
        for model_type in ('gemma4','gemma4_text'):
            self.assertEqual(declared_request_compatibility(metadata,model_type),
                             {'kind':'gemma_boolean_thinking_v1'})


if __name__=='__main__':unittest.main()
