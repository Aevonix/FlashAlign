# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The qualified Qwen SGLang context reaches discovery and staged commands."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile
from owner_mode.backend_setup import check_backend, prepare_backend
from owner_mode.profiles import validate_profile


class SGLangContext(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.model, self.draft = self.root/'model', self.root/'draft'
        self.model.mkdir()
        self.draft.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type':'qwen3_5','hidden_size':2}))
        (self.draft/'config.json').write_text(json.dumps({'architectures':['DFlash2DraftModel']}))
        self.profile = make_profile(self.root/'profile')
        profile = json.loads((self.profile/'profile.json').read_text())
        profile['sites'][0]['tensor'] = 'model.language_model.layers.0.self_attn.o_proj.weight'
        (self.profile/'profile.json').write_text(json.dumps(profile))
        self.spec = {'kind':'sglang','model_path':str(self.model),'draft_model':str(self.draft),'devices':['0']}

    def test_default_65k_context_preserves_qualified_dflash_settings(self):
        with (mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python':sys.executable,'source':str(self.root/'source')})):
            checked = check_backend(self.home, self.spec, validate_profile(self.profile), model_id='bench')
            deployed = prepare_backend(self.home, self.spec, self.profile, model_id='bench')
        self.assertEqual(checked['context_window'], {'tokens':65536,'source':'--context-length'})
        argv = deployed['commands'][0]['argv']
        self.assertEqual(argv[argv.index('--context-length')+1], '65536')
        self.assertEqual(argv.count('--context-length'), 1)
        self.assertEqual(argv[argv.index('--mem-fraction-static')+1], '0.82')
        self.assertEqual(argv[argv.index('--max-running-requests')+1], '2')
        self.assertEqual(argv[argv.index('--draft-algorithm')+1], 'DFLASH')
        self.assertEqual(argv[-2:], ['--speculative-num-draft-tokens','8'])
        self.assertFalse((Path(deployed['state'])/'running.json').exists())

    def test_explicit_context_and_active_draft_depth_are_preserved(self):
        spec = {**self.spec, 'arguments':['--context-length','32768']}
        active = {'kind':'sglang','draft_model':str(self.draft),'draft_algorithm':'DFLASH',
                  'speculative_arguments':['--speculative-num-draft-tokens','12']}
        with (mock.patch('owner_mode.backend_setup.active_speculation', return_value=active),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python':sys.executable,'source':str(self.root/'source')})):
            checked = check_backend(self.home, spec, validate_profile(self.profile), model_id='bench')
            deployed = prepare_backend(self.home, spec, self.profile, model_id='bench')
        self.assertEqual(checked['context_window']['tokens'], 32768)
        self.assertEqual(deployed['commands'][0]['argv'][-4:],
                         ['--context-length','32768','--speculative-num-draft-tokens','12'])


if __name__ == '__main__':
    unittest.main()
