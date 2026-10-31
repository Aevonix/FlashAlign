# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Default backend choice follows the selected checkpoint, not only its family."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
from contextlib import redirect_stdout
import io

import test_owner_mode
from owner_mode.backend_setup import _model_matches
from owner_mode.quickstart import backend_matches_checkpoint


class NativeSelection(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config={'model_type':'qwen3_5','quantization_config':{'quant_method':'exl3'}}
        (self.root/'config.json').write_text(json.dumps(self.config))
        (self.root/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        self.pin={'id':'Publisher/Quantized','revision':'a'*40,'model_type':'qwen3_5',
            'config_sha256':hashlib.sha256((self.root/'config.json').read_bytes()).hexdigest(),
            'index_sha256':hashlib.sha256((self.root/'model.safetensors.index.json').read_bytes()).hexdigest()}

    def test_complete_config_and_index_pin_can_omit_redundant_quantization_hash(self):
        self.assertEqual(_model_matches(self.root,self.config['quantization_config'],[self.pin]),[self.pin])
        supplied={**self.pin,'quantization_config_sha256':'0'*64}
        self.assertEqual(_model_matches(self.root,self.config['quantization_config'],[supplied]),[])
        self.config['quantization_config']['quant_method']='different'
        (self.root/'config.json').write_text(json.dumps(self.config))
        self.assertEqual(_model_matches(self.root,self.config['quantization_config'],[self.pin]),[])

    def test_auto_backend_does_not_send_a_quantized_alias_to_a_base_only_path(self):
        self.assertFalse(backend_matches_checkpoint({'qualified_model_ids':['Publisher/Base']},
                         self.pin['id'],self.root,self.config['quantization_config']))
        self.assertFalse(backend_matches_checkpoint({'tested_models':[{'id':'Publisher/Base'}]},
                         self.pin['id'],self.root,self.config['quantization_config']))
        self.assertTrue(backend_matches_checkpoint({'compatible_models':[self.pin]},
                        self.pin['id'],self.root,self.config['quantization_config']))
        self.assertTrue(backend_matches_checkpoint({},self.pin['id'],self.root,self.config['quantization_config']))

    def test_default_plan_selects_native_for_a_pinned_quantized_alias(self):
        from test_setup import make_profile, wizard
        from owner_mode.quickstart import make_default_plan
        profile=make_profile(self.root/'profile',model='Qwen/Qwen3.8-27B')
        document=json.loads((profile/'profile.json').read_text())
        document['compatible_models']=[self.pin]
        (profile/'profile.json').write_text(json.dumps(document))
        bundle=self.root/'bundle';bundle.mkdir()
        (bundle/'manifest.json').write_text('{"default_speculative_method":"mtp"}')
        home=self.root/'home';home.mkdir()
        args=SimpleNamespace(home=home,model_path=self.root,profile=profile,engine_bundle=None,
            backend='auto',profiles=[],engine_python=None,devices='0',tools_default=None,engine_only=True)
        with (mock.patch('owner_mode.quickstart.download_engine_bundle',return_value=bundle) as download,
              mock.patch('owner_mode.backend_setup.active_speculation',return_value=None),
              redirect_stdout(io.StringIO())):
            path=make_default_plan(args,{},wizard.Console(color=False))
        plan=json.loads(path.read_text())
        self.assertEqual(plan['engine']['kind'],'tensorfold')
        self.assertEqual(plan['model'],self.pin['id'])
        download.assert_called_once()


if __name__=='__main__':unittest.main()
