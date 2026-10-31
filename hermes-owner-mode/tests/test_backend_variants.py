# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Model-specific setup must reject mismatches and preserve native speculation."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from test_setup import wizard
from owner_mode.backend_setup import KIT_ROOT, _model_matches, check_backend, prepare_backend
from owner_mode.model_metadata import safetensors_header_fingerprint
from owner_mode.profiles import validate_profile
from owner_mode.quickstart import make_default_plan
from owner_mode.setup_runtime import SetupError


class BackendVariants(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.model = self.root/'model'
        self.model.mkdir()
        config = {'model_type': 'qwen4_exp', 'hidden_size': 2, 'quantization_config': {'quant_method': 'fp8'}}
        (self.model/'config.json').write_text(json.dumps(config))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        compatible = {'id': 'fixture-fp8', 'revision': 'a'*40, 'model_type': 'qwen4_exp',
                      'config_sha256': sha(self.model/'config.json'),
                      'index_sha256': sha(self.model/'model.safetensors.index.json'),
                      'quantization_config_sha256': hashlib.sha256(json.dumps(
                          config['quantization_config'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        self.profile = self.root/'profile'
        self.profile.mkdir()
        sites = [
            {'model_role': 'target_and_native_mtp', 'layer': 0, 'site': 'text_embedding_out', 'source_tensors': ['embed']},
            {'model_role': 'target', 'layer': 1, 'site': 'ple_value_out', 'source_tensors': ['ple']},
            {'model_role': 'native_mtp', 'layer': 0, 'site': 'attn_out', 'source_tensors': ['mtp']}]
        profile = {'schema_version': 1, 'name': 'fixture-group', 'model': {'id': 'fixture-base', 'revision': 'pin'},
                   'hidden_size': 2, 'rank': 1, 'scale': 1, 'tensor': 'shared.Q',
                   'transform': 'residual_writer_group_projection_v1',
                   'requires': ['writer_groups', 'text_embedding_before_multimodal_merge',
                                'ple_value_projection', 'native_mtp_writer_groups'],
                   'compatible_models': [compatible], 'sites': sites}
        (self.profile/'profile.json').write_text(json.dumps(profile))
        header = json.dumps({'shared.Q': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [0, 8]}}).encode()
        (self.profile/'profile.safetensors').write_bytes(struct.pack('<Q', len(header))+header+struct.pack('<ff', 1., 0.))
        self.kit = KIT_ROOT/'sglang-qwen4'
        self.metadata = json.loads((self.kit/'backend.json').read_text())
        self.metadata['compatible_models'] = [compatible]
        self.spec = {'kind': 'sglang', 'model_path': str(self.model), 'devices': ['0', '1']}
        self.proc = self.root/'proc'
        self.proc.mkdir()

    def check(self, **changes):
        with mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit, self.metadata, 'sglang')])):
            return check_backend(self.home, {**self.spec, **changes}, validate_profile(self.profile),
                                 model_id='fixture-fp8', proc_root=self.proc)

    def test_fp8_variant_uses_native_mtp_on_two_devices(self):
        result = self.check()
        self.assertEqual(result['drafting'], {'draft_algorithm': 'EAGLE', 'draft_model': str(self.model)})
        self.assertEqual(result['context_window'], {'tokens': 65536, 'source': '--context-length'})
        with self.assertRaisesRegex(SetupError, 'GPU count'):
            self.check(devices=['0'])

    def test_metadata_mismatch_and_external_draft_are_rejected(self):
        with self.assertRaisesRegex(SetupError, 'draft model will not be replaced'):
            self.check(draft_model=str(self.root/'unrelated-draft'))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{"extra":"file"}}')
        with self.assertRaisesRegex(SetupError, 'fingerprints'):
            self.check()

    def test_staging_preserves_nextn_and_custom_depth(self):
        active = {'kind': 'sglang', 'draft_algorithm': 'NEXTN', 'draft_model': str(self.model),
                  'speculative_arguments': ['--speculative-num-steps', '5']}
        with (mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit, self.metadata, 'sglang')])),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=active),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python': sys.executable, 'source': str(self.root/'source')})):
            result = prepare_backend(self.home, self.spec, self.profile, model_id='fixture-fp8')
        argv = result['commands'][0]['argv']
        self.assertEqual(argv[argv.index('--draft-algorithm')+1], 'NEXTN')
        self.assertNotIn('--speculative-algorithm', argv)
        self.assertEqual(argv[-2:], ['--speculative-num-steps', '5'])
        self.assertEqual(argv[argv.index('--context-length')+1], '65536')
        self.assertEqual(argv.count('--context-length'), 1)
        self.assertEqual(result['commands'][0]['env']['CUDA_VISIBLE_DEVICES'], '0,1')

    def test_default_setup_detects_variant_and_native_draft_without_prompts(self):
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=self.profile, engine_bundle=None,
                               backend='auto', profiles=[], engine_python=None, devices='0,1',
                               tools_default=None, engine_only=True)
        kits = lambda: iter([(self.kit, self.metadata, 'sglang')])
        with (mock.patch('owner_mode.backend_setup._kits', side_effect=kits),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.quickstart.download_engine_bundle', side_effect=AssertionError('network')),
              mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')),
              redirect_stdout(io.StringIO())):
            path = make_default_plan(args, {}, wizard.Console(color=False))
        plan = json.loads(path.read_text())
        self.assertEqual(plan['model'], 'fixture-fp8')
        self.assertEqual(plan['engine']['kind'], 'sglang')
        self.assertEqual(plan['engine']['draft_model'], str(self.model))
        self.assertEqual(plan['engine']['draft_algorithm'], 'EAGLE')

    def test_single_shard_fingerprint_reads_header_and_size(self):
        weights = self.model/'model.safetensors'
        header = b'{"__metadata__":{}}'
        weights.write_bytes(struct.pack('<Q', len(header))+header+b'payload')
        expected = {'file': weights.name, 'sha256': hashlib.sha256(header).hexdigest(), 'size': weights.stat().st_size}
        self.assertEqual(safetensors_header_fingerprint(weights), expected)
        compatible = copy.deepcopy(self.metadata['compatible_models'][0])
        del compatible['index_sha256']
        compatible['weights_header'] = expected
        self.assertEqual(len(_model_matches(self.model, {'quant_method': 'fp8'}, [compatible])), 1)
        weights.write_bytes(struct.pack('<Q', 10**9))
        with self.assertRaisesRegex(SetupError, 'header'):
            _model_matches(self.model, {'quant_method': 'fp8'}, [compatible])


if __name__ == '__main__':
    unittest.main()
