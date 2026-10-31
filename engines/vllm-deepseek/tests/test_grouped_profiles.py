# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Synthetic profile contracts and CPU projection checks; no model is loaded."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from flashalign_deepseek.core import Profile, image_project, project, scoped_spans
from flashalign_deepseek.deepseek_grouped import DEEPSEEK_MODEL
from flashalign_vllm.core import Selection


def fixture(path, *, rank=2, full=False):
    rows = [{'layer': 3, 'site': 'attn_out', 'tensor': 'layers.3.attn.wo_b.weight', 'operator': 'shared'}]
    requires = {'grouped_basis', 'selected_tp_rows'}
    if full:
        rows += [
            {'layer': 3, 'site': 'shared_expert_out', 'tensor': 'layers.3.ffn.shared_experts.w2.weight', 'operator': 'shared'},
            {'layer': 3, 'site': 'moe_out', 'tensor_pattern': 'layers.3.ffn.*.w2.weight',
             'components': ['routed_experts', 'shared_expert'],
             'application': 'combined_moe_output_before_mhc_residual', 'operator': 'shared'},
            {'layer': 1, 'site': 'engram_value_out', 'tensor': 'layers.1.engram.wkv.weight',
             'output_slice': [20480, 25600], 'operator': 'shared'},
            {'layer': 0, 'site': 'embedding_out', 'tensor': 'embed.weight', 'operator': 'shared'},
            {'layer': 0, 'site': 'vision_aligner_out', 'tensor': 'aligner.w2.weight',
             'bias_tensor': 'aligner.w2.bias', 'application': 'before_unchanged_bias', 'operator': 'shared'},
            {'layer': 0, 'site': 'image_marker_out', 'tensors': ['image_start', 'image_end', 'image_newline'],
             'application': 'typed_image_span_delimiters_without_aligner_bias', 'operator': 'shared'},
        ]
        requires.update(('shared_expert_output', 'combined_moe_output', 'engram_value_slice',
                         'vision_pre_bias', 'typed_image_span'))
    meta = {'schema': 'flashalign.grouped_projection.v1', 'name': 'synthetic-grouped',
            'model': DEEPSEEK_MODEL, 'hidden_size': 5120, 'transform': 'projection',
            'requires': sorted(requires), 'operators': {'shared': {'basis': 'basis', 'rank': rank, 'scale': .75}},
            'sites': rows}
    basis = torch.zeros(rank, 5120)
    basis[:, :rank] = torch.eye(rank)
    write(path, meta, {'basis': basis})
    return meta


def write(path, meta, tensors=None):
    path.mkdir(exist_ok=True)
    (path/'profile.json').write_text(json.dumps(meta))
    if tensors is not None:
        save_file(tensors, path/'profile.safetensors')
    (path/'sha256.json').write_text(json.dumps({name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                                               for name in ('profile.json', 'profile.safetensors')}))


class GroupedProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = fixture(self.root, full=True)

    def changed(self):
        write(self.root, self.meta)
        return Profile(self.root)

    def test_rank_two_partial_sites_and_missing_site_identity(self):
        self.meta = fixture(self.root)
        profile = self.changed()
        self.assertEqual(set(profile.sites), {(3, 'attn_out')})
        value = torch.tensor([[float('nan'), -0.] + [0.]*5118])
        selection = Selection(profile.name, profile.digest, 1.)
        with scoped_spans([(0, 1, selection)]):
            self.assertIs(project(value, {profile.name: profile}, 2, 'attn_out'), value)

    def test_rank_128_remains_supported(self):
        fixture(self.root, rank=128)
        profile = Profile(self.root)
        self.assertEqual(tuple(profile.operators['shared'][0].shape), (128, 5120))

    def test_starter_profile_loads_without_rewriting(self):
        path = Path(__file__).resolve().parents[3]/'profiles/deepseek41-flash1017-v1'
        before = {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                  for name in ('profile.json', 'profile.safetensors', 'sha256.json')}
        profile = Profile(path)
        self.assertEqual(len(profile.sites), 85)
        self.assertEqual({q.shape[0] for q, _ in profile.operators.values()}, {1})
        self.assertEqual(before, {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                                  for name in before})

    def test_rank_two_mixed_rows_and_strides(self):
        profile = self.changed()
        selection = Selection(profile.name, profile.digest, .5)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                value = torch.zeros(3, 10240, dtype=dtype)[:, ::2]
                value[0, :2] = torch.tensor([float('nan'), -0.], dtype=dtype)
                value[1, :3] = torch.tensor([8., 4., 3.], dtype=dtype)
                original = value.clone()
                with scoped_spans([(1, 2, selection)]):
                    actual = project(value, {profile.name: profile}, 3, 'attn_out')
                torch.testing.assert_close(actual[1, :3], torch.tensor([5., 2.5, 3.], dtype=dtype))
                self.assertTrue(torch.equal(actual[[0, 2]].contiguous().view(torch.uint8),
                                            original[[0, 2]].contiguous().view(torch.uint8)))
                self.assertTrue(torch.equal(value.contiguous().view(torch.uint8),
                                            original.contiguous().view(torch.uint8)))

    def test_rank_two_image_bias_and_marker_contract(self):
        profile = self.changed()
        selection = Selection(profile.name, profile.digest, .5)
        value = torch.zeros(3, 5120)
        value[:, :2] = torch.tensor([8., 4.])
        bias = torch.zeros(5120)
        bias[:2] = torch.tensor([2., 2.])
        actual = image_project(value, torch.tensor([0, 1, 3]), profile, selection, bias)
        torch.testing.assert_close(actual[[0, 2], :2], torch.tensor([[5., 2.5], [5., 2.5]]))
        torch.testing.assert_close(actual[1, :2], torch.tensor([5.75, 3.25]))

    def test_wrong_native_targets_fail_in_direct_engine_loader(self):
        mutations = {
            'attention': (0, 'tensor', 'layers.3.attn.wo_a.weight'),
            'shared': (1, 'tensor', 'layers.3.ffn.shared_experts.w1.weight'),
            'moe': (2, 'application', 'after_residual'),
            'engram': (3, 'output_slice', [0, 5120]),
            'embedding': (4, 'tensor', 'unknown.weight'),
            'vision': (5, 'application', 'after_bias'),
            'marker': (6, 'tensors', ['image_start']),
        }
        for name, (index, field, wrong) in mutations.items():
            with self.subTest(site=name):
                fixture(self.root, full=True)
                self.meta = json.loads((self.root/'profile.json').read_text())
                self.meta['sites'][index][field] = wrong
                with self.assertRaisesRegex(ValueError, 'native output'):
                    self.changed()

    def test_unused_operator_and_bad_requirements_fail(self):
        self.meta['operators']['unused'] = dict(self.meta['operators']['shared'])
        with self.assertRaisesRegex(ValueError, 'unused'):
            self.changed()
        self.meta['operators'].pop('unused')
        self.meta['requires'].append(self.meta['requires'][0])
        with self.assertRaisesRegex(ValueError, 'requirements'):
            self.changed()

    def test_duplicate_site_and_boolean_rank_fail(self):
        self.meta['sites'].append(dict(self.meta['sites'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.changed()
        self.meta['sites'].pop()
        self.meta['operators']['shared']['rank'] = True
        with self.assertRaisesRegex(ValueError, 'rank'):
            self.changed()


if __name__ == '__main__':
    unittest.main()
