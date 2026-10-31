# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Read-only, dependency-free inspection of synthetic profile artifacts."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_setup import wizard, write_tensors
from owner_mode.profile_inspector import inspect_profile, main, render
from owner_mode.profiles import DEEPSEEK_MODEL, validate_profile


def fixture(root, *, transform='projection', checksums=True):
    root.mkdir()
    meta = {'name': 'synthetic', 'model': {'id': 'fixture/model', 'revision': 'pinned'},
            'hidden_size': 4, 'transform': transform, 'requires': [],
            'status': 'DO_NOT_EXPORT private freeform',
            'source': {'private_notes': 'DO_NOT_EXPORT'},
            'sites': [{'layer': 0, 'site': 'attn_out', 'tensor': 'layers.0.output.weight',
                       'rank': 2, 'scale': .75, 'application': 'DO_NOT_EXPORT'}]}
    tensors = {'L0.attn_out.Q': ([2, 4], [1., 0., 0., 0., 0., 1., 0., 0.])}
    if transform == 'row_normalized_projection_v1':
        meta['requires'] = ['row_gain', 'row_norm_preconditioning']
        tensors.update({'L0.attn_out.g': ([4], [1., 2., 1., 2.]),
                        'L0.attn_out.m': ([4], [2., 3., 4., 5.])})
    elif transform == 'input_low_rank_delta_v1':
        meta['requires'] = ['matrix_input']
        meta['sites'][0]['input_size'] = 3
        tensors['L0.attn_out.B'] = ([2, 3], [1., -1., 2., -2., 0., 0.])
    elif transform == 'weighted_output_projection_v1':
        meta['requires'] = ['per_direction_strength']
        meta['sites'][0]['scale'] = 1
        tensors['L0.attn_out.strengths'] = ([2], [.5, 1.])
    (root/'profile.json').write_text(json.dumps(meta))
    write_tensors(root, tensors)
    if checksums:
        seal(root)
    return root


def seal(root):
    (root/'sha256.json').write_text(json.dumps({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                               for name in ('profile.json', 'profile.safetensors')}))


class ProfileInspector(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_standard_summary_is_read_only_and_omits_freeform_metadata(self):
        profile = fixture(self.root/'profile')
        before = {p.name: (p.read_bytes(), p.stat().st_mode) for p in profile.iterdir()}
        with mock.patch('socket.socket', side_effect=AssertionError('No network allowed')):
            result = inspect_profile(profile)
        self.assertEqual(result['schema'], 'flashalign.profile-inspection.v1')
        self.assertEqual((result['site_count'], result['basis_count'], result['basis_rows'], result['maximum_rank']),
                         (1, 1, 2, 2))
        self.assertEqual(result['tensors'][0]['basis'],
                         {'rank': 2, 'row_norm_min': 1., 'row_norm_max': 1., 'max_abs_cross_dot': 0.})
        self.assertEqual(result['assessment']['supplied_checksums'], 'verified')
        self.assertEqual(result['assessment']['model_quality'], 'not_assessed')
        self.assertNotIn('DO_NOT_EXPORT', json.dumps(result))
        self.assertTrue(all('values' not in row for row in result['tensors']))
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mode) for p in profile.iterdir()})
        self.assertNotIn('audit', validate_profile(profile))

    def test_extension_tensor_statistics(self):
        for form in ('row_normalized_projection_v1', 'input_low_rank_delta_v1',
                     'weighted_output_projection_v1'):
            with self.subTest(form=form):
                result = inspect_profile(fixture(self.root/form, transform=form))
                self.assertEqual(result['transform'], form)
                self.assertTrue(any('range' in item for item in result['tensors']))
                self.assertEqual(result['maximum_rank'], 2)
        matrix = inspect_profile(self.root/'input_low_rank_delta_v1')
        self.assertEqual(matrix['sites'][0]['input_size'], 3)

    def test_sign_change_does_not_create_an_anomaly(self):
        profile = fixture(self.root/'profile')
        first = inspect_profile(profile)
        write_tensors(profile, {'L0.attn_out.Q': ([2, 4], [-1., 0., 0., 0., 0., -1., 0., 0.])})
        seal(profile)
        second = inspect_profile(profile)
        self.assertEqual(first['tensors'][0]['basis'], second['tensors'][0]['basis'])
        self.assertNotEqual(first['hashes'], second['hashes'])

    def test_grouped_operator_inventory(self):
        profile = self.root/'grouped'; profile.mkdir()
        meta = {'schema': 'flashalign.grouped_projection.v1', 'name': 'synthetic-grouped',
                'model': DEEPSEEK_MODEL, 'hidden_size': 5120, 'transform': 'projection',
                'requires': ['grouped_basis', 'selected_tp_rows'],
                'operators': {'shared': {'basis': 'basis', 'rank': 2, 'scale': .5}},
                'sites': [{'layer': 1, 'site': 'attn_out', 'tensor': 'layers.1.attn.wo_b.weight',
                           'operator': 'shared'}]}
        (profile/'profile.json').write_text(json.dumps(meta))
        values = [0.] * 10240; values[0] = values[5121] = 1.
        write_tensors(profile, {'basis': ([2, 5120], values)})
        seal(profile)
        result = inspect_profile(profile)
        self.assertEqual(result['operators'], meta['operators'])
        self.assertEqual(result['sites'][0]['operator'], 'shared')
        self.assertIn('rank 2, scale 0.5', render(result, details=True))

    def test_absent_partial_and_invalid_checksums(self):
        profile = fixture(self.root/'profile', checksums=False)
        self.assertEqual(inspect_profile(profile)['assessment']['supplied_checksums'], 'not_supplied')
        (profile/'sha256.json').write_text(json.dumps({'profile.json': hashlib.sha256((profile/'profile.json').read_bytes()).hexdigest()}))
        self.assertEqual(inspect_profile(profile)['assessment']['supplied_checksums'], 'partial')
        (profile/'sha256.json').write_text(json.dumps({'profile.json': '0'*64}))
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['inspect', str(profile), '--json'])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())['artifact_valid'])

    def test_untrusted_terminal_text_is_sanitized(self):
        profile = fixture(self.root/'profile')
        meta = json.loads((profile/'profile.json').read_text())
        meta['model']['id'] = 'fixture\x1b[2J\x1b]8;;https://example.invalid\x07/model\x1b]8;;\x07'
        (profile/'profile.json').write_text(json.dumps(meta)); seal(profile)
        text = render(inspect_profile(profile), details=True)
        self.assertNotIn('\x1b', text)
        self.assertNotIn('\x07', text)

    def test_standalone_script_needs_no_installed_dependencies_or_home(self):
        profile = fixture(self.root/'profile')
        script = Path(__file__).resolve().parents[1]/'profile_inspector.py'
        env = os.environ.copy()
        env.update({'HOME': str(self.root/'empty-home'), 'CUDA_VISIBLE_DEVICES': '',
                    'PYTHONPATH': str(self.root/'missing-package')})
        result = subprocess.run([sys.executable, '-I', '-S', str(script), 'inspect', str(profile), '--json'],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['artifact_valid'])
        self.assertFalse((self.root/'empty-home').exists())

    def test_provisioning_inventory_contains_inspector(self):
        self.assertIn('profile_inspector.py', wizard.MODULES)

    def test_writer_groups_report_one_shared_basis_and_typed_site_fields(self):
        profile = fixture(self.root/'profile')
        meta = json.loads((profile/'profile.json').read_text())
        meta.update({'transform': 'residual_writer_group_projection_v1',
                     'requires': ['writer_groups', 'text_embedding_before_multimodal_merge',
                                  'ple_value_projection', 'native_mtp_writer_groups'],
                     'schema_version': 1, 'rank': 1, 'scale': 1, 'tensor': 'shared.Q',
                     'compatible_models': [{'id': 'fixture/model', 'revision': 'a'*40,
                                            'model_type': 'fixture', 'config_sha256': 'b'*64,
                                            'index_sha256': 'c'*64}],
                     'sites': [
                         {'layer': 0, 'site': 'text_embedding_out',
                          'model_role': 'target_and_native_mtp', 'source_tensors': ['embedding.weight']},
                         {'layer': 0, 'site': 'ple_value_out', 'model_role': 'target',
                          'source_tensors': ['layers.0.ple.weight']},
                         {'layer': 0, 'site': 'attn_out', 'model_role': 'native_mtp',
                          'source_tensors': ['mtp.layers.0.output.weight']},
                         {'layer': 0, 'site': 'combined_mlp_out', 'model_role': 'target',
                          'source_tensors': ['layers.0.experts.weight'], 'expert_count': 2,
                          'fused_expert_tensor_axes': ['expert', 'output', 'input'],
                          'notes': 'DO_NOT_EXPORT'}]})
        (profile/'profile.json').write_text(json.dumps(meta))
        write_tensors(profile, {'shared.Q': ([1, 4], [1., 0., 0., 0.])})
        seal(profile)
        result = inspect_profile(profile)
        self.assertEqual((result['site_count'], result['basis_count'], result['basis_rows']), (4, 1, 1))
        self.assertTrue(all(site['operator'] == 'shared.Q' for site in result['sites']))
        self.assertEqual(result['operators']['shared.Q']['rank'], 1)
        self.assertEqual(result['sites'][-1]['expert_count'], 2)
        self.assertNotIn('DO_NOT_EXPORT', json.dumps(result))
        self.assertIn('rank 1, scale 1', render(result, details=True))

    def test_hashes_identify_the_validated_snapshot(self):
        from owner_mode import profiles
        profile = fixture(self.root/'profile', checksums=False)
        payload = profile/'profile.safetensors'
        original = payload.read_bytes()
        read = profiles._read_bounded

        def mutate_after_read(path, limit):
            value = read(path, limit)
            if Path(path) == payload:
                changed = bytearray(original)
                changed[-32:] = b'\0'*32
                payload.write_bytes(changed)
            return value

        with mock.patch.object(profiles, '_read_bounded', side_effect=mutate_after_read):
            result = inspect_profile(profile)
        self.assertTrue(result['artifact_valid'])
        self.assertEqual(result['hashes']['profile.safetensors'], hashlib.sha256(original).hexdigest())
        self.assertNotEqual(result['hashes']['profile.safetensors'], hashlib.sha256(payload.read_bytes()).hexdigest())
        with self.assertRaisesRegex(profiles.ProfileError, 'orthonormal'):
            inspect_profile(profile)

    def test_deeply_nested_metadata_and_tensor_headers_report_errors(self):
        for component in ('metadata', 'header'):
            with self.subTest(component=component):
                profile = fixture(self.root/component, checksums=False)
                nested = ('['*10000+'0'+']'*10000).encode()
                if component == 'metadata':
                    (profile/'profile.json').write_bytes(nested)
                else:
                    (profile/'profile.safetensors').write_bytes(len(nested).to_bytes(8, 'little')+nested+b'\0'*32)
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(['inspect', str(profile), '--json'])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(output.getvalue())['error_type'], 'invalid_artifact')

    def test_explicit_work_budget_reports_incomplete_not_invalid(self):
        profile = fixture(self.root/'profile')
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['inspect', str(profile), '--max-basis-work', '1', '--json'])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 3)
        self.assertIsNone(result['artifact_valid'])
        self.assertEqual(result['error_type'], 'work_limit')
        self.assertTrue(inspect_profile(profile, max_basis_work=12)['artifact_valid'])

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'FIFO requires POSIX')
    def test_fifo_payload_is_rejected_without_waiting_for_a_writer(self):
        profile = fixture(self.root/'profile', checksums=False)
        payload = profile/'profile.safetensors'
        payload.unlink(); os.mkfifo(payload)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['inspect', str(profile), '--json']), 2)
        self.assertFalse(json.loads(output.getvalue())['artifact_valid'])


if __name__ == '__main__':
    unittest.main()
