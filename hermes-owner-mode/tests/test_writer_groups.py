# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Writer-group files require complete typed declarations and model fingerprints."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from test_setup import wizard
from owner_mode.profiles import ProfileError, copy_profile_notices, match_profile, validate_profile


class WriterGroups(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = {'schema_version': 1, 'name': 'group-fixture', 'model': {'id': 'base', 'revision': 'base-pin'},
            'hidden_size': 2, 'transform': 'residual_writer_group_projection_v1',
            'requires': ['writer_groups', 'text_embedding_before_multimodal_merge',
                         'ple_value_projection', 'native_mtp_writer_groups'],
            'rank': 1, 'scale': 1, 'tensor': 'shared.Q',
            'sites': [
                {'layer': 0, 'site': 'text_embedding_out', 'model_role': 'target_and_native_mtp', 'source_tensors': ['embed.weight']},
                {'layer': 1, 'site': 'ple_value_out', 'model_role': 'target', 'source_tensors': ['ple.value.weight']},
                {'layer': 0, 'site': 'attn_out', 'model_role': 'target', 'source_tensors': ['target.attn.weight']},
                {'layer': 0, 'site': 'attn_out', 'model_role': 'native_mtp', 'source_tensors': ['mtp.attn.weight']}],
            'compatible_models': [{'id': 'quantized', 'revision': 'a'*40, 'model_type': 'fixture',
                'config_sha256': 'b'*64, 'index_sha256': 'c'*64, 'quantization_config_sha256': 'd'*64}]}
        header = json.dumps({'shared.Q': {'dtype': 'F32', 'shape': [1, 2], 'data_offsets': [0, 8]}}).encode()
        (self.root/'profile.safetensors').write_bytes(struct.pack('<Q', len(header))+header+struct.pack('<ff', 1., 0.))

    def save(self):
        (self.root/'profile.json').write_text(json.dumps(self.profile))
        return validate_profile(self.root)

    def test_shared_basis_and_separate_native_mtp_role_validate(self):
        result = self.save()
        self.assertTrue(result['artifact_valid'])
        self.assertEqual(result['transform'], 'residual_writer_group_projection_v1')

    def test_duplicate_group_or_missing_feature_coverage_rejects(self):
        self.profile['sites'].append(dict(self.profile['sites'][0]))
        with self.assertRaisesRegex(ProfileError, 'Duplicate'):
            self.save()
        self.profile['sites'] = self.profile['sites'][2:4]
        with self.assertRaisesRegex(ProfileError, 'feature coverage'):
            self.save()

    def test_unknown_requirement_is_not_ignored(self):
        self.profile['requires'].append('unimplemented_feature')
        with self.assertRaisesRegex(ProfileError, 'Unsupported'):
            self.save()

    def test_compatible_revision_and_fingerprint_validation(self):
        value = self.save()
        self.assertTrue(match_profile(value, {'id': 'quantized', 'revision': 'a'*40})[0])
        self.assertFalse(match_profile(value, {'id': 'quantized', 'revision': 'e'*40})[0])
        self.profile['compatible_models'][0]['config_sha256'] = 7
        with self.assertRaisesRegex(ProfileError, 'fingerprints'):
            self.save()

    def test_boolean_geometry_rejected(self):
        self.profile['rank'] = True
        with self.assertRaisesRegex(ProfileError, 'geometry'):
            self.save()

    def test_unknown_or_boolean_schema_rejected(self):
        for value in (2, True, None):
            self.profile['schema_version'] = value
            with self.assertRaisesRegex(ProfileError, 'schema version'):
                self.save()

    def test_single_shard_layout_fingerprint_validates(self):
        model = self.profile['compatible_models'][0]
        del model['index_sha256']
        model['weights_header'] = {'file': 'model.safetensors', 'sha256': 'e'*64, 'size': 200}
        self.assertTrue(self.save()['artifact_valid'])
        for value in ('../model.safetensors', '/tmp/model.safetensors'):
            model['weights_header']['file'] = value
            with self.assertRaisesRegex(ProfileError, 'weight fingerprints'):
                self.save()

    def test_source_license_survives_staging(self):
        source = b'Source license\r\n'
        (self.root/'LICENSE.source').write_bytes(source)
        target = self.root/'staged'
        target.mkdir()
        copy_profile_notices(self.root, target)
        self.assertEqual((target/'LICENSE.source').read_bytes(), source)


if __name__ == '__main__':
    unittest.main()
