# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""New trusted formats/operators reuse bounded artifact and tensor validation."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile, write_tensors
from owner_mode.profiles import MAX_PROFILE, ProfileError, stage_profile, validate_profile
from owner_mode.profile_adapters import (ArtifactFormatAdapter, DEFAULT_ARTIFACT_FORMATS,
                                         DEFAULT_OPERATORS, OperatorAdapter, OperatorLayout,
                                         TensorRule, using_profile_adapters)


class ProfileAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def binary_profile(self):
        root = self.root/'binary'
        root.mkdir()
        metadata = {'name': 'binary-test', 'model': {'id': 'new-family', 'revision': 'test-revision'},
                    'format': 'example.binary.v1'}
        (root/'profile.json').write_text(json.dumps(metadata))
        (root/'coefficients.bin').write_bytes(struct.pack('<ff', 1.0, 2.0))
        (root/'bias.bin').write_bytes(struct.pack('<f', 0.5))
        return root

    @staticmethod
    def binary_validator(context):
        values = context.read_payload('coefficients.bin', limit=8)
        bias = context.read_payload('bias.bin', limit=4)
        if len(values) != 8 or len(bias) != 4 or not all(math.isfinite(v) for v in struct.unpack('<fff', values+bias)):
            raise ProfileError('Invalid example coefficients')
        return {'name': context.profile['name'], 'model': context.profile['model'],
                'transform': 'example_affine_v1', 'requires': ['example_affine'],
                'artifact_valid': True, 'hashes': {
                    'profile.json': hashlib.sha256(context.manifest_bytes).hexdigest(),
                    'coefficients.bin': hashlib.sha256(values).hexdigest(),
                    'bias.bin': hashlib.sha256(bias).hexdigest()}}

    def test_new_format_can_validate_and_stage_without_central_dispatch_edits(self):
        root = self.binary_profile()
        adapter = ArtifactFormatAdapter('example', 'example.binary.v1', self.binary_validator)
        with self.assertRaisesRegex(ProfileError, 'Unsupported profile format'):
            validate_profile(root)
        registry = DEFAULT_ARTIFACT_FORMATS.with_adapter(adapter)
        with using_profile_adapters(formats=registry):
            result = validate_profile(root)
            self.assertTrue(result['artifact_valid'])
            self.assertEqual(result['format'], 'example.binary.v1')
            self.assertEqual(result['artifact_files'], ['bias.bin', 'coefficients.bin', 'profile.json'])
            stage_profile(root, self.root/'staged')
            self.assertEqual(validate_profile(self.root/'staged')['hashes'], result['hashes'])
        self.assertIsNone(DEFAULT_ARTIFACT_FORMATS.get('example.binary.v1'))

    def test_custom_descriptor_must_bind_read_payload_and_manifest_bytes(self):
        root = self.binary_profile()
        for missing_read in (False, True):
            def bad(context):
                result = self.binary_validator(context)
                if missing_read:
                    result['hashes']['not-read.bin'] = 'a'*64
                else:
                    result['hashes']['coefficients.bin'] = 'b'*64
                return result
            registry = DEFAULT_ARTIFACT_FORMATS.with_adapter(ArtifactFormatAdapter('bad', 'example.binary.v1', bad))
            with self.subTest(missing_read=missing_read), self.assertRaisesRegex(ProfileError, 'validated payload bytes'):
                validate_profile(root, format_registry=registry)

    def test_custom_format_uses_bounded_reads_and_checks_existing_receipts(self):
        root = self.binary_profile()
        for filename, limit in (('../secret', 4), ('coefficients.bin', MAX_PROFILE+1), ('coefficients.bin', 4)):
            def bad(context):
                return context.read_payload(filename, limit=limit)
            registry = DEFAULT_ARTIFACT_FORMATS.with_adapter(ArtifactFormatAdapter('bad', 'example.binary.v1', bad))
            with self.subTest(filename=filename, limit=limit), self.assertRaises(ProfileError):
                validate_profile(root, format_registry=registry)
        (root/'sha256.json').write_text(json.dumps({'coefficients.bin': '0'*64}))
        registry = DEFAULT_ARTIFACT_FORMATS.with_adapter(ArtifactFormatAdapter('good', 'example.binary.v1', self.binary_validator))
        with self.assertRaisesRegex(ProfileError, 'checksum mismatch'):
            validate_profile(root, format_registry=registry)

    def test_new_operator_reuses_finite_positive_and_shape_validation(self):
        root = make_profile(self.root/'operator')
        metadata = json.loads((root/'profile.json').read_text())
        metadata.update(transform='example_vector_v1', requires=['example_vector'])
        (root/'profile.json').write_text(json.dumps(metadata))
        calls = []
        def geometry(context):
            calls.append(context.profile['name'])
            return OperatorLayout({'custom.vector': TensorRule((context.hidden_size,), 'positive')})
        adapter = OperatorAdapter('example_vector_v1', frozenset({'example_vector'}), geometry)
        registry = DEFAULT_OPERATORS.with_adapter(adapter)
        for values, passes in (([1., 2.], True), ([1., -2.], False), ([float('nan'), 2.], False)):
            write_tensors(root, {'custom.vector': ([2], values)})
            if passes:
                result = validate_profile(root, operator_registry=registry, include_audit=True)
                self.assertEqual(result['transform'], adapter.id)
                self.assertEqual(result['audit']['tensors'][0]['name'], 'custom.vector')
            else:
                with self.assertRaises(ProfileError):
                    validate_profile(root, operator_registry=registry)
        self.assertEqual(len(calls), 3)

    def test_new_operator_basis_is_validated_even_without_q_suffix(self):
        root = make_profile(self.root/'basis')
        metadata = json.loads((root/'profile.json').read_text())
        metadata.update(transform='example_basis_v1')
        (root/'profile.json').write_text(json.dumps(metadata))
        registry = DEFAULT_OPERATORS.with_adapter(OperatorAdapter('example_basis_v1', frozenset(),
                      lambda context: OperatorLayout({'directions': TensorRule((1, context.hidden_size), 'basis')})))
        write_tensors(root, {'directions': ([1, 2], [1., 1.])})
        with self.assertRaisesRegex(ProfileError, 'orthonormal'):
            validate_profile(root, operator_registry=registry)
        write_tensors(root, {'directions': ([1, 2], [1., 0.])})
        self.assertTrue(validate_profile(root, operator_registry=registry)['artifact_valid'])

    def test_unknown_operators_and_import_strings_cannot_register_code(self):
        root = make_profile(self.root/'untrusted')
        metadata = json.loads((root/'profile.json').read_text())
        metadata.update(transform='package.module:load', requires=[])
        (root/'profile.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ProfileError, 'Unsupported or inconsistent'):
            validate_profile(root)
        with self.assertRaises(ValueError):
            ArtifactFormatAdapter('bad', 'bad', 'package.module:load')
        with self.assertRaises(ValueError):
            DEFAULT_OPERATORS.with_adapter(DEFAULT_OPERATORS.get('projection'))
        with self.assertRaises(ValueError):
            DEFAULT_ARTIFACT_FORMATS.with_adapter(DEFAULT_ARTIFACT_FORMATS.get(None))


if __name__ == '__main__':
    unittest.main()
