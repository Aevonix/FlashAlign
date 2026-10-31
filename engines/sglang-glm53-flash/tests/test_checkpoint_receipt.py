# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_checkpoint import verify
from launch import check_extra, validate_runtime_versions


class CheckpointReceipt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root/'model'
        self.model.mkdir()
        data = b'bounded synthetic model bytes'
        (self.model/'weights').write_bytes(data)
        self.expected = self.root/'expected.json'
        self.expected.write_text(json.dumps({'stock': {'id': 'fixture', 'revision': 'pin', 'files': {
            'weights': {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}}}}))
        self.receipt = self.root/'state'/'model.json'

    def call(self, **kw):
        return verify(self.model, self.receipt, expected_path=self.expected, **kw)

    def test_full_hash_then_unchanged_receipt_reuse(self):
        with self.assertRaisesRegex(ValueError, 'missing or stale'):
            self.call()
        self.assertEqual(self.call(perform_hashes=True)['new_hashes'], 1)
        self.assertEqual(self.call()['new_hashes'], 0)
        self.assertEqual(self.receipt.stat().st_mode & 0o777, 0o600)

    def test_equal_size_payload_mutation_invalidates(self):
        self.call(perform_hashes=True)
        p = self.model/'weights'
        p.write_bytes(b'x'*p.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'missing or stale'):
            self.call()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.call(perform_hashes=True)

    def test_hf_style_symlink_target_change_invalidates(self):
        payload = self.model/'weights'
        original = self.root/'blob1'
        payload.rename(original)
        payload.symlink_to(original)
        self.call(perform_hashes=True)
        replacement = self.root/'blob2'
        replacement.write_bytes(original.read_bytes())
        payload.unlink()
        payload.symlink_to(replacement)
        with self.assertRaisesRegex(ValueError, 'missing or stale'):
            self.call()

    def test_launch_overrides_are_bounded(self):
        check_extra(['--served-model-name', 'fixture', '--context-length=1024', '--enable-cache-report'])
        for values in (['--speculative-algorithm', 'NONE'], ['--chat-template', 'elsewhere'],
                       ['--hf-config-path', 'elsewhere'], ['--context-length']):
            with self.subTest(values=values), self.assertRaises(ValueError):
                check_extra(values)

    def test_runtime_versions_are_pinned_before_any_gpu_import(self):
        versions = {'sglang':'0.5.20','torch':'2.13.0+cu130','triton':'3.7.1',
                    'safetensors':'0.8.0','transformers':'5.12.1','tilelang':'0.1.12'}
        validate_runtime_versions(versions.__getitem__)
        versions['torch'] = '2.14.1'
        with self.assertRaisesRegex(ValueError,'torch'):
            validate_runtime_versions(versions.__getitem__)


if __name__ == '__main__':
    unittest.main()
