# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Release hashes must cover the exact files in every checksummed engine kit."""
import hashlib
import json
from pathlib import Path
import unittest


class EngineManifestIntegrity(unittest.TestCase):
    def test_packaged_files_match_their_complete_hash_manifest(self):
        engines = Path(__file__).resolve().parents[2]/'engines'
        manifests = sorted([*engines.glob('*/sha256.json'), *engines.glob('*/manifest.json')])
        self.assertTrue(manifests)
        for manifest in manifests:
            with self.subTest(kit=manifest.parent.name):
                value = json.loads(manifest.read_text())
                if manifest.name == 'manifest.json':
                    value = value['files']
                if isinstance(value, list):
                    hashes = {row['path']: row['sha256'] for row in value}
                    self.assertEqual(len(hashes), len(value), 'duplicate manifest paths')
                else:
                    hashes = value
                files = {str(path.relative_to(manifest.parent)): path
                         for path in manifest.parent.rglob('*')
                         if path.is_file() and path != manifest
                         and '__pycache__' not in path.parts and path.suffix != '.pyc'}
                self.assertEqual(set(hashes), set(files), 'manifest does not cover the complete kit')
                for name, path in files.items():
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashes[name], name)


if __name__ == '__main__':
    unittest.main()
