# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from flashalign_deepseek.verify_checkpoint import verify


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'model';self.root.mkdir()
        (self.root/'config.json').write_bytes(b'{"test":1}')
        self.weights=b'synthetic-checkpoint-fixture'
        (self.root/'part.safetensors').write_bytes(self.weights)
        self.pins={'model':'synthetic-model','revision':'test-revision',
                   'metadata_sha256':{'config.json':hashlib.sha256((self.root/'config.json').read_bytes()).hexdigest()},
                   'shards':{'part.safetensors':{'bytes':len(self.weights),'sha256':hashlib.sha256(self.weights).hexdigest()}}}
        self.receipt=Path(self.tmp.name)/'state'/'receipt.json'

    def test_full_hash_then_fast_check_without_writing_weights(self):
        result=verify(self.root,self.receipt,self.pins,fill=True)
        self.assertEqual(result['rehashed_shards'],1)
        self.assertTrue(verify(self.root,self.receipt,self.pins)['verified'])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),['config.json','part.safetensors'])
        self.assertEqual(self.receipt.stat().st_mode & 0o777,0o600)

    def test_changed_same_size_shard_rejects_cached_receipt(self):
        verify(self.root,self.receipt,self.pins,fill=True)
        (self.root/'part.safetensors').write_bytes(b'x'*len(self.weights))
        with self.assertRaises(ValueError):verify(self.root,self.receipt,self.pins)
        with self.assertRaises(ValueError):verify(self.root,self.receipt,self.pins,fill=True)

    def test_resume_preserves_only_finished_shard_hashes(self):
        extra=b'second fixture'
        self.pins['shards']['extra.safetensors']={'bytes':len(extra),'sha256':hashlib.sha256(extra).hexdigest()}
        (self.root/'extra.safetensors').write_bytes(b'x'*len(extra))
        with self.assertRaises(ValueError):verify(self.root,self.receipt,self.pins,fill=True)
        state=json.loads(self.receipt.read_text())
        self.assertFalse(state['complete']);self.assertEqual(len(state['files']),1)
        (self.root/'extra.safetensors').write_bytes(extra)
        self.assertEqual(verify(self.root,self.receipt,self.pins,fill=True)['rehashed_shards'],1)

    def test_model_metadata_and_unsafe_receipt_are_rejected(self):
        verify(self.root,self.receipt,self.pins,fill=True)
        self.receipt.chmod(0o644)
        with self.assertRaises(ValueError):verify(self.root,self.receipt,self.pins)
        self.receipt.chmod(0o600)
        (self.root/'config.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):verify(self.root,self.receipt,self.pins)

    def test_huggingface_snapshot_symlink_and_retarget_detection(self):
        blob=self.root.parent/'blobs'/'hash-a';blob.parent.mkdir();blob.write_bytes(self.weights)
        shard=self.root/'part.safetensors'
        # Preserve the original fixture while modeling a symlinked HF snapshot.
        shard.rename(self.root/'original-fixture')
        shard.symlink_to(Path('..')/'blobs'/'hash-a')
        self.assertTrue(verify(self.root,self.receipt,self.pins,fill=True)['verified'])
        self.assertTrue(verify(self.root,self.receipt,self.pins)['verified'])
        same=self.root.parent/'blobs'/'hash-b';same.write_bytes(self.weights)
        shard.rename(self.root/'prior-link')
        shard.symlink_to(Path('..')/'blobs'/'hash-b')
        with self.assertRaises(ValueError):verify(self.root,self.receipt,self.pins)
        self.assertEqual(verify(self.root,self.receipt,self.pins,fill=True)['rehashed_shards'],1)


if __name__=='__main__':unittest.main(verbosity=2)
