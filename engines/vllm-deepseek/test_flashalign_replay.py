# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import concurrent.futures
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from flashalign_deepseek.flashalign_replay import ReplayStore, ReplayStoreError, deployment_replay_store


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'nonces.sqlite3'
        self.key = b'k'*32
        self.assertEqual(len(self.key), 32)

    def store(self, **kwargs):
        return ReplayStore(self.path, self.key, **kwargs)

    def test_accept_once_and_after_process_style_restart(self):
        first = self.store()
        self.assertTrue(first.consume('a'*32, 1061, now=1000))
        self.assertFalse(first.consume('a'*32, 1061, now=1001))
        restarted = self.store()
        self.assertFalse(restarted.consume('a'*32, 1061, now=1002))
        self.assertTrue(restarted.consume('b'*32, 1061, now=1002))

    def test_future_timestamp_survives_restart(self):
        # A valid signature timestamp1055 is ahead of both process startups.
        self.assertTrue(self.store().consume('future_nonce_123456', 1116, now=1000))
        self.assertFalse(self.store().consume('future_nonce_123456', 1116, now=1005))

    def test_separate_process_restart_preserves_future_nonce(self):
        code = ('import sys; from flashalign_deepseek.flashalign_replay import ReplayStore; '
                's=ReplayStore(sys.argv[1],b"k"*32); '
                'print(int(s.consume("future_nonce_123456",1116,now=float(sys.argv[2]))))')
        first = subprocess.check_output([sys.executable,'-c',code,str(self.path),'1000'],text=True)
        second = subprocess.check_output([sys.executable,'-c',code,str(self.path),'1005'],text=True)
        self.assertEqual(first.strip(),'1')
        self.assertEqual(second.strip(),'0')

    def test_namespace_and_stored_values_do_not_contain_nonce_or_key(self):
        self.assertTrue(self.store().consume('literal_nonce_123456', 1061, now=1000))
        other = ReplayStore(self.path, b'x'*32)
        self.assertTrue(other.consume('literal_nonce_123456', 1061, now=1000))
        raw = self.path.read_bytes()
        self.assertNotIn(b'literal_nonce_123456', raw)
        self.assertNotIn(self.key, raw)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_atomic_across_independent_connections(self):
        stores = [self.store() for _ in range(8)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda store: store.consume('same_nonce_1234567', 1061, now=1000), stores))
        self.assertEqual(sum(results), 1)

    def test_capacity_is_atomic_and_expiry_reclaims_space(self):
        stores = [self.store(capacity=3) for _ in range(8)]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda pair: pair[1].consume(f'nonce_{pair[0]:016d}', 1001, now=1000), enumerate(stores)))
        self.assertEqual(sum(results), 3)
        self.assertFalse(stores[0].consume('not_yet_room_12345', 1061, now=1001))
        self.assertTrue(stores[0].consume('expired_room_12345', 1061, now=1002))

    def test_unavailable_locked_store_fails_closed(self):
        store = self.store(timeout=0.01)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute('BEGIN IMMEDIATE')
            self.assertFalse(store.consume('blocked_nonce_12345', 1061, now=1000))
        finally:
            connection.rollback()
            connection.close()
        self.assertTrue(store.consume('blocked_nonce_12345', 1061, now=1000))

    def test_replaced_database_fails_closed(self):
        store = self.store()
        self.path.rename(self.path.with_suffix('.old'))
        self.path.write_bytes(b'')
        self.path.chmod(0o600)
        self.assertFalse(store.consume('replacement_123456', 1061, now=1000))

    def test_nonprivate_directory_and_file_are_rejected(self):
        parent = self.path.parent
        parent.chmod(0o755)
        with self.assertRaises(ReplayStoreError):
            self.store()
        parent.chmod(0o700)
        self.path.write_bytes(b'')
        self.path.chmod(0o644)
        with self.assertRaises(ReplayStoreError):
            self.store()

    def test_symlink_and_hardlink_files_are_rejected(self):
        other = self.path.parent/'other'
        other.write_bytes(b'')
        other.chmod(0o600)
        self.path.symlink_to(other)
        with self.assertRaises(ReplayStoreError):
            self.store()
        self.path.unlink()
        os.link(other, self.path)
        with self.assertRaises(ReplayStoreError):
            self.store()

    def test_invalid_expiry_and_nonce_fail_closed(self):
        store = self.store()
        for nonce, expiry in [('short',1061),('bad:nonce_123456789',1061),('a'*32,float('nan')),('a'*32,999),('a'*32,1122)]:
            self.assertFalse(store.consume(nonce, expiry, now=1000))

    def test_corrupt_database_is_rejected(self):
        self.path.write_bytes(b'not a database')
        self.path.chmod(0o600)
        with self.assertRaises(ReplayStoreError):
            self.store()

    def test_helper_explicit_overrides_environment(self):
        destination=self.path.parent/'explicit'/'nonces.sqlite3'
        with patch.dict(os.environ,{'FLASHALIGN_REPLAY_STORE':str(self.path.parent/'unused'/'db')}):
            store=deployment_replay_store(self.key,destination)
        self.assertEqual(store.path,destination)
        self.assertEqual(destination.parent.stat().st_mode & 0o777,0o700)
        self.assertFalse((self.path.parent/'unused').exists())

    def test_helper_environment_and_xdg_state_paths(self):
        destination=self.path.parent/'configured'/'db'
        with patch.dict(os.environ,{'FLASHALIGN_REPLAY_STORE':str(destination)}):
            self.assertEqual(deployment_replay_store(self.key).path,destination)
        with patch.dict(os.environ,{'XDG_STATE_HOME':str(self.path.parent/'xdg'),'FLASHALIGN_REPLAY_STORE':''}):
            store=deployment_replay_store(self.key)
            self.assertEqual(store.path,self.path.parent/'xdg'/'flashalign'/'replay'/'nonces.sqlite3')

    def test_helper_home_fallback_and_existing_permissions_unchanged(self):
        with patch.dict(os.environ,{'XDG_STATE_HOME':'relative-ignored','FLASHALIGN_REPLAY_STORE':''}),patch.object(Path,'home',return_value=self.path.parent):
            store=deployment_replay_store(self.key)
        self.assertEqual(store.path,self.path.parent/'.local'/'state'/'flashalign'/'replay'/'nonces.sqlite3')
        unsafe=self.path.parent/'existing'
        unsafe.mkdir(mode=0o755)
        unsafe.chmod(0o755)
        with self.assertRaises(ReplayStoreError):
            deployment_replay_store(self.key,unsafe/'db')
        self.assertEqual(unsafe.stat().st_mode & 0o777,0o755)


if __name__ == '__main__':
    unittest.main(verbosity=2)
