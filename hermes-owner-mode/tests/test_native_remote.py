# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Real rank staging and owned supervisor cleanup without a GPU or model."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
from types import SimpleNamespace

from test_setup import wizard
from owner_mode import native_remote
from owner_mode.native_cluster import archive


class NativeRemote(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        supervisor = Path(native_remote.__file__).with_name('engine_supervisor.py').read_bytes()
        self.files = {'engine_supervisor.py': (supervisor, False),
                      'engine.key': (bytes(range(32)), False),
                      'unit.json': ((json.dumps({'argv': [sys.executable, '-c', 'import time;time.sleep(60)'], 'env': {}})+'\n').encode(), False)}
        self.request = {'parent': str(self.root), 'id': 'a'*20,
                        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'executable': exe}
                                  for name, (data, exe) in self.files.items()}}

    def test_repeatable_private_staging_and_tamper_detection(self):
        payload = archive(self.files)
        first = native_remote.stage(self.request, payload)
        second = native_remote.stage(self.request, payload)
        self.assertFalse(first['reused'])
        self.assertTrue(second['reused'])
        path = Path(first['state'])
        self.assertEqual((path/'engine.key').stat().st_mode & 0o777, 0o600)
        (path/'unit.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            native_remote.stage(self.request, payload)
        with self.assertRaisesRegex(ValueError, 'changed'):
            native_remote.start({'state': str(path)})

    def test_supervised_start_and_stop_reap_only_the_recorded_process(self):
        result = native_remote.stage(self.request, archive(self.files))
        self.addCleanup(native_remote.stop, {'state': result['state']})
        started = native_remote.start({'state': result['state']})
        self.assertEqual(started['status'], 'starting')
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            status = native_remote.state_of(result['state'])
            if status.get('command_alive'):
                break
            time.sleep(.05)
        self.assertTrue(status['command_alive'])
        self.assertEqual(native_remote.stop({'state': result['state']})['status'], 'stopped')
        self.assertTrue(native_remote.state_of(result['state'])['all_children_reaped'])

    def test_replay_parent_is_prepared_and_persistent_journal_survives_stage_and_stop(self):
        replay = self.root/'persistent'/'native-replay'/('b'*64)
        request = {**self.request, 'owner_replay_dir': str(replay)}
        result = native_remote.stage(request, archive(self.files))
        self.assertEqual(replay.parent.stat().st_mode & 0o777, 0o700)
        self.assertFalse(replay.exists())
        replay.mkdir(mode=0o700)
        (replay/'state.bin').write_bytes(b'durable replay evidence')
        native_remote.stage(request, archive(self.files))
        native_remote.stop({'state': result['state']})
        self.assertEqual((replay/'state.bin').read_bytes(), b'durable replay evidence')

    def test_replay_path_rejects_links_and_nonprivate_parents(self):
        parent = self.root/'public'
        parent.mkdir(mode=0o755)
        with self.assertRaisesRegex(ValueError, 'private'):
            native_remote.prepare_replay_directory(str(parent/'journal'))
        private = self.root/'private'
        private.mkdir(mode=0o700)
        target = self.root/'target'
        target.mkdir(mode=0o700)
        (private/'linked').symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'link'):
            native_remote.prepare_replay_directory(str(private/'linked'))

    def test_payload_cannot_escape_owned_rank(self):
        wrong = {**self.files, '../outside': (b'bad', False)}
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            native_remote.stage(self.request, archive(wrong))
        self.assertFalse((self.root.parent/'outside').exists())

    def test_unified_memory_gpu_uses_process_occupancy_when_memory_is_na(self):
        query = SimpleNamespace(returncode=0, stdout='0, GPU-fixture, [N/A], [N/A], 12.1\n')
        idle = SimpleNamespace(returncode=0, stdout='')
        with mock.patch.object(native_remote.subprocess, 'run', side_effect=[query, idle]):
            devices, architectures = native_remote.cuda_inventory()
        self.assertEqual(devices, ['0'])
        self.assertEqual(architectures, {'0': 'sm_121'})
        busy = SimpleNamespace(returncode=0, stdout='GPU-fixture, 123\n')
        with mock.patch.object(native_remote.subprocess, 'run', side_effect=[query, busy]):
            self.assertEqual(native_remote.cuda_inventory()[0], [])


if __name__ == '__main__':
    unittest.main()
