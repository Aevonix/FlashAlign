# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Local administration selects installed adapters without inventing identities."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ManageChannelsTests(unittest.TestCase):
    def test_selection_unknown_adapter_and_disabled_grant(self):
        script = Path(__file__).resolve().parents[1]/'manage.py'
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)/'approvals.json'
            def run(*args, check=True):
                return subprocess.run([sys.executable, str(script), '--store', str(store), *args],
                                      text=True, capture_output=True, check=check)
            run('approve', 'telegram', '123456789', '--account', 'bot-a')
            listed = json.loads(run('channels').stdout)['adapters']
            self.assertFalse(next(row for row in listed if row['id'] == 'telegram')['enabled'])
            run('channels', '--set', 'whatsapp', 'telegram')
            enabled = json.loads(store.read_text())
            self.assertEqual(enabled['enabled_channels'], ['telegram', 'whatsapp'])
            before = store.read_bytes()
            self.assertNotEqual(run('channels', '--set', 'unknown', check=False).returncode, 0)
            self.assertEqual(store.read_bytes(), before)
            run('channels', '--set', 'whatsapp')
            after = json.loads(store.read_text())
            self.assertNotEqual(after['grants'][0]['id'], enabled['grants'][0]['id'])


if __name__ == '__main__':
    unittest.main()
