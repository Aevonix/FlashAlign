# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Native OpenClaw owner ingress requires isolated direct-message routing."""
import json
from pathlib import Path
import tempfile
import unittest

from test_setup import wizard
from owner_mode.setup_runtime import SetupError, openclaw_session_scope


class OpenClawSessionSetup(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.config = self.home/'.openclaw/openclaw.json'
        self.config.parent.mkdir()
        self.key = self.home/'engine.key'
        self.key.write_bytes(b'x'*32)
        self.key.chmod(0o600)

    def provision(self):
        return wizard.provision(self.home, ['openclaw'], 'http://127.0.0.1:1234/v1',
                                'fixture', 'fixture-profile', [], False, str(self.key))

    def test_default_and_existing_isolated_scope_plans_are_read_only(self):
        self.assertEqual(openclaw_session_scope({})['selected'], 'per-account-channel-peer')
        for scope in ['per-account-channel-peer', 'per-channel-peer']:
            config = {'session': {'dmScope': scope}}
            self.assertFalse(openclaw_session_scope(config)['change_required'])
            self.assertEqual(config, {'session': {'dmScope': scope}})

    def test_shared_scope_is_backed_up_and_isolated_without_losing_settings(self):
        original = {'session': {'dmScope': 'main', 'idleMinutes': 100}}
        self.config.write_text(json.dumps(original))
        result = self.provision()
        current = json.loads(self.config.read_text())
        self.assertEqual(current['session'], {'dmScope': 'per-account-channel-peer', 'idleMinutes': 100})
        recovery = Path(result['recovery_directory'])
        journal = json.loads((recovery/'transaction.json').read_text())
        self.assertEqual(journal['status'], 'committed')
        backup = next(entry for entry in journal['entries']
                      if entry['path'] == '.openclaw/openclaw.json')
        self.assertEqual(json.loads((recovery/backup['backup']).read_text()), original)
        self.assertTrue(result['openclaw_session_scope']['change_required'])

    def test_existing_channel_scope_is_preserved(self):
        self.config.write_text('{"session":{"dmScope":"per-channel-peer"}}')
        result = self.provision()
        self.assertEqual(json.loads(self.config.read_text())['session']['dmScope'], 'per-channel-peer')
        self.assertFalse(result['openclaw_session_scope']['change_required'])

    def test_bad_scope_is_rejected_before_approvals_or_config_change(self):
        for value in [None, [], {'dmScope': 'unknown'}]:
            raw = json.dumps({'session': value})
            self.config.write_text(raw)
            with self.assertRaises(SetupError):
                self.provision()
            self.assertEqual(self.config.read_text(), raw)
            self.assertFalse((self.home/'.config/owner-mode/approvals.json').exists())

    def test_discovery_reports_bad_scope_without_blocking_other_harnesses(self):
        self.config.write_text('{"session":{"dmScope":"unknown"}}')
        result = wizard.discover(self.home, which=lambda _: None)
        self.assertIn('session_scope_error', result['openclaw'])
        self.assertIn('hermes', result)


if __name__ == '__main__':
    unittest.main()
