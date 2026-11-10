# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The main CLI uses the existing local access operations."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import test_owner_mode
from owner_mode.contacts import ApprovalStore
from flashalign.__main__ import main


class AccessCLI(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.store = ApprovalStore(self.home / '.config/owner-mode/approvals.json')

    def run_command(self, *argv):
        output = io.StringIO()
        package = sys.modules['owner_mode']
        try:
            with mock.patch.object(Path, 'home', return_value=self.home), redirect_stdout(output):
                main(list(argv))
        finally:
            # The first access command imports manage.py, whose standalone bootstrap replaces the
            # package object. Keep the suite's package and its bindings, as support.load_cli does.
            sys.modules['owner_mode'] = package
            for fullname, child in list(sys.modules.items()):
                if fullname.startswith('owner_mode.'):
                    parent_name, _, attribute = fullname.rpartition('.')
                    parent = sys.modules.get(parent_name)
                    if parent is not None and child is not None:
                        setattr(parent, attribute, child)
        return json.loads(output.getvalue())

    def test_approve_uses_default_store_and_exact_account(self):
        self.run_command('approve', 'rcs', '+10000000000', '--account', 'account-a')
        self.assertTrue(self.store.approved('rcs', 'account-a', '+10000000000'))
        self.assertIsNone(self.store.approved('rcs', 'default', '+10000000000'))

    def test_revoke_uses_explicit_store(self):
        store = ApprovalStore(self.home / 'custom.json')
        store.set_contact('rcs', 'default', '+10000000000', True)
        self.run_command('revoke', 'rcs', '+10000000000', '--store', str(store.path))
        self.assertIsNone(store.approved('rcs', 'default', '+10000000000'))
        self.assertFalse(self.store.path.exists())

    def test_mode_preserves_contact_choice(self):
        self.store.set_contact('rcs', 'default', '+10000000000', True)
        grants = self.store.read()['grants']
        self.run_command('mode', 'always_on')
        self.assertTrue(self.store.always_on())
        self.run_command('mode', 'verified')
        self.assertFalse(self.store.always_on())
        self.assertEqual(self.store.read()['grants'], grants)

    def test_channels_list_and_replace(self):
        self.assertIn('adapters', self.run_command('channels'))
        self.run_command('channels', '--set', 'rcs', 'telegram')
        self.assertEqual(self.store.channels(), {'rcs', 'telegram'})
        self.run_command('channels', '--set')
        self.assertFalse(self.store.channels())

    def test_tool_uses_existing_override(self):
        self.run_command('tool', 'sessions_spawn', 'off')
        self.assertFalse(self.store.tool_enabled('sessions_spawn'))
        self.run_command('tool', 'sessions_spawn', 'on')
        self.assertTrue(self.store.tool_enabled('sessions_spawn'))
