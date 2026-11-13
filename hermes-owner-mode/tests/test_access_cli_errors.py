# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A refused access command from the main CLI ends with one fixed message and a failing exit code."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import test_owner_mode
from owner_mode.contacts import ApprovalStore
from flashalign.__main__ import main

CONTACT = '+10000000000'
REFUSED = {
    'approve on an unknown channel': (('approve', 'unknown', CONTACT), 'Channel has no installed adapter'),
    'approve an empty contact': (('approve', 'rcs', ''), 'Invalid transport identity'),
    'approve a blank contact': (('approve', 'rcs', ' '), 'Invalid transport identity'),
    'select an unknown channel': (('channels', '--set', 'rcs', 'unknown'), 'Enabled channel has no installed adapter'),
}


class AccessCLIErrors(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = ApprovalStore(Path(temporary.name) / 'approvals.json')
        self.store.set_contact('rcs', 'default', CONTACT, True)

    def run_command(self, *argv):
        output, errors = io.StringIO(), io.StringIO()
        package = sys.modules['owner_mode']
        try:
            with redirect_stdout(output), redirect_stderr(errors):
                status = main([*argv, '--store', str(self.store.path)])
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
        return status, output.getvalue(), errors.getvalue()

    def test_refused_command_prints_one_fixed_line_and_fails(self):
        before = self.store.path.read_bytes()
        for label, (argv, reason) in REFUSED.items():
            with self.subTest(command=label):
                status, output, errors = self.run_command(*argv)
                self.assertEqual(status, 1)
                self.assertEqual(errors, 'Access command stopped: ' + reason + '\n')
                self.assertEqual(output, '')
                self.assertEqual(self.store.path.read_bytes(), before)

    def test_accepted_command_still_succeeds_without_a_message(self):
        status, output, errors = self.run_command('channels', '--set', 'rcs')
        self.assertIsNone(status)
        self.assertEqual(errors, '')
        self.assertEqual(self.store.channels(), {'rcs'})
        self.assertIn('"enabled_channels"', output)


if __name__ == '__main__':
    unittest.main()
