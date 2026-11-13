# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""An installed extension's operating-system error keeps its type and is reported by setup."""
from contextlib import redirect_stdout
import io
import unittest
from unittest import mock

import test_installed_adapters as fixtures
from support import load_cli
from owner_mode import engine_adapters

wizard = load_cli('setup')


class ExtensionOSErrors(unittest.TestCase):
    setUp = fixtures.InstalledAdapters.setUp

    def install(self):
        error = PermissionError('fixture extension cannot read its settings')
        entry = fixtures.entry('unreadable-engine', mock.Mock(side_effect=error))
        self.entries['flashalign.engine_adapters'] = [entry]
        return entry, error

    def test_factory_os_error_keeps_its_type_and_names_the_extension(self):
        entry, error = self.install()
        with self.assertRaises(OSError) as caught:
            engine_adapters.engine_adapters()
        self.assertIn(entry.name, str(caught.exception))
        self.assertIn(entry.dist.name, str(caught.exception))
        self.assertIs(caught.exception.__cause__, error)
        self.assertFalse(engine_adapters._loaded)

    def test_setup_reports_extension_os_error_as_action_needed(self):
        entry, _ = self.install()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(wizard.cli(['--help']), 1)
        self.assertIn('ACTION NEEDED', output.getvalue())
        self.assertIn(entry.name, output.getvalue())
        self.assertIn(entry.dist.name, output.getvalue())
        self.assertIn('fixture extension cannot read its settings', output.getvalue())
