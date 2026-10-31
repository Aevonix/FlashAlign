# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Command-line test imports preserve normal package lookup across Python versions."""
import importlib
import sys
import unittest
from unittest import mock

from support import load_cli
from test_setup import wizard


class CliLoading(unittest.TestCase):
    def test_repeated_cli_imports_preserve_package_and_dotted_patch_targets(self):
        package = sys.modules['owner_mode']
        backend = importlib.import_module('owner_mode.backend_setup')
        terminal = importlib.import_module('owner_mode.terminal_ui')
        for name in ('manage', 'setup', 'manage', 'setup'):
            with self.subTest(cli=name):
                load_cli(name)
                self.assertIs(sys.modules['owner_mode'], package)
                self.assertIs(package.backend_setup, backend)
                self.assertIs(package.terminal_ui, terminal)
                with mock.patch('owner_mode.backend_setup._select_kit', return_value='test-kit'):
                    self.assertEqual(backend._select_kit(), 'test-kit')
                with mock.patch('owner_mode.terminal_ui._logo', return_value='test-logo'):
                    self.assertEqual(terminal._logo(), 'test-logo')


if __name__ == '__main__':
    unittest.main()
