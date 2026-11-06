# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Setup resource selection for source and installed layouts."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from support import load_cli

load_cli('setup')
from owner_mode import resources


class ResourceTests(unittest.TestCase):
    def test_checkout_keeps_its_local_resource_paths(self):
        root = Path(resources.__file__).resolve().parents[1]
        for name in ('engines', 'profiles', 'openclaw-owner-mode'):
            self.assertEqual(resources.setup_resource(name), root / name)

    def test_installed_resources_are_resolved_from_the_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'owner_mode'
            (package / '_assets/engines').mkdir(parents=True)
            # An unrelated sibling directory is not a source checkout.
            (root / 'engines').mkdir()
            with mock.patch.object(resources, '__file__', str(package / 'resources.py')), \
                    mock.patch('importlib.resources.files', return_value=package):
                self.assertEqual(resources.setup_resource('engines'), package / '_assets/engines')
                with self.assertRaisesRegex(ValueError, 'unavailable'):
                    resources.setup_resource('profiles')

    def test_unknown_groups_are_not_paths(self):
        for name in ('..', '/engines', 'engines/../profiles'):
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                resources.setup_resource(name)

    def test_resume_uses_the_current_interpreter_and_wizard(self):
        self.assertEqual(resources.setup_command(),
                         [sys.executable, str(Path(resources.__file__).with_name('setup.py'))])
