# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
import tempfile
from pathlib import Path
import unittest
import subprocess
import sys

import test_owner_mode  # Initializes the local owner_mode package.
from owner_mode import runtime_files as inventory


class RuntimeFilesTests(unittest.TestCase):
    def test_new_installed_module_is_copied_without_a_central_name_list(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('__init__.py', 'new_adapter.py', *inventory.DATA_FILES):
                (root / name).touch()
            (root / 'tests').mkdir()
            (root / 'tests' / 'test_private.py').touch()
            (root / 'key.secret').touch()
            files = inventory.runtime_files(root)
            self.assertIn('new_adapter.py', files)
            self.assertNotIn('key.secret', files)
            self.assertNotIn('test_private.py', files)
            (root / 'new_adapter.py').unlink()
            (root / 'new_adapter.py').symlink_to(root / '__init__.py')
            with self.assertRaises(ValueError):
                inventory.runtime_files(root)

    def test_source_inventory_contains_every_flat_module_and_required_data(self):
        root = Path(__file__).resolve().parents[1]
        files = inventory.runtime_files(root)
        self.assertEqual({p.name for p in root.glob('*.py')},
                         {name for name in files if name.endswith('.py')})
        self.assertTrue(set(inventory.DATA_FILES).issubset(files))

    def test_managed_kit_imports_shared_runtime_without_the_checkout(self):
        from owner_mode.backend_setup import _copy_kit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.mkdir(); (source/'backend.json').write_text('{}')
            target = root/'installed'
            files = _copy_kit(source, target, shared_runtime=True)
            self.assertIn('owner_mode/profile_lifecycle.py', files)
            self.assertIn('owner_mode/profile_adapters.py', files)
            command = [sys.executable, '-I', '-c',
                       'import sys; sys.path.insert(0,sys.argv[1]); '
                       'from owner_mode.profile_lifecycle import LifecycleManager; '
                       'from owner_mode.profiles import validate_profile', str(target)]
            result = subprocess.run(command, capture_output=True, text=True, cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(_copy_kit(source, target, shared_runtime=True), files)
            (target/'owner_mode/profile_lifecycle.py').write_text('changed')
            with self.assertRaises(Exception):
                _copy_kit(source, target, shared_runtime=True)
