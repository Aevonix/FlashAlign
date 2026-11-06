# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Build-resource regressions without a model, network or full wheel build."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from setuptools import Distribution
from setuptools.command.build_py import build_py


spec = importlib.util.spec_from_file_location('flashalign_resource_build',
    Path(__file__).resolve().parents[1] / 'setup.py')
builder = importlib.util.module_from_spec(spec)
with mock.patch('setuptools.setup'):
    spec.loader.exec_module(builder)


class ResourceBuildTests(unittest.TestCase):
    def test_newer_build_target_is_replaced_and_unlisted_sources_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = 'engines/example/runtime.py'
            source = root / name
            source.parent.mkdir(parents=True)
            source.write_text('current\n')
            (source.parent / 'untracked-private.txt').write_text('not a package input\n')
            build = root / 'build'
            target = build / 'owner_mode/_assets' / name
            target.parent.mkdir(parents=True)
            target.write_text('stale\n')
            newer = source.stat().st_mtime + 60
            os.utime(target, (newer, newer))
            command = builder.BuildWithResources(Distribution())
            command.build_lib = str(build)
            with (mock.patch.object(builder, 'ROOT', root),
                  mock.patch.object(builder, 'ASSETS', [name]),
                  mock.patch.object(build_py, 'run')):
                command.run()
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertFalse((target.parent / 'untracked-private.txt').exists())
            self.assertFalse(command.force)


if __name__ == '__main__':
    unittest.main()
