# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Include the reviewed setup resources in wheels and source distributions."""
import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent
MANIFEST = 'packaging/assets.json'
ASSETS = json.loads((ROOT / MANIFEST).read_text())['files']
if (not isinstance(ASSETS, list) or any(not isinstance(name, str) for name in ASSETS)
        or len(set(ASSETS)) != len(ASSETS)
        or any(Path(name).is_absolute() or '..' in Path(name).parts for name in ASSETS)):
    raise ValueError('Invalid setup resource manifest')


class BuildWithResources(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / 'owner_mode' / '_assets'
        expected = set(ASSETS)
        existing = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
        if existing - expected:
            raise ValueError('Stale setup resources in the build directory; use a clean build directory')
        for name in ASSETS:
            source = ROOT / name
            if source.is_symlink() or not source.resolve().is_relative_to(ROOT) or not source.is_file():
                raise ValueError('Missing or unsafe setup resource: ' + name)
            target = destination / name
            self.mkpath(str(target.parent))
            force = self.force
            try:
                self.force = True
                self.copy_file(str(source), str(target))
            finally:
                self.force = force

    def get_source_files(self):
        return super().get_source_files() + [MANIFEST, 'tools/package_assets.py',
            'packaging/check_wheel.py', 'packaging/test_build_resources.py',
            'packaging/README.md', *ASSETS]

    def get_outputs(self, include_bytecode=True):
        return super().get_outputs(include_bytecode) + [
            str(Path(self.build_lib) / 'owner_mode' / '_assets' / name) for name in ASSETS]


setup(cmdclass={'build_py': BuildWithResources})
