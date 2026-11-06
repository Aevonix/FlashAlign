# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Refresh or check the explicit, tracked setup-resource inventory."""
import argparse
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'packaging/assets.json'


def resource(name, profile_paths, kits):
    path = Path(name)
    declared = (path.parts[0] == 'openclaw-owner-mode'
                or (len(path.parts) > 1 and path.parts[0] == 'engines' and path.parts[1] in kits)
                or name in {'profiles/catalog.json', 'profiles/README.md', 'profiles/SOURCES.md'}
                or any(path.is_relative_to(Path('profiles') / item) for item in profile_paths))
    return (declared and not any(part in {'tests', '__pycache__'} for part in path.parts)
            and not path.name.startswith(('test_', '.'))
            and not path.name.endswith(('-test.mjs', '.pyc')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    catalog = json.loads((ROOT / 'profiles/catalog.json').read_text())
    profile_paths = {item['path'] for item in catalog['models']}
    if any(Path(item).is_absolute() or '..' in Path(item).parts for item in profile_paths):
        raise ValueError('Starter catalog contains an invalid profile path')
    kits = {'common'} | {Path(name).parts[1] for name in tracked
                         if len(Path(name).parts) == 3 and name.startswith('engines/')
                         and name.endswith('/backend.json')}
    value = {'version': 1, 'files': sorted(name for name in tracked
                                         if name and resource(name, profile_paths, kits))}
    if args.check:
        if json.loads(MANIFEST.read_text()) != value:
            raise SystemExit('Setup resources changed; run python tools/package_assets.py')
        print(f"Setup resource inventory matches {len(value['files'])} tracked files.")
    else:
        MANIFEST.parent.mkdir(exist_ok=True)
        MANIFEST.write_text(json.dumps(value, indent=2) + '\n')


if __name__ == '__main__':
    main()
