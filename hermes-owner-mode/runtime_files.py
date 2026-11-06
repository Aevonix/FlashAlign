# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Inventory the installed flat Python package and its declared runtime data."""
from pathlib import Path


DATA_FILES = (
    'LICENSE', 'NOTICE.md', 'plugin.yaml', 'engine_catalog.json',
    'terminal-logo.json', 'requirements.txt',
)
OPTIONAL_FILES = ('README.md',)


def runtime_files(root):
    """New installed modules accompany harness copies without a second module list."""
    root = Path(root)
    modules = sorted(path.name for path in root.glob('*.py'))
    if '__init__.py' not in modules:
        raise ValueError('FlashAlign runtime package is incomplete')
    names = [*modules, *DATA_FILES,
             *(name for name in OPTIONAL_FILES if (root / name).exists())]
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('FlashAlign runtime package contains an unsafe or missing file')
    return tuple(sorted(names))
