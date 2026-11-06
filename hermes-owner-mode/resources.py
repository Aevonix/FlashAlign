# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve setup assets from either a source checkout or an installed wheel."""
from pathlib import Path
import sys


def setup_resource(name):
    if name not in {'engines', 'profiles', 'openclaw-owner-mode'}:
        raise ValueError('Unknown setup resource group')
    root = Path(__file__).resolve().parents[1]
    source = root / name
    if (root / 'packaging/assets.json').is_file() and source.is_dir():
        return source
    # Wheels are installed as ordinary files. Keep the same relative layout so
    # engine scripts and their adjacent license/data files need no special paths.
    from importlib.resources import files
    resource = files('owner_mode').joinpath('_assets', name)
    if not isinstance(resource, Path) or not resource.is_dir():
        raise ValueError('Installed setup resources are unavailable: ' + name)
    return resource


def setup_command():
    return [sys.executable, str(Path(__file__).with_name('setup.py'))]
