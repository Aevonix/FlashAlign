# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Generate engine copies of the shared authentication wire contract."""
import argparse
from pathlib import Path


def targets(root):
    for source in sorted((root / 'engines').rglob('*.py')):
        if source.name == 'auth_contract.py' or '__pycache__' in source.parts:
            continue
        lines = source.read_text().splitlines()
        if any(line.startswith(('from .auth_contract import ', 'from auth_contract import ')) for line in lines):
            yield source.parent / 'auth_contract.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data = (root / 'hermes-owner-mode/auth_contract.py').read_bytes()
    for target in set(targets(root)):
        if args.check:
            if not target.is_file() or target.read_bytes() != data:
                raise SystemExit('Run python engines/common/sync_auth.py')
        else:
            target.write_bytes(data)


if __name__ == '__main__':
    main()
