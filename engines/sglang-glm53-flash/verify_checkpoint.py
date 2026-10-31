# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Full, resumable verification into owned state; never writes model files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat

ROOT = Path(__file__).resolve().parent


def identity(path):
    entry = Path(path).absolute()
    info = entry.lstat()
    target = entry.resolve(strict=True)
    actual = target.stat()
    if not stat.S_ISREG(actual.st_mode):
        raise ValueError('Checkpoint entry must resolve to a regular file')
    def fields(value):
        return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    return {'entry': str(entry), 'entry_stat': fields(info),
            'link': os.readlink(entry) if stat.S_ISLNK(info.st_mode) else None,
            'target': str(target), 'target_stat': fields(actual)}


def full_hash(path, before):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(before['target'], flags)
    try:
        actual = os.fstat(fd)
        if [actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns, actual.st_ctime_ns] != before['target_stat']:
            raise ValueError('Checkpoint changed before hashing')
        h = hashlib.sha256()
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        after = os.fstat(fd)
        if [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns] != before['target_stat']:
            raise ValueError('Checkpoint changed while hashing')
    finally:
        os.close(fd)
    if identity(path) != before:
        raise ValueError('Checkpoint path changed while hashing')
    return h.hexdigest()


def save(path, value):
    path = Path(path).absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new-' + secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify(model, receipt, role='stock', *, perform_hashes=False, expected_path=ROOT/'checkpoint-hashes.json'):
    model, receipt = Path(model).absolute(), Path(receipt).absolute()
    expected_raw = Path(expected_path).read_bytes()
    expected = json.loads(expected_raw)[role]
    tag = {'model': str(model), 'role': role,
           'expected_sha256': hashlib.sha256(expected_raw).hexdigest()}
    old = json.loads(receipt.read_text()) if receipt.is_file() else {}
    reused = old.get('files', {}) if all(old.get(k) == v for k, v in tag.items()) else {}
    result = {**tag, 'id': expected['id'], 'revision': expected['revision'], 'files': {}, 'complete': False}
    new_hashes = 0
    for name, record in expected['files'].items():
        before = identity(model/name)
        if before['target_stat'][2] != record['bytes']:
            raise ValueError('Wrong pinned checkpoint size: '+name)
        prior = reused.get(name, {})
        if prior.get('identity') == before and prior.get('sha256') == record['sha256']:
            got = record['sha256']
        elif perform_hashes:
            got = full_hash(model/name, before)
            new_hashes += 1
        else:
            raise ValueError('Full verified checkpoint receipt is missing or stale: '+name)
        if got != record['sha256']:
            raise ValueError('Pinned checkpoint payload hash mismatch: '+name)
        result['files'][name] = {'identity': before, 'sha256': got}
        if perform_hashes:
            save(receipt, result)
    result['complete'] = True
    if perform_hashes:
        save(receipt, result)
    result['new_hashes'] = new_hashes
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--role', choices=['stock', 'edited'], default='stock')
    p.add_argument('--verify', action='store_true', help='Read every new or changed payload; schedule outside model timing')
    a = p.parse_args()
    result = verify(a.model, a.receipt, a.role, perform_hashes=a.verify)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))


if __name__ == '__main__':
    main()
