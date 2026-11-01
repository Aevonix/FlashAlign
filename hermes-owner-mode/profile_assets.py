# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fetch a declared public native payload into an external, verified cache."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request

from .profiles import (ProfileError, NATIVE_FORMAT, MAX_HEADER, read_json,
                       validate_profile, copy_profile_notices)

MAX_DOWNLOAD = 2 * 1024**3
CHUNK = 8 * 1024 * 1024
REDIRECT_HOSTS = {'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}


def _public_url(value, *, redirect=False):
    if not isinstance(value, str) or len(value) > 16384:
        raise ProfileError('Invalid public profile asset URL')
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ProfileError('Invalid public profile asset URL') from error
    if (parsed.scheme != 'https' or parsed.hostname not in REDIRECT_HOSTS
            or parsed.username is not None or parsed.password is not None
            or port not in (None, 443) or parsed.fragment):
        raise ProfileError('Profile assets require a public GitHub HTTPS release URL')
    if not redirect and (parsed.hostname != 'github.com' or parsed.query
            or re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/releases/download/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', parsed.path) is None):
        raise ProfileError('Profile assets require a pinned GitHub release URL')
    return parsed


def download_record(profile):
    """Read local download metadata only; this function never makes a request."""
    root = Path(profile['path'])
    descriptor = root / 'payload-download.json'
    if descriptor.is_symlink():
        raise ProfileError('Profile download metadata must not be a symlink')
    value = read_json(descriptor)
    if (not isinstance(value, dict) or set(value) != {'version', 'file', 'url', 'bytes', 'sha256', 'terms'}
            or type(value['version']) is not int or value['version'] != 1
            or value['file'] != profile.get('payload_file')
            or type(value['bytes']) is not int or not 0 < value['bytes'] <= MAX_DOWNLOAD
            or value['bytes'] != profile.get('payload_bytes')
            or value['sha256'] != profile['hashes'].get(value['file'])):
        raise ProfileError('Profile download metadata differs from the native manifest')
    parsed = _public_url(value['url'])
    if parsed.path.rsplit('/', 1)[-1] != value['file']:
        raise ProfileError('Profile download filename differs from its release URL')
    terms = value['terms']
    if (not isinstance(terms, str) or not terms or Path(terms).name != terms
            or terms in ('.', '..') or '\\' in terms):
        raise ProfileError('Profile source terms must be a local notice file')
    path = root / terms
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_HEADER:
        raise ProfileError('The declared profile source terms are missing')
    checksums = root / 'sha256.json'
    if checksums.is_file():
        recorded = read_json(checksums)
        if not isinstance(recorded, dict):
            raise ProfileError('Invalid checksum record')
        for name in ('payload-download.json', terms):
            wanted = recorded.get(name)
            if wanted is not None and hashlib.sha256((root/name).read_bytes()).hexdigest() != wanted:
                raise ProfileError('Profile download metadata or source terms changed')
    return {**value, 'terms_path': str(path)}


class _ReleaseRedirects(HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, request, fp, code, message, headers, new_url):
        _public_url(new_url, redirect=True)
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _download(record, destination):
    opener = build_opener(_ReleaseRedirects())
    request = Request(record['url'], headers={'User-Agent': 'FlashAlign-profile-setup/1'})
    digest = hashlib.sha256()
    size = 0
    deadline = time.monotonic() + 1800
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        with opener.open(request, timeout=30) as response:
            _public_url(response.geturl(), redirect=True)
            if response.status != 200:
                raise ProfileError('Public profile asset did not return HTTP 200')
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdecimal() or int(length) != record['bytes']):
                raise ProfileError('Public profile asset length differs from its declaration')
            while True:
                if time.monotonic() > deadline:
                    raise ProfileError('Public profile asset download timed out')
                block = response.read(CHUNK)
                if not block:
                    break
                size += len(block)
                if size > record['bytes']:
                    raise ProfileError('Public profile asset exceeds its declared size')
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    if size != record['bytes'] or digest.hexdigest() != record['sha256']:
        raise ProfileError('Public profile asset checksum or size differs')


def _owned_directory(path):
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022):
        raise ProfileError('The profile cache must be an owned directory not writable by other users')


def materialize_profile(path, *, cache_root=None):
    """Return a fully verified artifact; download only a missing declared payload."""
    checked = validate_profile(path, allow_missing_payload=True)
    if checked.get('artifact_valid') is True:
        return checked
    if checked.get('format') != NATIVE_FORMAT or checked.get('payload_status') != 'missing':
        raise ProfileError('The selected profile is incomplete')
    record = download_record(checked)
    if cache_root is None:
        cache_root = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache'))) / 'flashalign/profiles'
    cache_root = Path(cache_root).expanduser()
    _owned_directory(cache_root)
    source = Path(checked['path'])
    identity = hashlib.sha256(json.dumps({**checked['hashes'],
        'payload-download.json': hashlib.sha256((source/'payload-download.json').read_bytes()).hexdigest(),
        record['terms']: hashlib.sha256((source/record['terms']).read_bytes()).hexdigest()}, sort_keys=True).encode()).hexdigest()
    destination = cache_root / identity
    if os.path.lexists(destination):
        if destination.is_symlink():
            raise ProfileError('Cached profile must not be a symlink')
        existing = validate_profile(destination)
        if existing['hashes'] != checked['hashes']:
            raise ProfileError('Cached profile differs from its declaration; use a fresh cache directory')
        return existing
    temporary = Path(tempfile.mkdtemp(prefix='profile-', dir=cache_root))
    try:
        for name in ('profile.json', 'sha256.json', 'payload-download.json'):
            original = source / name
            if original.is_file() and not original.is_symlink():
                shutil.copyfile(original, temporary/name)
                (temporary/name).chmod(0o600)
        copy_profile_notices(source, temporary)
        # Surface the source notice through discovery/setup; preserve it with
        # the cached profile even if a future notice filename differs.
        if not (temporary/record['terms']).exists():
            shutil.copyfile(source/record['terms'], temporary/record['terms'])
            (temporary/record['terms']).chmod(0o600)
        _download(record, temporary/record['file'])
        ready = validate_profile(temporary)
        if ready['hashes'] != checked['hashes']:
            raise ProfileError('Downloaded profile differs from its declaration')
        os.rename(temporary, destination)
        return validate_profile(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
