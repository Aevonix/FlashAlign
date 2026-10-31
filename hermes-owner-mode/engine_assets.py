# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Download pinned runtime-only assets without a GitHub account or profile fetch."""
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .setup_runtime import SetupError


CATALOG = Path(__file__).with_name('engine_catalog.json')
MAX_ARCHIVE = 2 * 1024**3
MAX_EXPANDED = 4 * 1024**3
MAX_FILES = 4096
ASSET_HOSTS = {'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}
RUNTIME_DOCUMENTS = frozenset({
    'LICENSE', 'NOTICE.md', 'THIRD_PARTY_NOTICES.md', 'COMMERCIAL.md', 'CLA.md', 'TRADEMARKS.md',
    'legal/CLA-TERMS.md', 'legal/COMMERCIAL-LICENSE.md', 'legal/COMMERCIAL-ORDER.md', 'legal/LICENSING.md',
})


def _https_url(url, *, initial=False):
    try:
        parts = urlsplit(url)
        port = parts.port
    except (TypeError, ValueError) as error:
        raise SetupError('Invalid runtime asset URL') from error
    if (parts.scheme != 'https' or parts.hostname not in ASSET_HOSTS
            or parts.username or parts.password or port not in (None, 443)
            or parts.fragment):
        raise SetupError('Runtime assets require the supported HTTPS release host')
    if initial and (parts.hostname != 'github.com' or parts.query
                    or re.fullmatch(r'/Aevonix/FlashAlign/releases/download/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.tar\.gz', parts.path) is None):
        raise SetupError('Runtime asset URL is not a pinned FlashAlign release')
    return url


class _ReleaseRedirects(HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(entry):
    request = Request(_https_url(entry['url'], initial=True), headers={
        'User-Agent': 'Aevonix-FlashAlign/1', 'Accept': 'application/octet-stream'})
    try:
        with build_opener(_ReleaseRedirects()).open(request, timeout=45) as response:
            _https_url(response.geturl())
            data = response.read(entry['bytes']+1)
    except SetupError:
        raise
    except (OSError, ValueError) as error:
        raise SetupError('Runtime asset download failed; retry or provide --engine-bundle') from error
    return data


def _entry(model_type, catalog_path, platform_id=None):
    path = Path(catalog_path)
    try:
        raw = path.read_bytes()
        if len(raw) > 128 * 1024:
            raise SetupError('Runtime asset catalog is too large')
        catalog = json.loads(raw)
    except (OSError, ValueError) as error:
        raise SetupError('Runtime asset catalog is unavailable') from error
    if (not isinstance(catalog, dict) or type(catalog.get('version')) is not int or catalog['version'] != 2
            or not isinstance(catalog.get('packages'), list) or len(catalog['packages']) > 64):
        raise SetupError('Invalid runtime asset catalog')
    system = platform_id or platform.system().lower()+'-'+platform.machine().lower()
    matches = []
    for row in catalog['packages']:
        if not isinstance(row, dict):
            raise SetupError('Invalid runtime asset entry')
        models, platforms = row.get('model_types'), row.get('platforms')
        if (not isinstance(models, list) or not all(isinstance(value, str) and value for value in models)
                or not isinstance(platforms, list) or not all(isinstance(value, str) and value for value in platforms)):
            raise SetupError('Runtime asset compatibility must be declared')
        if model_type not in models or system not in platforms:
            continue
        if (not isinstance(row.get('id'), str) or re.fullmatch(r'[A-Za-z0-9._-]{1,128}', row['id']) is None
                or not isinstance(row.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is None
                or type(row.get('bytes')) is not int or not 0 < row['bytes'] <= MAX_ARCHIVE
                or not isinstance(row.get('url'), str)):
            raise SetupError('Invalid pinned runtime asset')
        _https_url(row['url'], initial=True)
        files = row.get('files')
        if not isinstance(files, dict) or not 0 < len(files) <= MAX_FILES:
            raise SetupError('Runtime asset must declare its complete file hashes')
        total = 0
        for name, record in files.items():
            if not isinstance(name, str) or _runtime_path(name) != name or not isinstance(record, dict):
                raise SetupError('Invalid runtime asset file declaration')
            if (not isinstance(record.get('sha256'), str) or re.fullmatch(r'[0-9a-f]{64}', record['sha256']) is None
                    or type(record.get('bytes')) is not int or not 0 <= record['bytes'] <= MAX_EXPANDED):
                raise SetupError('Invalid runtime asset file hash or size')
            total += record['bytes']
        if total > MAX_EXPANDED or not {'manifest.json', 'LICENSE', 'NOTICE.md'} <= set(files):
            raise SetupError('Runtime asset file inventory is incomplete or too large')
        matches.append(row)
    if len(matches) > 1:
        raise SetupError('Runtime asset selection is ambiguous')
    return matches[0] if matches else None


def _runtime_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or any(ord(char) < 32 for char in name)):
        raise SetupError('Unsafe runtime archive path')
    normalized = str(path)
    if (normalized not in {'manifest.json'} | RUNTIME_DOCUMENTS
            and not (len(path.parts) > 1 and path.parts[0] in ('bin', 'lib', 'assets', 'licenses', 'sources'))):
        raise SetupError('Runtime archive contains files outside its package')
    if path.name in ('profile.json', 'profile.safetensors') or 'libcuda.so' in path.name:
        raise SetupError('Runtime assets cannot contain profiles or GPU drivers')
    return normalized


def _extract(data, destination, entry):
    try:
        _extract_archive(data, destination, entry)
    except (tarfile.TarError, EOFError, OSError) as error:
        raise SetupError('Runtime archive could not be read or staged') from error


def _extract_archive(data, destination, entry):
    # Bound metadata and directory records as well as file contents before tar
    # parsing. TarFile otherwise expands hidden PAX records outside our loop.
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
        expanded = compressed.read(MAX_EXPANDED+1)
    if len(expanded) > MAX_EXPANDED:
        raise SetupError('Runtime archive exceeds its expanded size limit')
    try:
        archive = tarfile.open(fileobj=io.BytesIO(expanded), mode='r:')
    except (tarfile.TarError, OSError) as error:
        raise SetupError('Invalid runtime archive') from error
    files = {}
    total = 0
    members = 0
    with archive:
        for member in archive:
            members += 1
            if members > MAX_FILES:
                raise SetupError('Runtime archive exceeds its entry limit')
            if member.isdir():
                if str(PurePosixPath(member.name)) not in ('.', 'bin', 'lib', 'assets', 'licenses', 'sources'):
                    _runtime_path(member.name)
                continue
            if not member.isfile() or member.size < 0:
                raise SetupError('Runtime assets accept regular files only')
            name = _runtime_path(member.name)
            total += member.size
            if name in files or len(files) >= MAX_FILES or total > MAX_EXPANDED:
                raise SetupError('Runtime archive exceeds its limits or repeats a file')
            stream = archive.extractfile(member)
            if stream is None:
                raise SetupError('Runtime archive file is missing')
            payload = stream.read(member.size+1)
            if len(payload) != member.size:
                raise SetupError('Truncated runtime archive')
            parent = destination
            for part in PurePosixPath(name).parts[:-1]:
                parent = parent/part
                parent.mkdir(exist_ok=True, mode=0o700)
            target = destination/name
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as output:
                output.write(payload)
            files[name] = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    if not {'manifest.json', 'LICENSE', 'NOTICE.md'} <= set(files):
        raise SetupError('Runtime archive is missing required package or license files')
    try:
        manifest = json.loads((destination/'manifest.json').read_text())
    except ValueError as error:
        raise SetupError('Invalid runtime package manifest') from error
    models = manifest.get('model_types') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('engine') != 'tensorfold-native' or manifest.get('version') != 2 or manifest.get('runtime_version') != '1.1.9'
            or not isinstance(models, list) or not all(isinstance(value, str) for value in models)
            or set(models) != set(entry['model_types'])):
        raise SetupError('Runtime package does not match its catalog entry')
    from .native_bundle import validate_manifest
    validate_manifest(manifest)
    if files != entry['files']:
        raise SetupError('Runtime package files do not match the pinned catalog')
    receipt = {'version': 2, 'asset': entry['id'], 'sha256': entry['sha256'], 'files': files}
    with (destination/'asset-receipt.json').open('x') as output:
        json.dump(receipt, output, indent=2)
        output.write('\n')
    (destination/'asset-receipt.json').chmod(0o600)


def _cached(path, entry):
    try:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise SetupError('Runtime cache must be an owned, unmodified directory')
        receipt_path = path/'asset-receipt.json'
        info = receipt_path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                or info.st_mode & 0o022 or info.st_size > 128*1024):
            raise SetupError('Invalid runtime cache receipt')
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('version') != 2 or receipt.get('sha256') != entry['sha256']:
            raise SetupError('Runtime cache does not match the selected asset')
        files = receipt.get('files')
        if files != entry['files'] or receipt.get('asset') != entry['id']:
            raise SetupError('Runtime cache inventory differs from the pinned catalog')
        found = set()
        for item in path.rglob('*'):
            info = item.lstat()
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise SetupError('Runtime cache entries must be owned and not writable by others')
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise SetupError('Runtime cache accepts regular files and directories only')
            found.add(str(item.relative_to(path)))
        if found != set(files) | {'asset-receipt.json'}:
            raise SetupError('Runtime cache contains changed files')
        for name, expected in files.items():
            target = path/_runtime_path(name)
            if any(parent.is_symlink() for parent in target.parents if parent != path.parent):
                raise SetupError('Runtime cache contains a symlink')
            info = target.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                    or info.st_size != expected['bytes'] or info.st_size > MAX_EXPANDED
                    or hashlib.sha256(target.read_bytes()).hexdigest() != expected['sha256']):
                raise SetupError('Runtime cache changed; preserve it and provide a verified package')
        return path
    except SetupError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SetupError('Runtime cache is incomplete; preserve it and provide a verified package') from error


def _cache_parent(home):
    path = Path(home).expanduser().resolve()
    for part in ('.local', 'share', 'flashalign', 'engine-bundles'):
        path = path/part
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise SetupError('Runtime cache parent must be an owned directory')
    return path


@contextmanager
def _locked(parent, digest):
    import fcntl
    path = parent/('.'+digest+'.lock')
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise SetupError('Invalid runtime cache lock')
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def download_runtime_bundle(home, model_type, *, catalog_path=None, platform_id=None):
    entry = _entry(model_type, catalog_path or CATALOG, platform_id)
    if entry is None:
        return None
    base = Path(home).expanduser().resolve()/'.local/share/flashalign/engine-bundles'
    destination = base/entry['sha256']
    if destination.exists() or destination.is_symlink():
        _cache_parent(home)
        return _cached(destination, entry)
    data = _fetch(entry)
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise SetupError('Runtime asset checksum or size does not match its pinned catalog')
    parent = _cache_parent(home)
    with _locked(parent, entry['sha256']):
        if destination.exists() or destination.is_symlink():
            return _cached(destination, entry)
        with tempfile.TemporaryDirectory(prefix='.bundle-', dir=parent) as temporary:
            staged = Path(temporary)/'package'
            staged.mkdir(mode=0o700)
            _extract(data, staged, entry)
            if destination.exists() or destination.is_symlink():
                raise SetupError('Runtime cache destination appeared during staging')
            staged.rename(destination)
    return _cached(destination, entry)
