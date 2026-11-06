# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Resolve a trusted engine kit's local profile-control layout during setup."""
import os
from pathlib import Path, PurePosixPath
import re
import stat
import hashlib
import tempfile

from .setup_runtime import SetupError


SCHEMA = 'flashalign.lifecycle-launch.v1'


def _private_directory(path):
    """Create every missing managed parent privately, regardless of umask."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        _private_directory(path.parent)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise SetupError('Profile control directories must be owned with mode 0700')


def prepare_lifecycle(metadata, state, argv):
    spec = metadata.get('profile_lifecycle')
    if spec is None:
        return list(argv), {}
    allowed = {'schema', 'source_argument', 'subdirectory', 'state_argument', 'socket_argument'}
    if not isinstance(spec, dict) or set(spec) - allowed or spec.get('schema') != SCHEMA:
        raise SetupError('Invalid engine profile lifecycle declaration')
    for key in ('source_argument', 'state_argument', 'socket_argument'):
        if key in spec and (not isinstance(spec[key], str) or not re.fullmatch(r'--[a-z][a-z0-9-]*', spec[key])):
            raise SetupError('Invalid lifecycle launcher argument')
    before, separator, after = list(argv), [], []
    if '--' in before:
        index = before.index('--')
        before, separator, after = before[:index], ['--'], before[index+1:]
    base = Path(state)
    if 'source_argument' in spec:
        option = spec['source_argument']
        if before.count(option) != 1 or before.index(option)+1 >= len(before):
            raise SetupError('Lifecycle source directory is absent from the launcher')
        base = Path(before[before.index(option)+1])
    child = spec.get('subdirectory', 'control')
    if (not isinstance(child, str) or not child or PurePosixPath(child).is_absolute()
            or any(part in ('', '..', '.') for part in child.split('/')) or '\\' in child):
        raise SetupError('Invalid lifecycle directory layout')
    source_base = base.absolute()
    base = source_base / child
    address = base / 'admin.sock'
    if len(os.fsencode(address)) > 103:
        if not spec.get('socket_argument'):
            raise SetupError('This engine cannot place its local administration socket at a shorter path')
        identity = hashlib.sha256(os.fsencode(base)).hexdigest()[:20]
        short = Path(tempfile.gettempdir()) / ('flashalign-' + str(os.getuid()) + '-' + identity)
        short.mkdir(mode=0o700, exist_ok=True)
        info = short.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise SetupError('Local administration socket directory is not private')
        address = short / 'admin.sock'
        if len(os.fsencode(address)) > 103:
            raise SetupError('The system temporary path is too long for local administration')
    paths = ([source_base] if 'source_argument' in spec else []) + [base, base / 'incoming']
    for path in paths:
        _private_directory(path)
    for key, value in (('state_argument', base), ('socket_argument', address)):
        option = spec.get(key)
        if option:
            if any(part.split('=', 1)[0] == option for part in argv):
                raise SetupError('Managed profile control arguments cannot be overridden')
            before.extend([option, str(value)])
    return before + separator + after, {'admin_socket': str(address),
        'admin_staging_directory': str(base / 'incoming'),
        'profile_lifecycle': {'schema': SCHEMA, 'status': 'prepared'}}


def check_lifecycle(deployment, *, timeout=5., client_factory=None):
    """Confirm local control readiness without changing any profile selection."""
    if not deployment.get('profile_lifecycle'):
        return {}
    if client_factory is None:
        from .profile_admin_client import AdminClient
        client_factory = AdminClient
    try:
        with client_factory(deployment['admin_socket'], timeout=timeout) as client:
            status = client.request()
        if not isinstance(status, dict) or status.get('admission_fenced') is not False:
            raise ValueError('Profile admission is unavailable')
        row = status['aliases'].get(deployment['profile'])
        generation = row.get('generation') if isinstance(row, dict) else None
        if not generation or not any(item.get('id') == generation and item.get('allocated') is True
                                     for item in status['generations']):
            raise ValueError('The selected profile has no committed generation')
    except Exception as error:
        raise SetupError('The engine profile-control service is not ready') from error
    return {'profile_lifecycle': {'schema': SCHEMA, 'status': 'ready', 'generation': generation}}
