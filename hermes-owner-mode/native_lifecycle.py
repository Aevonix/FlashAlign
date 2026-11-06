# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a declared native profile-control service without model-specific dispatch."""
import hashlib
import json
import os
from pathlib import Path

from .runtime_files import runtime_files
from .setup_runtime import SetupError

SCHEMA = 'flashalign.native-profile-control.v1'
PROTOCOLS = frozenset(('flashalign.tensorfold-control.v1', 'flashalign.tensorfold-control.v2'))


def declaration(manifest):
    value = manifest.get('profile_control')
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'schema', 'protocol', 'bridge'}
            or value.get('schema') != SCHEMA or not isinstance(value.get('protocol'), str)
            or value['protocol'] not in PROTOCOLS):
        raise SetupError('Unsupported native profile-control declaration')
    bridge = value.get('bridge')
    record = manifest.get('files', {}).get(bridge) if isinstance(bridge, str) else None
    if (not isinstance(record, dict) or not bridge.startswith('bin/') or not bridge.endswith('.py')
            or manifest.get('runtime_version') != '1.2.0'
            or manifest.get('rank_transport') not in ('cli', 'dsv41_env')):
        raise SetupError('Native profile control requires a matching inventoried bridge and current rank transport')
    return dict(value)


def addresses(rank_state):
    rank_state = Path(rank_state)
    suffix = hashlib.sha256(os.fsencode(rank_state)).hexdigest()[:24]
    control = Path('/tmp') / ('flashalign-native-' + suffix)
    return {'directory': control, 'native': control/'native.sock', 'admin': control/'admin.sock',
            'state': rank_state/'profile-control', 'incoming': rank_state/'profile-control/incoming'}


def runner_files():
    """Snapshot the installed helper payload before deriving a deployment identity."""
    source = Path(__file__).parent
    files = {'native_control_runner.py': ((source/'native_control_runner.py').read_bytes(), False)}
    for name in runtime_files(source):
        files['control-python/owner_mode/'+name] = ((source/name).read_bytes(), False)
    return files


def prepare_runner(manifest, rank_state, engine_argv, profile, *, python, timeout, helpers=None):
    """Return rank-zero supervisor arguments and a bounded, standard-library payload."""
    spec = declaration(manifest)
    if spec is None:
        raise SetupError('This native bundle does not declare profile control')
    root = Path(rank_state)
    paths = addresses(root)
    package = root/'control-python'
    files = dict(runner_files() if helpers is None else helpers)
    config = {'schema': SCHEMA, 'engine_argv': list(engine_argv), 'control_directory': str(paths['directory']),
              'control_socket': str(paths['native']), 'admin_socket': str(paths['admin']),
              'state_directory': str(paths['state']), 'staging_directory': str(paths['incoming']),
              'profile': str(profile), 'package': str(package),
              'bridge': str(root/'runtime'/spec['bridge']), 'timeout': timeout}
    files['native-control.json'] = ((json.dumps(config, sort_keys=True)+'\n').encode(), False)
    return [python, '-I', '-B', str(root/'native_control_runner.py'), str(root/'native-control.json')], files, paths
