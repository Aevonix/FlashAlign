# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Stage and supervise only the native ranks recorded in a deployment receipt."""
import concurrent.futures
import io
import json
import os
from pathlib import Path
import re
import shlex
import struct
import stat
import subprocess
import tarfile
import time

from .setup_runtime import SetupError, check_engine, model_rows


def ssh_options(ssh_config):
    options = []
    if ssh_config is not None:
        config = Path(ssh_config).expanduser().resolve()
        info = config.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise SetupError('SSH config must be owned and not writable by other users')
        options = ['-F', str(config)]
    return options


def rpc(host, operation, *, payload=b'', ssh_config=None, **data):
    if (not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:-]{0,253}', host)
            or host.startswith('-')):
        raise SetupError('Use a hostname, IPv4 address or SSH config alias for each native host')
    helper = Path(__file__).with_name('native_remote.py').read_text()
    command = ['python3', '-c', helper]
    options = ssh_options(ssh_config)
    argv = ([*command] if host == 'local' else
            ['ssh', *options, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ForwardAgent=no', host, shlex.join(command)])
    header = json.dumps({'operation': operation, **data}, separators=(',', ':')).encode()
    try:
        result = subprocess.run(argv, input=struct.pack('>I', len(header))+header+payload,
                                capture_output=True, timeout=180 if operation == 'stage' else 45)
        response = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SetupError(f'Native host {host}: {operation} did not return a valid receipt') from error
    if result.returncode or 'error' in response:
        raise SetupError(f"Native host {host}: {response.get('error', operation+' failed')}")
    return response


def archive(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:') as tar:
        for name, (data, executable) in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o700 if executable else 0o600
            tar.addfile(entry, io.BytesIO(data))
    return output.getvalue()


def stop(deployment):
    from .engine_setup import private_json
    failures = []
    for unit in reversed([*deployment['remote_units'], *deployment.get('controller_units', [])]):
        try:
            state = rpc(unit['host'], 'stop', state=unit['state'], ssh_config=unit.get('ssh_config'))
            if state.get('status') != 'stopped':
                failures.append({'host': unit['host'], 'state': state})
        except SetupError as error:
            failures.append({'host': unit['host'], 'error': str(error)})
    result = {'status': 'incomplete' if failures else 'stopped', 'failures': failures}
    private_json(Path(deployment['state'])/'cluster-running.json', result)
    return result


def activate(deployment, *, timeout=None):
    from .engine_setup import private_json
    state = Path(deployment['state'])
    deadline = time.monotonic()+(deployment.get('startup_timeout', 1800) if timeout is None else timeout)
    units = deployment['remote_units']
    started = []
    try:
        # Followers may wait for the leader's rendezvous. Every receipt is durable before starting the next rank.
        ordered = sorted(units, key=lambda unit: unit['rank'] == 0) + deployment.get('controller_units', [])
        for unit in ordered:
            receipt = rpc(unit['host'], 'start', state=unit['state'], ssh_config=unit.get('ssh_config'))
            started.append({**unit, 'receipt': receipt})
            private_json(state/'cluster-running.json', {'status': 'starting', 'units': started})
        while time.monotonic() < deadline:
            statuses = []
            for unit in ordered:
                status = rpc(unit['host'], 'status', state=unit['state'], ssh_config=unit.get('ssh_config'))
                statuses.append(status)
                if status.get('status') in ('stopped', 'cleanup_incomplete'):
                    raise SetupError('A native rank exited during startup; inspect its private engine.log')
            try:
                models = model_rows(deployment['base_url'], timeout=min(5, max(.1, deadline-time.monotonic())))
            except SetupError:
                time.sleep(.5)
                continue
            if not all(item.get('command_alive') for item in statuses):
                time.sleep(.1)
                continue
            if not any(row['id'] == deployment['model'] for row in models):
                raise SetupError('The native leader serves a different model')
            checked = check_engine(deployment['base_url'], deployment['model'], deployment['profile'],
                                   deployment['key_file'], deadline=deadline)
            result = {**deployment, 'status': 'running', 'engine_check': checked}
            private_json(state/'cluster-running.json', {'status': 'running', 'units': started})
            private_json(state/'deployment.json', result)
            return result
        raise SetupError('Native cluster startup timed out before profile activation was verified')
    except BaseException as error:
        if started:
            result = stop({**deployment, 'remote_units': started, 'controller_units': []})
            if result['status'] != 'stopped':
                raise SetupError('Native cluster startup failed and owned-rank cleanup is incomplete') from error
        raise
