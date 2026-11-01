# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the pinned native-profile engine in an owned Docker container."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import socket
import subprocess
import sys
import time

from engine_config import DEFAULT_ARGUMENTS, DEFAULT_SPECULATION, ENVIRONMENT, IMAGE, MODEL_ID, MODEL_MOUNT
from prepare import docker_image
from supervisor import (available_memory, healthy, native_coverage, owned_container,
                        private_directory, process_identity, resources, run, save, stop_owned)
from flashalign_native.native_profile import NativeProfile
from flashalign_native.protocol import load_key
from flashalign_native.verify_checkpoint import verify

ROOT = Path(__file__).resolve().parent
STOP = False
GIB = 1024**3


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('model', 'profile', 'key-file', 'replay-file', 'state-dir', 'model-receipt'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--host', choices=('127.0.0.1', 'localhost', '0.0.0.0'), default='127.0.0.1')
    p.add_argument('--port', type=int, default=8041)
    p.add_argument('--devices', default='0,1,2,3')
    p.add_argument('--served-model-name', default=MODEL_ID)
    p.add_argument('--speculative-config')
    p.add_argument('--arguments-json', default='[]')
    p.add_argument('--startup-timeout', type=int, default=3600)
    p.add_argument('--allow-untested-memory', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def speculation(value, model):
    config = dict(DEFAULT_SPECULATION) if value is None else json.loads(value)
    if not isinstance(config, dict) or set(config) - {'model'} != set(DEFAULT_SPECULATION):
        raise ValueError('This backend requires the supported adaptive DSpark7 configuration')
    for name, expected in DEFAULT_SPECULATION.items():
        actual = config[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError('Existing speculative settings are incompatible and will not be changed or disabled')
    if 'model' in config:
        if not isinstance(config['model'], str) or Path(config['model']).expanduser().resolve() != model:
            raise ValueError('Native DSpark must use the same pinned stock model')
        config['model'] = MODEL_MOUNT
    return config


def native_arguments(value):
    supplied = json.loads(value)
    if not isinstance(supplied, list) or any(not isinstance(x, str) for x in supplied):
        raise ValueError('Native arguments must be a string array')
    limits = {'--max-model-len': (1, 1048576, int), '--max-num-seqs': (1, 8, int),
              '--max-num-batched-tokens': (1, 4096, int),
              '--max-cudagraph-capture-size': (1, 128, int),
              '--gpu-memory-utilization': (0.01, 0.93, float)}
    overrides = {}
    index = 0
    while index < len(supplied):
        option = supplied[index]
        if '=' in option:
            option, value = option.split('=', 1)
            index += 1
        else:
            if index + 1 >= len(supplied):
                raise ValueError('Missing native argument value')
            value = supplied[index + 1]
            index += 2
        if option not in limits or option in overrides:
            raise ValueError('Unsupported or duplicate native argument')
        lower, upper, convert = limits[option]
        try:
            number = convert(value)
        except ValueError as error:
            raise ValueError('Invalid native argument value') from error
        if not lower <= number <= upper:
            raise ValueError('Native argument exceeds the supported resource range')
        overrides[option] = value
    result = []
    index = 0
    while index < len(DEFAULT_ARGUMENTS):
        option = DEFAULT_ARGUMENTS[index]
        result.append(option)
        if option in limits:
            result.append(overrides.get(option, DEFAULT_ARGUMENTS[index + 1]))
            index += 2
        else:
            index += 1
    return result


def mount(source, target, readonly=False):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator='')
    fields = ['type=bind', 'src=' + str(source), 'dst=' + target]
    if readonly:
        fields.append('readonly')
    writer.writerow(fields)
    return ['--mount', stream.getvalue()]


def container_command(args, run_dir, *, name, labels, model, key, replay, profile, devices):
    command = ['docker', 'create', '--pull=never', '--init', '--restart=no', '--name', name,
               '--user', str(os.getuid()) + ':' + str(os.getgid()),
               '--gpus', json.dumps('device=' + ','.join(devices)),
               '--ipc=private', '--shm-size=32g', '--memory=224g', '--memory-swap=224g',
               '--cap-add=IPC_LOCK', '--ulimit=memlock=-1:-1', '--security-opt=no-new-privileges',
               '--publish', ('127.0.0.1' if args.host == 'localhost' else args.host) + ':' + str(args.port) + ':8000']
    for field, value in sorted(labels.items()):
        command += ['--label', field + '=' + value]
    for source, target, readonly in ((model, MODEL_MOUNT, True),
            (ROOT, '/fa-package', True), (profile, '/fa-profile', True),
            (key, '/fa-private/key', True), (replay.parent, '/fa-replay', False),
            (run_dir.parent.parent / 'cache', '/cache', False),
            (run_dir / 'receipts', '/fa-native-receipts', False),
            (run_dir / 'diagnostics', '/fa-diagnostics', False)):
        command += mount(source, target, readonly)
    environment = {**ENVIRONMENT, 'FLASHALIGN_NATIVE_DIAGNOSTICS': '/fa-diagnostics'}
    # Only explicit engine tuning variables cross the container boundary. Do
    # not copy arbitrary host environment, credentials or provider tokens.
    for field in ('OMP_NUM_THREADS', 'B12X_COMPILE_WORKERS', 'CUDA_DEVICE_MAX_CONNECTIONS',
                  'NCCL_IB_DISABLE', 'NCCL_P2P_LEVEL', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS'):
        if field in os.environ:
            environment[field] = os.environ[field]
    for field, value in sorted(environment.items()):
        command += ['--env', field + '=' + value]
    return command + [IMAGE, '/opt/venv/bin/python', '-m', 'flashalign_native',
        '--profile', '/fa-profile', '--key-file', '/fa-private/key',
        '--replay-file', '/fa-replay/' + replay.name, '--canonical-model', MODEL_ID,
        '--native-receipts', '/fa-native-receipts', '--',
        'serve', MODEL_MOUNT, '--host', '0.0.0.0', '--port', '8000',
        '--served-model-name', args.served_model_name,
        '--speculative-config', json.dumps(speculation(args.speculative_config, model), separators=(',', ':')),
        *native_arguments(args.arguments_json)]


def check_live_worker_pids(record, coverage):
    pids = [row['pid'] for row in coverage['workers']]
    code = '''import json,os,sys
from pathlib import Path
b=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
h=os.sysconf('SC_CLK_TCK');out={}
for p in json.loads(sys.argv[1]):
 f=Path('/proc/'+str(p)+'/stat').read_text().rsplit(')',1)[1].split()
 out[str(p)]={'state':f[0],'started':b+int(f[19])/h}
print(json.dumps(out))
'''
    values = json.loads(run(['docker', 'exec', record['container_id'], '/opt/venv/bin/python',
                             '-c', code, json.dumps(pids)], timeout=20))
    if set(values) != {str(pid) for pid in pids} or any(
            row.get('state') == 'Z' or row.get('started', 0) < record['started_unix'] - 2
            for row in values.values()):
        raise ValueError('Native receipts do not identify the current workers')


def main(argv=None):
    global STOP
    STOP = False
    args = parser().parse_args(argv)
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise ValueError('This native backend supports Linux x86_64 only')
    if not 1 <= args.port <= 65535 or not 60 <= args.startup_timeout <= 7200:
        raise ValueError('Invalid port or startup deadline')
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,199}', args.served_model_name) is None:
        raise ValueError('Invalid served model name')
    state = private_directory(args.state_dir)
    private_directory(state / 'cache')
    profile = NativeProfile(args.profile)
    profile.validate_payload()
    model = args.model.expanduser().resolve(strict=True)
    pins = json.loads((ROOT / 'flashalign_native/model-pins.json').read_text())
    verify(model, args.model_receipt, pins)
    load_key(args.key_file)
    key = args.key_file.expanduser().resolve(strict=True)
    replay = args.replay_file.expanduser().absolute()
    private_directory(replay.parent)
    if replay.is_symlink():
        raise ValueError('Replay database must not be a symlink')
    image = docker_image()
    if image is None:
        raise ValueError('Prepare the pinned native Docker image before activation')
    devices = args.devices.split(',')
    if any(not re.fullmatch(r'[A-Za-z0-9-]+', value) for value in devices):
        raise ValueError('Invalid GPU selector')
    resource = resources(devices, allow_untested_memory=args.allow_untested_memory)
    with socket.socket() as test:
        test.bind(('127.0.0.1' if args.host == 'localhost' else args.host, args.port))
    run_dir = private_directory(state / 'runs' / secrets.token_hex(12))
    private_directory(run_dir / 'receipts'); private_directory(run_dir / 'diagnostics')
    labels = {'io.aevonix.flashalign.deployment': hashlib.sha256(str(state).encode()).hexdigest(),
              'io.aevonix.flashalign.run': run_dir.name}
    name = 'flashalign-native-' + run_dir.name
    command = container_command(args, run_dir, name=name, labels=labels, model=model,
        key=key, replay=replay, profile=profile.directory, devices=devices)
    if args.dry_run:
        print(json.dumps({'image': IMAGE, 'profile': profile.name, 'digest': profile.digest,
                          'devices': devices, 'resources': resource, 'starts_engine': False}))
        return 0
    record = None; guard = None
    def stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        container_id = run(command, timeout=90)
        record = {'container_id': container_id, 'image_id': image['id'], 'labels': labels,
                  'parent_pid': os.getpid(), 'parent_start': process_identity(os.getpid()),
                  'started_unix': time.time(), 'profile_digest': profile.digest}
        save(run_dir / 'container.json', record)
        record_sha = hashlib.sha256((run_dir / 'container.json').read_bytes()).hexdigest()
        guard_log = os.open(run_dir / 'guard.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            guard = subprocess.Popen([sys.executable, str(ROOT / 'guard.py'), '--record',
                str(run_dir / 'container.json'), '--sha256', record_sha],
                stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            os.close(guard_log)
        deadline = time.monotonic() + 10
        while not (run_dir / 'guard-ready.json').is_file():
            if guard.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Container cleanup guard did not start')
            time.sleep(0.1)
        if json.loads((run_dir / 'guard-ready.json').read_text()).get('record_sha256') != record_sha:
            raise RuntimeError('Container cleanup guard identity differs')
        run(['docker', 'start', container_id], timeout=60)
        ready = False; failures = 0; started = time.monotonic()
        save(state / 'current.json', {'run': str(run_dir), **record})
        while not STOP:
            if guard.poll() is not None:
                raise RuntimeError('Container cleanup guard exited')
            current = owned_container(record)
            if current is None or not current['running']:
                raise RuntimeError('Native engine container exited')
            if available_memory() < 24 * GIB:
                raise RuntimeError('Host/cgroup memory fell below the runtime floor')
            try:
                context = healthy(args.host, args.port, args.served_model_name, profile)
                coverage = native_coverage(run_dir / 'receipts', profile,
                                           started_unix=record['started_unix'])
                check_live_worker_pids(record, coverage)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
                failures += 1
                if ready and failures >= 3:
                    raise RuntimeError('Native engine repeatedly failed readiness')
            else:
                failures = 0
                if not ready:
                    data = {'ready': True, 'container_id': container_id, 'profile': profile.name,
                            'profile_digest': profile.digest, 'context_window': context,
                            'native_coverage': coverage}
                    save(run_dir / 'ready.json', data)
                    print(json.dumps(data), flush=True)
                    ready = True
            if not ready and time.monotonic() - started > args.startup_timeout:
                raise TimeoutError('Native engine startup deadline exceeded')
            time.sleep(5)
    finally:
        try:
            if record is not None:
                stop_owned(record)
        finally:
            if guard is not None:
                guard.terminate()
                try:
                    guard.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Container cleanup guard did not finish')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
