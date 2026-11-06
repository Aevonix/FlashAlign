# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Run a native engine and its owner-UID profile-control bridge as one supervised unit."""
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


def private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('Native control directories must be owned and private')


def private_socket(path):
    try:
        info = Path(path).lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('Native control socket has an unsafe owner or mode')
    return True


def configuration(path):
    path = Path(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 1024*1024):
        raise ValueError('Native control configuration must be an owned private file')
    value = json.loads(path.read_text())
    expected = {'schema', 'engine_argv', 'control_directory', 'control_socket', 'admin_socket',
                'state_directory', 'staging_directory', 'profile', 'package', 'bridge', 'timeout'}
    if not isinstance(value, dict) or set(value) != expected or value['schema'] != 'flashalign.native-profile-control.v1':
        raise ValueError('Unsupported native control configuration')
    args = value['engine_argv']
    if (not isinstance(args, list) or not args or len(args) > 256
            or any(not isinstance(a, str) or not a or len(a) > 16384 or '\0' in a for a in args)):
        raise ValueError('Invalid native engine arguments')
    if type(value['timeout']) not in (int, float) or not 1 <= value['timeout'] <= 7200:
        raise ValueError('Invalid native control startup timeout')
    for key in expected-{'schema', 'engine_argv', 'timeout'}:
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute() or '\0' in value[key]:
            raise ValueError('Native control paths must be absolute')
    if any(Path(value[key]).parent != Path(value['control_directory']) for key in ('control_socket', 'admin_socket')):
        raise ValueError('Native sockets must use their private control directory')
    return value


def run(value):
    for key in ('control_directory', 'state_directory', 'staging_directory'):
        private_directory(value[key])
    # Only the immutable, setup-staged package is added in isolated Python mode.
    sys.path.insert(0, value['package'])
    from owner_mode.native_socket_forward import prepare as prepare_socket
    for key in ('control_socket', 'admin_socket'):
        prepare_socket(value[key])
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(name, stop)
    engine = subprocess.Popen(value['engine_argv'])
    bridge = None
    try:
        deadline = time.monotonic()+value['timeout']
        while not private_socket(value['control_socket']):
            if stopping or engine.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Native engine control startup did not complete')
            time.sleep(.05)
        bootstrap = 'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name="__main__")'
        bridge = subprocess.Popen([sys.executable, '-I', '-B', '-c', bootstrap, value['package'], value['bridge'],
            '--control-socket', value['control_socket'], '--admin-socket', value['admin_socket'],
            '--state-dir', value['state_directory'], '--profile', value['profile']])
        while not private_socket(value['admin_socket']):
            if stopping or engine.poll() is not None or bridge.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Native profile bridge startup did not complete')
            time.sleep(.05)
        while not stopping:
            if engine.poll() is not None or bridge.poll() is not None:
                raise RuntimeError('Native engine or profile bridge exited')
            time.sleep(.1)
    finally:
        for process in (bridge, engine):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    run(configuration(sys.argv[1]))
