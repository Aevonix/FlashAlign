# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Prepare and run an isolated TensorFold deployment from a checked engine bundle."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlsplit

from .auth import read_key, ensure_key
from .engine_supervisor import descendants as _descendants
from .drafting import draft_settings, argument
from .profiles import read_json, validate_profile, stage_profile, _compatible_models, ProfileError
from .setup_runtime import SetupError, check_engine, model_rows, remaining_startup_time
from .model_metadata import (declared_openclaw_compat, declared_request_compatibility,
                             declared_openclaw_request_policy)

_children = {}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def check_profile_checkpoint(model_path, profile):
    """Match one complete declared checkpoint variant, never a mixture of pins."""
    try:
        alternatives = _compatible_models(profile.get('compatible_models', []))
    except ProfileError as error:
        raise SetupError('Invalid compatible checkpoint metadata') from error
    fields = ('config_sha256', 'index_sha256', 'quantization_config_sha256')
    candidates = []
    for target in [profile['model'], *alternatives]:
        pins = {field: target[field] for field in fields if target.get(field) is not None}
        for expected in pins.values():
            if (not isinstance(expected, str) or len(expected) != 64
                    or any(char not in '0123456789abcdef' for char in expected)):
                raise SetupError('Invalid checkpoint fingerprint in the selected profile')
        if pins:
            candidates.append(pins)
    if not candidates:
        return
    required = set().union(*(pins.keys() for pins in candidates))
    actual = {}
    for field, name in (('config_sha256', 'config.json'),
                        ('index_sha256', 'model.safetensors.index.json')):
        path = Path(model_path)/name
        if field in required:
            actual[field] = digest(path) if path.is_file() else None
    if 'quantization_config_sha256' in required:
        config = read_json(Path(model_path)/'config.json')
        text = config.get('text_config') or {}
        quantization = config.get('quantization_config') or text.get('quantization_config')
        actual['quantization_config_sha256'] = hashlib.sha256(json.dumps(
            quantization, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if not any(all(actual.get(field) == expected for field, expected in pins.items())
               for pins in candidates):
        raise SetupError('Checkpoint metadata differs from the selected profile')


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name+'.new-'+secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    os.replace(temporary, path)


def inside(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or not value.parts:
        raise SetupError('Engine bundle contains an unsafe path')
    result = Path(root).joinpath(*value.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise SetupError('Engine bundle path escapes its directory')
    return result





def check_bundle(bundle, model, profile):
    from .native_bundle import check_bundle as check_native_bundle
    return check_native_bundle(bundle, model, profile)



def declared_input_modalities(manifest):
    """Use an explicit serving-lane declaration, never the checkpoint architecture."""
    if 'input_modalities' not in manifest:
        return None
    value = manifest['input_modalities']
    if (not isinstance(value, list) or not value or 'text' not in value
            or any(not isinstance(item, str) or item not in ('text', 'image') for item in value)
            or len(set(value)) != len(value)):
        raise SetupError('Engine bundle input_modalities must uniquely declare text and optional image')
    return value.copy()


def tensorfold_arguments(spec, manifest):
    """Allow only an explicit context override beyond the checked package flags."""
    arguments = manifest.get('arguments', [])
    if not isinstance(arguments, list) or any(not isinstance(item, str) for item in arguments):
        raise SetupError('Engine bundle arguments must be an argument array')
    extra = spec.get('arguments', [])
    if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
        raise SetupError('TensorFold engine.arguments must be an argument array')
    if not extra:
        return arguments.copy()
    if spec.get('command'):
        raise SetupError('TensorFold engine.arguments cannot be combined with a custom command')
    if len(extra) == 2 and extra[0] == '--context':
        value = extra[1]
    elif len(extra) == 1 and extra[0].startswith('--context='):
        value = extra[0].split('=', 1)[1]
    else:
        raise SetupError('TensorFold engine.arguments supports only one --context override')
    if not value.isascii() or not value.isdecimal() or len(value) > 10 or not 0 < int(value) <= 2**31-1:
        raise SetupError('TensorFold --context must be a positive token count')
    # Replace a package default, instead of relying on duplicate-flag parsing.
    result = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == '--context':
            if index + 1 >= len(arguments):
                raise SetupError('Engine bundle --context is missing its value')
            index += 2
        elif item.startswith('--context='):
            index += 1
        else:
            result.append(item)
            index += 1
    return result + ['--context', str(int(value))]





def unused_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def prepare_python_headers(python, state, *, install=False, environment=None):
    """Find matching Python headers or unpack an APT development package locally."""
    environment = environment or {}
    try:
        result = subprocess.run([str(python), '-c',
            'import json,sys,sysconfig; print(json.dumps({"version":list(sys.version_info[:2]),'
            '"include":sysconfig.get_path("include"),"platinclude":sysconfig.get_path("platinclude"),'
            '"cc":sysconfig.get_config_var("CC"),"cflags":sysconfig.get_config_var("CFLAGS")}))'],
            capture_output=True, text=True, check=True, timeout=15)
        info = json.loads(result.stdout)
        major, minor = info['version']
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SetupError('Could not inspect the engine Python build headers') from error
    include = Path(info['include'])
    includes = list(dict.fromkeys([include, Path(info.get('platinclude') or include)]))
    configured = environment.get('CPATH', os.environ.get('CPATH', ''))
    candidates = [*includes, *(Path(value) for value in configured.split(os.pathsep) if value)]
    def validate(cpath):
        effective = {**os.environ, **environment, 'CPATH': cpath}
        try:
            compiler = shlex.split(effective.get('CC', info.get('cc') or 'cc'))
            flags = shlex.split(info.get('cflags') or '')
            for name in ('CFLAGS', 'CPPFLAGS'):
                flags.extend(shlex.split(effective.get(name, '')))
            if not compiler:
                raise OSError('No C compiler selected')
            command = [*compiler, *flags, *(flag for path in includes for flag in ('-I', str(path))),
                       '-x', 'c', '-fsyntax-only', '-']
            source = (f'#include <Python.h>\n#if PY_MAJOR_VERSION != {major} || PY_MINOR_VERSION != {minor}\n'
                      '#error Python header version differs from the engine interpreter\n#endif\n')
            subprocess.run(command, input=source, env=effective,
                           capture_output=True, text=True, check=True, timeout=15)
        except (OSError, ValueError) as error:
            raise SetupError('Could not run the C compiler to validate Python development headers; '
                             'install a C compiler or set engine.environment.CC and valid compiler flags') from error
        except subprocess.SubprocessError as error:
            raise SetupError(f'Python {major}.{minor} development headers cannot be compiled; '
                             'check engine.environment.CPATH (including multiarch pyconfig.h) and compiler flags') from error
    def matches(path):
        if not (path/'Python.h').is_file() or not (path/'patchlevel.h').is_file():
            return False
        import re
        text = (path/'patchlevel.h').read_text()
        return (re.search(rf'^#define\s+PY_MAJOR_VERSION\s+{major}\s*$', text, re.M) is not None
                and re.search(rf'^#define\s+PY_MINOR_VERSION\s+{minor}\s*$', text, re.M) is not None)
    if any(matches(path) for path in candidates):
        validate(configured)
        return {'CPATH': configured} if configured else {}
    base = Path(state)/'python-headers'
    unpacked = base/'unpacked'
    selected = unpacked/f'usr/include/python{major}.{minor}'
    receipt = base/'receipt.json'
    if receipt.is_file() and matches(selected):
        recorded = read_json(receipt)
        if any(not (unpacked/name).is_file() or digest(unpacked/name) != value
               for name, value in recorded['files'].items()):
            raise SetupError('Managed Python headers changed; use a fresh deployment')
    else:
        if not install or not shutil.which('apt-get') or not shutil.which('dpkg-deb'):
            raise SetupError(f'Python {major}.{minor} development headers are missing; install matching headers or set engine.environment.CPATH')
        packages = base/'packages'
        packages.mkdir(parents=True, exist_ok=True, mode=0o700)
        log = base/'prepare.log'
        package_name = f'libpython{major}.{minor}-dev'
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'ab') as stream:
            try:
                subprocess.run(['apt-get', 'download', package_name], cwd=packages,
                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
                archives = sorted(packages.glob(package_name+'_*.deb'))
                if len(archives) != 1:
                    raise SetupError('Python header package selection is ambiguous; use a fresh deployment')
                subprocess.run(['dpkg-deb', '--extract', str(archives[0]), str(unpacked)],
                    stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=60)
            except (OSError, subprocess.SubprocessError) as error:
                raise SetupError(f'Could not prepare Python {major}.{minor} headers; install them or set engine.environment.CPATH') from error
        if not matches(selected):
            raise SetupError('Downloaded Python headers do not match the engine Python version')
        files = {str(path.relative_to(unpacked)): digest(path) for path in unpacked.rglob('*') if path.is_file()}
        private_json(receipt, {'package': archives[0].name, 'sha256': digest(archives[0]), 'files': files})
    paths = [str(selected), str(unpacked/'usr/include')]
    if configured:
        paths.append(configured)
    cpath = os.pathsep.join(paths)
    validate(cpath)
    return {'CPATH': cpath}


def available_devices():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible and visible != '-1':
        return [item.strip() for item in visible.split(',') if item.strip()]
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10, check=True)
        return [row[0].strip() for line in result.stdout.splitlines()
                if len(row := line.split(',')) == 3 and float(row[1]) < .05 * float(row[2])]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def process_identity(pid, *, include_zombies=False):
    try:
        root = Path('/proc')/str(pid)
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z' and not include_zombies:
            return None
        return {'pid': pid, 'start': fields[19], 'uid': root.stat().st_uid}
    except (OSError, IndexError, ValueError):
        return None


def _group_members(group):
    """Capture live process identities, never commands or environments."""
    result = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(path.name)
            identity = process_identity(pid)
            if identity and os.getpgid(pid) == group and process_identity(pid) == identity:
                result.append(identity)
        except (OSError, ValueError):
            continue
    return result


def _signal_identity(record, signum):
    """A pidfd keeps a validated process from becoming a reused PID target."""
    if record.get('uid') != os.getuid() or process_identity(record.get('pid')) != record:
        return False
    if not callable(getattr(os, 'pidfd_open', None)) or not callable(getattr(signal, 'pidfd_send_signal', None)):
        return False
    try:
        fd = os.pidfd_open(record['pid'])
        try:
            if process_identity(record['pid']) != record:
                return False
            signal.pidfd_send_signal(fd, signum)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def _require_pidfds():
    if not callable(getattr(os, 'pidfd_open', None)) or not callable(getattr(signal, 'pidfd_send_signal', None)):
        raise SetupError('Managed engine cleanup requires Linux pidfd support; no process was started')
    try:
        fd = os.pidfd_open(os.getpid())
        os.close(fd)
    except OSError as error:
        raise SetupError('Managed engine cleanup requires working Linux pidfd support; no process was started') from error


def _owned_group_leader(record):
    if not isinstance(record, dict) or type(record.get('pid')) is not int:
        return False
    try:
        return (process_identity(record['pid']) == record and record['uid'] == os.getuid()
                and os.getpgid(record['pid']) == record['pid'])
    except ProcessLookupError:
        return False


def _capture_owned_members(records, previous=()):
    """Follow verified ancestry as well as legacy groups, retaining birth IDs."""
    members = {tuple(record.get(key) for key in ('pid', 'start', 'uid')): record
               for record in previous if isinstance(record, dict)}
    roots = [record for record in previous if isinstance(record, dict)
             and process_identity(record.get('pid')) == record and record.get('uid') == os.getuid()]
    for record in records:
        if _owned_group_leader(record):
            roots.append(record)
            for member in _group_members(record['pid']):
                if member['uid'] == os.getuid():
                    members[tuple(member[key] for key in ('pid', 'start', 'uid'))] = member
    for member in _descendants(roots):
        members[tuple(member[key] for key in ('pid', 'start', 'uid'))] = member
    return list(members.values())


def _supervisor_state(state, descriptor):
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get('process'),dict): return None
    name = descriptor.get('receipt')
    if not isinstance(name, str) or Path(name).name != name or not name.startswith('supervisor-'):
        return None
    try:
        path = state/name
        if path.is_symlink() or path.stat().st_uid != os.getuid(): return None
        value = read_json(path)
        if not isinstance(value,dict) or value.get('supervisor')!=descriptor['process']:return None
        command=value.get('command')
        if command is not None and (not isinstance(command,dict) or set(command)!={'pid','start','uid'}):return None
        return value
    except (OSError, ValueError, TypeError):
        return None


def _supervisors_running(state, supervisors):
    if not isinstance(supervisors,list):return False
    for descriptor in supervisors:
        value = _supervisor_state(state, descriptor)
        if (not value or value.get('status') != 'running'
                or process_identity((value.get('command') or {}).get('pid')) != value.get('command')):
            return False
    return True


def stop_engine(deployment):
    """Stop verified descendants; keep subreapers alive until children are reaped."""
    if deployment.get("remote_units"):
        from .native_cluster import stop
        return stop(deployment)
    state = Path(deployment['state']); marker = state/'running.json'
    if not marker.exists(): return {'status': 'stopped'}
    previous = read_json(marker); records = previous.get('processes', [])
    supervisors = previous.get('supervisors', [])
    if (not isinstance(supervisors,list) or any(not isinstance(row,dict) or not isinstance(row.get('process'),dict) for row in supervisors)):
        return {'status':'incomplete','reason':'Invalid supervisor ownership receipt','running_receipt':str(marker)}
    protected = {tuple(row.get('process', {}).get(k) for k in ('pid','start','uid')) for row in supervisors}
    members = previous.get('owned_members', [])
    def capture():
        nonlocal members
        updated = _capture_owned_members(records, members)
        if updated != members:
            members = updated
            private_json(marker, {**previous, 'owned_members': members})
        return [row for row in members if process_identity(row.get('pid')) == row]
    # Every ownership snapshot is durable before any process can be signalled.
    members = _capture_owned_members(records, members)
    private_json(marker, {**previous, 'owned_members': members})
    for signum in (signal.SIGTERM, signal.SIGKILL):
        grace = 35 if deployment.get('container_runtime') == 'docker' and signum == signal.SIGTERM else 5
        sent = set(); deadline = time.monotonic()+grace
        while True:
            live = capture()
            for member in live:
                key = tuple(member[k] for k in ('pid','start','uid'))
                if key in sent or signum == signal.SIGKILL and key in protected: continue
                _signal_identity(member, signum); sent.add(key)
            if not live or time.monotonic() >= deadline: break
            time.sleep(.05)
    remaining = capture(); unverified = False
    for record in records:
        if isinstance(record, dict) and type(record.get('pid')) is int:
            current = process_identity(record['pid'])
            if current is not None and current != record: unverified = True
            remaining.extend(_group_members(record['pid']))
        process = _children.get(record.get('pid')) if isinstance(record, dict) else None
        if process is not None:
            try:
                process.wait(timeout=5);_children.pop(record['pid'],None)
            except subprocess.TimeoutExpired: unverified = True
    remaining = capture() + [member for record in records if isinstance(record,dict) and type(record.get('pid')) is int for member in _group_members(record['pid'])]
    for descriptor in supervisors:
        value = _supervisor_state(state, descriptor)
        if not value or value.get('all_children_reaped') is not True: unverified = True
    if remaining or unverified:
        return {'status': 'incomplete', 'reason': 'Verified workers remain or process ownership could not be established',
                'running_receipt': str(marker)}
    marker.unlink()
    return {'status': 'stopped'}


def _check_available_port(port):
    # Match a normal HTTP server's TCP reuse behavior. TIME_WAIT is not a live
    # listener, while an active listener on the same address still prevents bind.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError as error:
            raise SetupError('The engine port is occupied; choose a free port instead of stopping another service') from error


def prepare_engine(home, spec, profile_path, *, model_id, key_file=None):
    """Stage the selected engine without starting or restarting any service."""
    if not isinstance(spec, dict):
        raise SetupError('Engine setup must be an object')
    if spec.get('kind', 'tensorfold') != 'tensorfold':
        from .backend_setup import prepare_backend
        return prepare_backend(home, spec, profile_path, model_id=model_id, key_file=key_file)
    from .native_setup import prepare
    return prepare(home, spec, profile_path, model_id=model_id, key_file=key_file)



def check_engine_setup(home, spec, profile, *, model_id):
    """Read-only model, runtime, architecture and speculative-decoding checks."""
    _require_pidfds()
    if spec.get('kind', 'tensorfold') != 'tensorfold':
        from .backend_setup import check_backend
        return check_backend(home, spec, profile, model_id=model_id)
    from .native_setup import check
    return check(home, spec, profile, model_id=model_id)



def activate_engine(deployment, *, timeout=None):
    """Start only this managed deployment, then verify its effective profile."""
    if deployment.get('runtime') == 'native' and deployment.get('hardware_verified') is not True:
        raise SetupError('Native activation requires a verified NVIDIA compute capability; prepare again on the model host')
    if deployment.get("remote_units"):
        from .native_cluster import activate
        return activate(deployment, timeout=timeout)
    timeout = deployment.get('startup_timeout', 900) if timeout is None else timeout
    _require_pidfds()
    deadline = time.monotonic() + timeout
    state = Path(deployment['state'])
    running = state/'running.json'
    if running.exists():
        previous = read_json(running)
        try:
            if previous.get('supervisors') and not _supervisors_running(state, previous['supervisors']):
                raise SetupError('Managed supervisor is not serving its recorded command')
            check = check_engine(deployment['base_url'], deployment['model'], deployment['profile'],
                                 deployment['key_file'], deadline=deadline)
            previous['owned_members'] = _capture_owned_members(previous.get('processes', []), previous.get('owned_members', []))
            private_json(running, previous)
            return {**deployment, 'status': 'running', 'engine_check': check}
        except SetupError:
            alive = []
            for record in [*previous.get('processes', []), *previous.get('owned_members', [])]:
                if isinstance(record, dict) and process_identity(record['pid']) == record:
                    alive.append(record['pid'])
            for record in previous.get('processes', []):
                if isinstance(record, dict) and type(record.get('pid')) is int:
                    alive.extend(member['pid'] for member in _group_members(record['pid']))
            if alive:
                raise SetupError('A managed engine may still be running; use its recorded shutdown command before replacing it')
            if previous.get('supervisors') and any(not (_supervisor_state(state,row) or {}).get('all_children_reaped') for row in previous['supervisors']):
                raise SetupError('Managed supervisor completion is unverified; preserve its running.json receipt')
            running.unlink()
    parsed = urlsplit(deployment['base_url'])
    _check_available_port(parsed.port)
    if deployment['install_dependencies']:
        command = [deployment['python'], '-m', 'pip', 'install', *deployment['requirements']]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise SetupError('Engine dependency installation failed')
    children = []
    identities = []
    supervisors = []
    members = []
    log_fd = os.open(state/'engine.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(log_fd, 'ab') as log:
            for index, entry in enumerate(deployment['commands']):
                receipt = 'supervisor-'+str(index)+'.json'
                command = [sys.executable, str(Path(__file__).with_name('engine_supervisor.py')), '--state', str(state/receipt), '--', *entry['argv']]
                process = subprocess.Popen(command, env={**os.environ, **entry['env']},
                    cwd=deployment['source'], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                children.append(process)
                _children[process.pid] = process
                identity = process_identity(process.pid, include_zombies=True)
                if identity is None:
                    identities.append({'pid': process.pid, 'start': None, 'uid': os.getuid()})
                    private_json(running, {'processes': identities, 'owned_members': members, 'supervisors': supervisors})
                    raise SetupError('The managed engine exited before startup could be recorded')
                identities.append(identity)
                supervisors.append({'process':identity, 'receipt':receipt})
                members = _capture_owned_members(identities, members)
                private_json(running, {'processes': identities, 'owned_members': members, 'supervisors': supervisors})
        while time.monotonic() < deadline:
            members = _capture_owned_members(identities, members)
            private_json(running, {'processes': identities, 'owned_members': members, 'supervisors': supervisors})
            if any(process.poll() is not None for process in children):
                raise SetupError('The managed engine exited during startup; inspect its private engine.log')
            for descriptor in supervisors:
                value=_supervisor_state(state,descriptor)
                if value and value.get('command') and process_identity(value['command']['pid'])!=value['command']:
                    raise SetupError('The managed engine exited during startup; inspect its private engine.log')
            try:
                models = model_rows(deployment['base_url'], timeout=min(5, remaining_startup_time(deadline)))
            except SetupError:
                time.sleep(min(.5, max(0, deadline - time.monotonic())))
                continue
            if not any(model['id'] == deployment['model'] for model in models):
                raise SetupError('The managed engine did not serve the configured model')
            if not _supervisors_running(state, supervisors):
                time.sleep(min(.05, max(0, deadline-time.monotonic())));continue
            check = check_engine(deployment['base_url'], deployment['model'], deployment['profile'],
                                 deployment['key_file'], deadline=deadline)
            if not _supervisors_running(state, supervisors):
                raise SetupError('The managed command exited before activation completed')
            members = _capture_owned_members(identities, members)
            private_json(running, {'processes': identities, 'owned_members': members, 'supervisors': supervisors})
            result = {**deployment, 'status': 'running', 'engine_check': check}
            private_json(state/'deployment.json', result)
            return result
        raise SetupError('Engine startup timed out before profile activation was confirmed')
    except BaseException as error:
        if identities:
            members = _capture_owned_members(identities, members)
            private_json(running, {'processes': identities, 'owned_members': members, 'supervisors': supervisors})
            stopped = stop_engine(deployment)
            if stopped['status'] != 'stopped':
                raise SetupError('Engine startup failed and cleanup is incomplete; preserve and inspect its running.json receipt') from error
        raise
