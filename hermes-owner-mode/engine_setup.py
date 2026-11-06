# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Prepare and run an isolated TensorFold deployment from a checked engine bundle."""
import hashlib
import concurrent.futures
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
from .runtime_environment import python_environment
from . import engine_supervisor as _unit_supervisor
from . import native_remote as _native_units

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
        effective = {**os.environ, **python_environment(python, environment), 'CPATH': cpath}
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


def _stop_legacy_engine(deployment, *, stop_deadline=None, retain_marker=False):
    """Stop verified descendants; keep subreapers alive until children are reaped."""
    if deployment.get("remote_units"):
        from .native_cluster import stop
        return stop(deployment)
    state = Path(deployment['state']); marker = state/'running.json'
    stop_deadline = time.monotonic()+_native_units.stop_timeout(deployment.get('stop_timeout', 60)) if stop_deadline is None else stop_deadline
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
        sent = set(); deadline = min(stop_deadline, time.monotonic()+grace)
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
                process.wait(timeout=max(.001, min(5, stop_deadline-time.monotonic())));_children.pop(record['pid'],None)
            except subprocess.TimeoutExpired: unverified = True
    remaining = capture() + [member for record in records if isinstance(record,dict) and type(record.get('pid')) is int for member in _group_members(record['pid'])]
    for descriptor in supervisors:
        value = _supervisor_state(state, descriptor)
        if not value or value.get('all_children_reaped') is not True: unverified = True
    if remaining or unverified:
        return {'status': 'incomplete', 'reason': 'Verified workers remain or process ownership could not be established',
                'running_receipt': str(marker)}
    if not retain_marker:
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
    from .backend_setup import prepare_backend
    return prepare_backend(home, spec, profile_path, model_id=model_id, key_file=key_file)



def check_engine_setup(home, spec, profile, *, model_id):
    """Read-only model, runtime, architecture and speculative-decoding checks."""
    from .engine_adapters import engine_adapters
    adapter = engine_adapters().get(spec.get('kind', 'tensorfold'))
    if adapter.requires_local_supervisor:
        _require_pidfds()
    from .backend_setup import check_backend
    return check_backend(home, spec, profile, model_id=model_id)


def _activation(state):
    value = _unit_supervisor.read_unit_record(state/'activation.json')
    if value is None:
        return None
    nonce = value.get('nonce')
    if (type(value.get('version')) is not int or value['version'] != 1
            or not isinstance(nonce, str) or len(nonce) != 32
            or any(c not in '0123456789abcdef' for c in nonce)
            or value.get('phase') not in ('starting', 'running', 'stopping', 'stopped')
            or not isinstance(value.get('units'), list) or len(value['units']) > 64
            or type(value.get('legacy', False)) is not bool
            or not value['units'] and value.get('legacy') is not True
            or value.get('units') != [str(state/'launches'/nonce/f'unit-{i}')
                                     for i in range(len(value['units']))]):
        raise SetupError('Invalid managed activation ownership record')
    return value


def _unit_states(attempt, previous=None):
    expected = {row['unit']: row['process'] for row in (previous or {}).get('supervisors', [])
                if previous.get('nonce') == attempt['nonce'] and 'unit' in row}
    rows = []
    for path in attempt['units']:
        row = _native_units.state_of(path)
        virgin = row == {'status': 'stopped', 'all_children_reaped': True}
        if (not virgin and row.get('nonce') != attempt['nonce']
                or path in expected and row.get('supervisor') != expected[path]):
            raise SetupError('Managed unit ownership differs from its activation')
        rows.append(row)
    return rows


def _units_running(rows):
    return bool(rows) and all(row.get('status') == 'running'
                              and row.get('supervisor_alive') is True
                              and row.get('command_alive') is True for row in rows)


def _running_record(attempt, rows, previous=None):
    supervisors = [{'process': row['supervisor'], 'unit': path,
                    'receipt': 'supervisor.json'}
                   for path, row in zip(attempt['units'], rows) if row.get('supervisor')]
    processes = [row['process'] for row in supervisors]
    members = previous.get('owned_members', []) if previous and previous.get('nonce') == attempt['nonce'] else []
    return {'version': 2, 'nonce': attempt['nonce'], 'processes': processes,
            'supervisors': supervisors,
            'owned_members': _capture_owned_members(processes, members)}


def _publish_local(deployment, nonce, *, result=None):
    """A health result may publish only to the same uncancelled attempt."""
    state = Path(deployment['state'])
    observed = _activation(state)
    if observed is None or observed['nonce'] != nonce:
        raise SetupError('Managed activation was stopped or replaced during startup')
    # Unit locks may briefly wait for a spawning supervisor. Never hold the
    # deployment lock across those waits or an engine readiness request.
    previous = _unit_supervisor.read_unit_record(state/'running.json')
    rows = _unit_states(observed, previous)
    with _unit_supervisor.unit_lock(state):
        current = _activation(state)
        if current is None or current['nonce'] != nonce or current['phase'] not in ('starting', 'running'):
            raise SetupError('Managed activation was stopped or replaced during startup')
        if result is not None and not _units_running(rows):
            raise SetupError('The managed command exited before activation completed')
        previous = _unit_supervisor.read_unit_record(state/'running.json')
        _unit_supervisor.save(state/'running.json', _running_record(current, rows, previous))
        if result is not None:
            _unit_supervisor.save(state/'activation.json', {**current, 'phase': 'running'})
            _unit_supervisor.save(state/'deployment.json', result)
        return rows


def _remove_stopped_marker(state, current):
    marker = _unit_supervisor.read_unit_record(state/'running.json')
    if marker is None:
        return
    if marker.get('nonce') != current['nonce'] and not (current.get('legacy') and 'nonce' not in marker):
        raise SetupError('Managed running receipt belongs to a different attempt')
    (state/'running.json').unlink()
    _sync_directory(state)


def _sync_directory(state):
    directory = os.open(state, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _new_local_attempt(deployment):
    """Publish every possible unit before any command can be spawned."""
    state = Path(deployment['state'])
    with _unit_supervisor.unit_lock(state):
        current = _activation(state)
        if current is not None and current['phase'] != 'stopped':
            if current['phase'] == 'running':
                return current, False
            raise SetupError('A managed activation is pending or still running; use its recorded shutdown command')
        if current is not None:
            _remove_stopped_marker(state, current)
        legacy = _unit_supervisor.read_unit_record(state/'running.json')
        if legacy is not None:
            return {'legacy': legacy}, False
        entries = deployment.get('commands')
        if (not isinstance(entries, list) or not 1 <= len(entries) <= 64
                or any(not isinstance(entry, dict) or not isinstance(entry.get('argv'), list)
                       or not entry['argv'] or any(not isinstance(arg, str) for arg in entry['argv'])
                       or not isinstance(entry.get('env'), dict)
                       or any(not isinstance(key, str) or not isinstance(value, str)
                              for key, value in entry['env'].items()) for entry in entries)):
            raise SetupError('Invalid managed command declaration')
        nonce = secrets.token_hex(16)
        root = state/'launches'/nonce
        root.mkdir(parents=True, mode=0o700)
        source = Path(__file__).with_name('engine_supervisor.py').read_bytes()
        units = []
        for index, entry in enumerate(entries):
            unit = root/f'unit-{index}'
            unit.mkdir(mode=0o700)
            environment = entry['env']
            if entry['argv'][0] == deployment.get('python'):
                environment = python_environment(deployment['python'], environment)
            _unit_supervisor.save(unit/'unit.json', {'argv': entry['argv'], 'env': environment,
                                                   'cwd': str(Path(deployment['source']).absolute())})
            fd = os.open(unit/'engine_supervisor.py', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(source); stream.flush(); os.fsync(stream.fileno())
            files = {name: {'bytes': (unit/name).stat().st_size, 'sha256': digest(unit/name)}
                     for name in ('unit.json', 'engine_supervisor.py')}
            _unit_supervisor.save(unit/'stage-receipt.json', {'files': files})
            units.append(str(unit))
        _sync_directory(root)
        _sync_directory(root.parent)
        current = {'version': 1, 'nonce': nonce, 'phase': 'starting', 'units': units}
        _unit_supervisor.save(state/'activation.json', current)
        return current, True


def _external_cleanup_contract(deployment):
    """Validate one optional cleanup helper from the immutable staged engine kit."""
    spec = deployment.get('external_cleanup')
    if spec is None:
        if deployment.get('container_runtime') is not None:
            raise SetupError('External engine cleanup is not declared; prepare this deployment again')
        return None
    if (not isinstance(spec, dict)
            or set(spec) != {'schema', 'script', 'script_sha256', 'timeout_seconds'}
            or spec.get('schema') != 'flashalign.external-cleanup.v1'
            or not isinstance(spec.get('script'), str)
            or Path(spec['script']).name != spec['script']
            or spec['script'] in ('', '.', '..') or '\\' in spec['script']
            or not isinstance(spec.get('script_sha256'), str)
            or len(spec['script_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in spec['script_sha256'])):
        raise SetupError('Invalid external engine cleanup declaration')
    timeout = _native_units.stop_timeout(spec['timeout_seconds'])
    if (not isinstance(deployment.get('source'), str)
            or not isinstance(deployment.get('python'), str)
            or not Path(deployment['python']).is_absolute()
            or not isinstance(deployment.get('integration_files'), dict)):
        raise SetupError('Invalid staged external engine cleanup configuration')
    source = Path(deployment['source'])
    script = source/spec['script']
    if (not source.is_absolute() or source.is_symlink() or script.is_symlink()
            or not script.is_file() or script.stat().st_uid != os.getuid()
            or deployment.get('integration_files', {}).get(spec['script']) != spec['script_sha256']
            or digest(script) != spec['script_sha256']):
        raise SetupError('The staged external engine cleanup helper is unverified')
    return {**spec, 'timeout_seconds': timeout, 'path': str(script)}


def _verify_external_cleanup(deployment, contract, nonce, deadline):
    """Run outside managed unit ancestry; never infer external cleanup from reaping."""
    until = min(deadline, time.monotonic()+contract['timeout_seconds'])
    if until-time.monotonic() <= 0:
        return False
    verified = False
    unit = None
    cleanup_nonce = secrets.token_hex(16)
    try:
        parent = _native_units.owned_directory(Path(deployment['state'])/'external-cleanup-units', create=True)
        # A prior interrupted call retains its owned subreaper. Resolve it before
        # starting another helper; never orphan a helper's Docker CLI children.
        for previous in sorted(parent.iterdir()):
            row = _native_units.state_of(previous, lock_timeout=max(.001, until-time.monotonic()))
            if row.get('supervisor_alive') or row.get('all_children_reaped') is not True:
                remaining = until-time.monotonic()
                if remaining < 1 or _native_units.stop({'state': str(previous),
                        'timeout_seconds': remaining}).get('status') != 'stopped':
                    raise SetupError('Prior external cleanup remains unresolved')
        now = time.monotonic()
        work_until = until-min(10., max(.1, until-now-.1))
        if work_until <= now:
            raise SetupError('External cleanup has no remaining work budget')
        unit = parent/cleanup_nonce
        unit.mkdir(mode=0o700)
        _unit_supervisor.save(unit/'unit.json', {'argv': [deployment['python'], '-B', contract['path'],
            '--state-dir', deployment['state'], '--deadline-monotonic', str(work_until)],
            'cwd': deployment['source'], 'env': python_environment(deployment['python'])})
        source = Path(__file__).with_name('engine_supervisor.py').read_bytes()
        fd = os.open(unit/'engine_supervisor.py', os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(source); stream.flush(); os.fsync(stream.fileno())
        _unit_supervisor.save(unit/'stage-receipt.json', {'files': {
            name: {'bytes': (unit/name).stat().st_size, 'sha256': digest(unit/name)}
            for name in ('unit.json', 'engine_supervisor.py')}})
        _native_units.start({'state': str(unit), 'nonce': cleanup_nonce})
        while time.monotonic() < work_until:
            row = _native_units.state_of(unit, lock_timeout=max(.001, work_until-time.monotonic()))
            if row.get('all_children_reaped') is True and not row.get('supervisor_alive'):
                if row.get('exit_code') == 0 and (unit/'engine.log').stat().st_size <= 65536:
                    value = _unit_supervisor.read_unit_record(unit/'engine.log') or {}
                    verified = (value.get('schema') == 'flashalign.external-cleanup.v1'
                                and value.get('status') == 'stopped'
                                and value.get('external_resources_stopped') is True)
                break
            time.sleep(min(.02, max(0, work_until-time.monotonic())))
    except (OSError, ValueError, SetupError, subprocess.SubprocessError):
        verified = False
    finally:
        if unit is not None:
            remaining = until-time.monotonic()
            try:
                if remaining < 1:
                    verified = False
                else:
                    stopped = _native_units.stop({'state': str(unit), 'nonce': cleanup_nonce,
                                                 'timeout_seconds': remaining})
                    verified = (verified and stopped.get('status') == 'stopped'
                                and stopped.get('all_children_reaped') is True)
            except (OSError, ValueError, subprocess.SubprocessError):
                verified = False
    _unit_supervisor.save(Path(deployment['state'])/'external-cleanup.json', {
        'schema': 'flashalign.external-cleanup.v1', 'nonce': nonce,
        'script_sha256': contract['script_sha256'],
        'status': 'stopped' if verified else 'incomplete',
        'external_resources_stopped': verified})
    return verified


def _stop_local_attempt(deployment, nonce=None):
    state = Path(deployment['state'])
    deadline = time.monotonic()+_native_units.stop_timeout(deployment.get('stop_timeout', 60))
    with _unit_supervisor.unit_lock(state, timeout=max(.001, deadline-time.monotonic())):
        current = _activation(state)
        if nonce is not None and (current is None or current['nonce'] != nonce):
            return {'status': 'stopped', 'superseded': True}
        external = _external_cleanup_contract(deployment)
        if current is None:
            # Fence a legacy stop against a simultaneous new-style activation.
            current = {'version': 1, 'nonce': secrets.token_hex(16), 'phase': 'stopping',
                       'units': [], 'legacy': True}
        elif current['phase'] == 'stopped' and external is None:
            _remove_stopped_marker(state, current)
            return {'status': 'stopped'}
        _unit_supervisor.save(state/'activation.json', {**current, 'phase': 'stopping'})
        recorded = _unit_supervisor.read_unit_record(state/'running.json')
    # Leave room for the declared external-resource check inside the same total
    # stop budget. Its subprocess must not be a child of the unit being reaped.
    unit_deadline = deadline-external['timeout_seconds'] if external else deadline
    if current.get('legacy'):
        result = _stop_legacy_engine(deployment, stop_deadline=unit_deadline,
                                     retain_marker=external is not None)
    else:
        def stop_unit(path):
            remaining = unit_deadline-time.monotonic()
            if remaining < 1:
                return {'status': 'incomplete', 'reason': 'Deployment shutdown budget expired'}
            outcome = _native_units.stop({'state': path, 'nonce': current['nonce'], 'timeout_seconds': remaining})
            if outcome.get('status') == 'stopped' and outcome.get('all_children_reaped') is True:
                owner = _native_units.state_of(path, lock_timeout=max(.001, unit_deadline-time.monotonic())).get('supervisor')
                process = _children.get(owner['pid']) if owner else None
                if process is not None and process.poll() is not None:
                    process.wait(timeout=max(.001, min(1, unit_deadline-time.monotonic())))
                    _children.pop(owner['pid'], None)
            return outcome
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(1, len(current['units'])))) as pool:
            outcomes = list(pool.map(stop_unit, current['units']))
        if recorded is not None and recorded.get('nonce') == current['nonce']:
            _stop_recorded_orphans(recorded, stop_deadline=unit_deadline)
        result = {'status': 'stopped'} if all(row.get('status') == 'stopped'
                    and row.get('all_children_reaped') is True for row in outcomes) else {
                        'status': 'incomplete', 'reason': 'Managed unit cleanup is not confirmed'}
    with _unit_supervisor.unit_lock(state, timeout=max(.001, deadline-time.monotonic())):
        latest = _activation(state)
        if latest is None or latest['nonce'] != current['nonce']:
            return {'status': 'incomplete', 'reason': 'Managed activation ownership changed during cleanup'}
        if (result['status'] == 'stopped' and external is not None
                and not _verify_external_cleanup(deployment, external, current['nonce'], deadline)):
            result = {'status': 'incomplete', 'reason': 'External engine cleanup is not confirmed'}
        if result['status'] == 'stopped':
            _unit_supervisor.save(state/'activation.json', {**latest, 'phase': 'stopped'})
            _remove_stopped_marker(state, current)
    return result


def _stop_recorded_orphans(recorded, *, stop_deadline=None):
    """Retain legacy recovery of witnessed children after an anchor crash.

    This never establishes complete reaping. The unit protocol must still
    report incomplete when its supervisor could not attest cleanup.
    """
    roots = recorded.get('processes', [])
    if not roots or any(process_identity(row['pid']) == row for row in roots):
        return
    protected = {tuple(row[key] for key in ('pid', 'start', 'uid')) for row in roots}
    members = recorded.get('owned_members', [])
    for signum in (signal.SIGTERM, signal.SIGKILL):
        deadline = min(time.monotonic()+5, stop_deadline) if stop_deadline is not None else time.monotonic()+5
        while time.monotonic() < deadline:
            members = _capture_owned_members(roots, members)
            live = [row for row in members if process_identity(row['pid']) == row
                    and tuple(row[key] for key in ('pid', 'start', 'uid')) not in protected]
            if not live:
                return
            for row in live:
                _signal_identity(row, signum)
            time.sleep(.05)


def stop_engine(deployment):
    """Cancel known intents, including launches with no final running receipt."""
    if deployment.get('remote_units'):
        from .native_cluster import stop
        return stop(deployment)
    try:
        return _stop_local_attempt(deployment)
    except (OSError, ValueError, SetupError):
        return {'status': 'incomplete', 'reason': 'Managed activation ownership or cleanup is unverified'}


class ManagedNotReady(SetupError):
    """A managed command is owned but its protocol is not ready yet."""


def _engine_readiness(deployment, rows, deadline):
    try:
        models = model_rows(deployment['base_url'], timeout=min(5, remaining_startup_time(deadline)))
    except SetupError as error:
        raise ManagedNotReady(str(error)) from error
    if not any(row['id'] == deployment['model'] for row in models):
        raise SetupError('The managed engine did not serve the configured model')
    from .lifecycle_setup import check_lifecycle
    try:
        lifecycle = check_lifecycle(deployment, timeout=min(5, remaining_startup_time(deadline)))
    except SetupError as error:
        raise ManagedNotReady(str(error)) from error
    check = check_engine(deployment['base_url'], deployment['model'], deployment['profile'],
                         deployment['key_file'], deadline=deadline)
    return {'engine_check': check, **lifecycle}


def _reuse_legacy(deployment, previous, deadline, readiness, result_base):
    state = Path(deployment['state'])
    if previous.get('supervisors') and not _supervisors_running(state, previous['supervisors']):
        raise SetupError('Managed supervisor is not serving its recorded command; stop it before replacement')
    rows = [_supervisor_state(state, row) for row in previous.get('supervisors', [])]
    result = {**result_base, 'status': 'running', **readiness(deployment, rows, deadline)}
    with _unit_supervisor.unit_lock(state):
        if _activation(state) is not None or _unit_supervisor.read_unit_record(state/'running.json') != previous:
            raise SetupError('Managed legacy activation changed during its health check')
    return result


def activate_engine(deployment, *, timeout=None):
    """Start a serialized durable attempt without locking out a startup stop."""
    if deployment.get('runtime') == 'native' and deployment.get('hardware_verified') is not True:
        raise SetupError('Native activation requires a verified NVIDIA compute capability; prepare again on the model host')
    if deployment.get('remote_units'):
        from .native_cluster import activate
        return activate(deployment, timeout=timeout)
    return activate_managed_commands(deployment, readiness=_engine_readiness, timeout=timeout)


def activate_managed_commands(deployment, *, readiness, timeout=None, result_base=None):
    """Own local command lifetimes and delegate only protocol readiness.

    The callback receives verified unit states and an absolute deadline. It
    returns result fields, or raises ManagedNotReady for a bounded retry.
    """
    if deployment.get('install_dependencies') is not False:
        raise SetupError('Install dependencies during preparation before activating the managed engine')
    result_base = deployment if result_base is None else result_base
    timeout = deployment.get('startup_timeout', 900) if timeout is None else timeout
    _require_pidfds()
    deadline = time.monotonic() + timeout
    attempt, fresh = _new_local_attempt(deployment)
    if 'legacy' in attempt:
        return _reuse_legacy(deployment, attempt['legacy'], deadline, readiness, result_base)
    try:
        if fresh:
            _check_available_port(urlsplit(deployment['base_url']).port)
            for path in attempt['units']:
                # A simultaneous stop installs a nonce tombstone even when this
                # start has not reached the unit yet, fencing all delayed calls.
                started = _native_units.start({'state': path, 'nonce': attempt['nonce']})
                owner = started.get('supervisor')
                if owner and owner['pid'] in _native_units._children:
                    _children[owner['pid']] = _native_units._children[owner['pid']]
                _publish_local(deployment, attempt['nonce'])
        while time.monotonic() < deadline:
            rows = _publish_local(deployment, attempt['nonce'])
            if any(row.get('all_children_reaped') is True
                   or row.get('status') in ('stopping', 'stopped', 'cleanup_incomplete')
                   or row.get('command') and not row.get('command_alive') for row in rows):
                raise SetupError('The managed engine exited during startup; inspect its private unit log')
            if not _units_running(rows):
                time.sleep(min(.05, max(0, deadline-time.monotonic())))
                continue
            try:
                fields = readiness(deployment, rows, deadline)
            except ManagedNotReady:
                time.sleep(min(.05, max(0, deadline-time.monotonic())))
                continue
            result = {**result_base, 'status': 'running', **fields}
            _publish_local(deployment, attempt['nonce'], result=result)
            return result
        raise SetupError('Engine startup timed out before profile activation was confirmed')
    except BaseException as error:
        if not fresh:
            # A transient health failure cannot tear down an earlier owner's
            # healthy deployment. Only this call's fresh attempt is unwound.
            raise
        stopped = _stop_local_attempt(deployment, attempt['nonce'])
        if stopped['status'] != 'stopped':
            raise SetupError('Engine startup failed and cleanup is incomplete; preserve its activation receipt') from error
        if isinstance(error, (OSError, ValueError)) and not isinstance(error, SetupError):
            raise SetupError('Managed engine launch failed; inspect its private unit receipts') from error
        raise
