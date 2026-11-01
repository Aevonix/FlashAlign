# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Owned-container lifecycle and native-profile readiness checks."""
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import urllib.request


def run(command, *, timeout=30):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Command failed: ' + command[0])
    return result.stdout.strip()


def save(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.flashalign-', delete=False) as stream:
        os.chmod(stream.name, 0o600)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def private_directory(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise ValueError('Deployment directory must not be a symlink')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('Deployment directory must be owned and private')
    return path


def process_identity(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_container(record):
    result = subprocess.run(['docker', 'inspect', '--format',
        '{"id":{{json .Id}},"image":{{json .Image}},"labels":{{json .Config.Labels}},"running":{{json .State.Running}}}',
        record['container_id']], capture_output=True, text=True, timeout=20)
    if result.returncode:
        return None
    data = json.loads(result.stdout)
    if (data.get('id') != record['container_id'] or data.get('image') != record['image_id']
            or any((data.get('labels') or {}).get(key) != value for key, value in record['labels'].items())):
        raise ValueError('Container identity or ownership labels differ')
    return data


def stop_owned(record):
    data = owned_container(record)
    if data is None or not data['running']:
        return
    # Docker sends SIGTERM first; its bounded fallback applies only to this
    # already identity-checked container. No host process name is signalled.
    run(['docker', 'stop', '--time', '10', record['container_id']], timeout=25)
    data = owned_container(record)
    if data is not None and data['running']:
        raise RuntimeError('Owned container did not stop')


def gpu_inventory():
    result = run(['nvidia-smi', '--query-gpu=index,uuid,compute_cap,memory.total,memory.free',
                  '--format=csv,noheader,nounits'], timeout=20)
    rows = []
    for line in result.splitlines():
        index, uuid, capability, total, free = [field.strip() for field in line.split(',')]
        rows.append({'index': index, 'uuid': uuid, 'capability': capability,
                     'total': float(total) * 1024**2, 'free': float(free) * 1024**2})
    return rows


def available_memory():
    entries = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(entries['MemAvailable'].split()[0]) * 1024
    root = Path('/sys/fs/cgroup')
    current = root
    relative = next((line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::')), '')
    candidate = (root / relative.lstrip('/')).resolve()
    if candidate.is_relative_to(root) and candidate.exists():
        current = candidate
    while current.is_relative_to(root):
        maximum, used = current / 'memory.max', current / 'memory.current'
        if maximum.exists() and used.exists() and maximum.read_text().strip() != 'max':
            counts = dict(line.split() for line in (current / 'memory.stat').read_text().splitlines())
            reclaimable = min(int(counts.get('inactive_file', 0)), max(0, int(counts.get('file', 0))
                - int(counts.get('shmem', 0)) - int(counts.get('unevictable', 0))
                - int(counts.get('file_dirty', 0)) - int(counts.get('file_writeback', 0))))
            limit = int(maximum.read_text())
            available = min(available, max(0, limit - int(used.read_text())) + reclaimable)
        if current == root:
            break
        current = current.parent
    return available


def resources(devices, *, allow_untested_memory=False, inventory=None, memory=None):
    inventory = gpu_inventory() if inventory is None else inventory
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('This native DeepSeek backend requires four distinct GPUs')
    selected = []
    for device in devices:
        matches = [row for row in inventory if device in (row['index'], row['uuid'])]
        if len(matches) != 1:
            raise ValueError('Selected GPU is absent or ambiguous')
        selected += matches
    if any(row['capability'] != '12.0' or row['total'] < 94 * 1024**3
           or row['free'] < 90 * 1024**3 for row in selected):
        raise ValueError('This native path requires four free SM120 GPUs with at least 94 GiB each')
    memory = available_memory() if memory is None else memory
    if memory < 24 * 1024**3:
        raise ValueError('The 24 GiB host/cgroup runtime memory floor is not met')
    if not allow_untested_memory and memory < 221 * 1024**3:
        raise ValueError('This native path requires 221 GiB startup headroom; lower-memory starts are unqualified')
    return {'gpu_count': 4, 'available_memory_bytes': memory,
            'below_startup_policy': memory < 221 * 1024**3}


def read_receipt(path, *, limit=1024 * 1024, after=0):
    """Read an owned, bounded receipt without following symlinks or a rewrite."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) & 0o077
                or not 0 < before.st_size <= limit or before.st_mtime < after):
            raise ValueError('Receipt must be fresh, owned, private and bounded')
        raw = os.pread(fd, before.st_size, 0)
        current = os.fstat(fd)
        if len(raw) != before.st_size or any(getattr(before, field) != getattr(current, field)
                for field in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('Receipt changed while reading')
        return raw
    finally:
        os.close(fd)


def native_coverage(directory, profile, *, started_unix):
    files = sorted(Path(directory).glob('rank-*-pid-*.json'))
    if len(files) != 4:
        raise ValueError('Waiting for all four native worker receipts')
    spec = profile.spec
    pids, ranks, records = set(), set(), []
    for path in files:
        raw = read_receipt(path, after=started_unix)
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError('Native receipt must be an object')
        rank, pid = row.get('tp_rank'), row.get('pid')
        if (type(rank) is not int or rank not in range(4) or rank in ranks
                or type(pid) is not int or pid <= 0 or pid in pids
                or path.name != f'rank-{rank}-pid-{pid}.json'
                or row.get('kind') != 'native_wo_b_worker_coverage'
                or row.get('complete') is not True or row.get('profile') != profile.name
                or row.get('profile_digest') != profile.digest
                or row.get('manifest_sha256') != spec.raw_sha256
                or row.get('payload_sha256') != spec.payload_sha256
                or row.get('tp_world_size') != 4
                or set(row.get('layers', {})) != {str(layer) for layer in spec.layers}):
            raise ValueError('Native worker identity or coverage mismatch')
        for layer in spec.layers:
            bound = row['layers'][str(layer)]
            if (bound.get('layer') != layer or bound.get('tp_rank') != rank
                    or bound.get('profile_digest') != profile.digest
                    or any(bound.get(field) is not True for field in ('shared_stock_wo_a',
                        'packed_values_exact', 'packed_row_and_mma_scales_exact', 'stock_storage_unchanged'))):
                raise ValueError('Native matrix binding differs')
            tensors = bound.get('tensors', [])
            names = {f'layers.{layer}.attn.wo_b.{suffix}' for suffix in ('weight', 'scale')}
            if len(tensors) != 2 or {item.get('tensor') for item in tensors} != names:
                raise ValueError('Native matrix tensor set differs')
            for item in tensors:
                target = spec.tensor(item['tensor'])
                if (item.get('stock_sha256') != target.stock_partitions[rank]
                        or item.get('edited_sha256') != target.edited_partitions[rank]
                        or item.get('device_sha256') != target.edited_partitions[rank]
                        or item.get('independent_storage') is not True
                        or item.get('bytes') != target.shape[0] * target.shape[1] // 4):
                    raise ValueError('Native partition bytes differ')
        pids.add(pid); ranks.add(rank)
        records.append({'rank': rank, 'pid': pid, 'sha256': hashlib.sha256(raw).hexdigest()})
    return {'bindings': len(spec.layers) * 4, 'partitions': len(spec.tensors) * 4, 'workers': records}


def healthy(host, port, served_model, profile):
    host = '127.0.0.1' if host == '0.0.0.0' else host
    base = 'http://' + host + ':' + str(port)
    with urllib.request.urlopen(base + '/health', timeout=5) as response:
        if response.status != 200:
            raise ValueError('Engine health failed')
    with urllib.request.urlopen(base + '/v1/models', timeout=5) as response:
        models = json.load(response)['data']
    if (len(models) != 1 or models[0].get('id') != served_model
            or type(models[0].get('max_model_len')) is not int
            or not 0 < models[0]['max_model_len'] <= 1048576):
        raise ValueError('Engine model or context differs')
    with urllib.request.urlopen(base + '/v1/owner-mode/profiles', timeout=5) as response:
        profiles = json.load(response)['data']
    if len(profiles) != 1 or profiles[0].get('name') != profile.name or profiles[0].get('loaded') is not True:
        raise ValueError('Native profile is not loaded')
    return models[0]['max_model_len']
