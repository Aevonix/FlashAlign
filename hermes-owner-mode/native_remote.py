# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Standalone, owned-directory operations for native engine ranks over SSH."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import secrets
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import time

MAX_PAYLOAD = 4 * 1024**3
_children = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def private_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.new-'+secrets.token_hex(8))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(value, out)
        out.write('\n')
    os.replace(temporary, path)


def safe_relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value or str(p) != value:
        raise ValueError('Unsafe rank staging path')
    return p


def owned_directory(path, create=False):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError('Rank directory must be absolute')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if create and not current.exists():
            current.mkdir(mode=0o700)
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('Rank directory contains a link or non-directory')
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Rank directory must be owned and private')
    return path


def prepare_replay_directory(path):
    """Prepare only the stable parent; the engine initializes its journal leaf."""
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError('Owner replay directory must be an absolute private path')
    directory = Path(path)
    owned_directory(directory.parent, create=True)
    if directory.exists() or directory.is_symlink():
        owned_directory(directory)
    return directory


def member_path(root, name):
    relative = safe_relative(name)
    parent = Path(root)
    for component in relative.parts[:-1]:
        parent /= component
        if parent.exists() and not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError('Rank payload parent is not a regular directory')
    return Path(root).joinpath(*relative.parts)


def inspect_engram(request, model_config):
    contract = request['engram_contract']
    source, packed = request.get('engram_source'), request.get('engram_dir')
    if any(x is not None and (not isinstance(x,str) or not x) for x in (source,packed)):
        raise ValueError('Engram source paths must be nonempty strings')
    if source and packed:
        raise ValueError('Choose one Engram source: original safetensors or packed tables')
    declaration = contract['source']
    if not source and not packed:
        cache = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'hub'
        candidate = cache/('models--'+declaration['id'].replace('/', '--'))/'snapshots'/declaration['revision']
        if candidate.is_dir(): source = str(candidate)
    if not source and not packed:
        raise ValueError('DeepSeek requires original Engram tables: set engram_source to the pinned '+declaration['id']+' checkpoint folder containing config, index and Engram shards, or set engram_dir to validated packed tables')
    config = model_config.get('text_config') or model_config
    layers, counts = config.get('engram_layer_ids'), config.get('engram_num_embeddings')
    if (not isinstance(layers, list) or not layers or len(layers)>64 or not isinstance(counts, list) or len(layers) != len(counts)
            or len(set(layers)) != len(layers) or any(type(x) is not int or not 0 <= x < 4096 for x in layers)
            or any(type(x) is not int or not 0 < x < 2**32 for x in counts)):
        raise ValueError('DeepSeek checkpoint does not declare its Engram geometry')
    path = Path(source or packed).expanduser().resolve()
    receipts = []
    if packed:
        rank, world = request['tp_rank'], request['tp_world']
        for layer, count in zip(layers, counts):
            file = path/f'engram-l{layer}-r{rank}of{world}.bin'
            with file.open('rb') as f: header = f.read(48)
            if len(header) != 48: raise ValueError('Packed Engram header is incomplete')
            magic, actual_layer, lo, hi, total, row_bytes = struct.unpack('<6Q', header)
            if (magic != 0x31344E4531565344 or actual_layer != layer or total != count
                    or row_bytes != 264 or not 0 <= lo < hi <= total
                    or file.stat().st_size != 4096+(hi-lo)*row_bytes):
                raise ValueError('Packed Engram table does not match the selected model and rank')
            receipts.append({'file':file.name,'header_sha256':hashlib.sha256(header).hexdigest(),'bytes':file.stat().st_size})
        return {'kind':'packed','path':str(path),'files':receipts}
    if digest(path/'config.json') != declaration['config_sha256'] or digest(path/'model.safetensors.index.json') != declaration['index_sha256']:
        raise ValueError('Original Engram checkpoint metadata differs from the pinned public source')
    weight_map = json.loads((path/'model.safetensors.index.json').read_text()).get('weight_map', {})
    headers = {}
    for layer, count in zip(layers, counts):
        for suffix, dtype, width in [('weight','F8_E4M3',256),('scale','F8_E8M0',8)]:
            name = f'layers.{layer}.engram.embed.{suffix}'
            shard = weight_map.get(name)
            if not isinstance(shard,str) or Path(shard).name != shard:
                raise ValueError('Original Engram index does not contain the required tables')
            if shard not in headers:
                file = path/shard
                with file.open('rb') as f:
                    prefix=f.read(8)
                    if len(prefix)!=8: raise ValueError('Engram shard header is incomplete')
                    size=struct.unpack('<Q',prefix)[0]
                    if not 2 <= size <= 32*1024**2: raise ValueError('Invalid Engram shard header size')
                    raw=f.read(size)
                if len(raw)!=size: raise ValueError('Engram shard header is incomplete')
                parsed=json.loads(raw);file_size=file.stat().st_size
                offsets=[row.get('data_offsets') for name,row in parsed.items() if name!='__metadata__' and isinstance(row,dict)]
                if (not offsets or any(not isinstance(x,list) or len(x)!=2 or any(type(n) is not int for n in x) or not 0<=x[0]<=x[1] for x in offsets)
                        or 8+size+max(x[1] for x in offsets)!=file_size):
                    raise ValueError('Engram shard file size does not match its complete header')
                headers[shard]=(parsed,8+size,file_size)
                receipts.append({'file':shard,'header_sha256':hashlib.sha256(prefix+raw).hexdigest(),'bytes':file.stat().st_size})
            header, offset, total=headers[shard]
            tensor=header.get(name,{})
            offsets=tensor.get('data_offsets',[])
            if (tensor.get('dtype')!=dtype or tensor.get('shape')!=[count,width] or len(offsets)!=2
                    or any(type(x) is not int for x in offsets) or not 0 <= offsets[0] <= offsets[1]
                    or offsets[1]-offsets[0]!=count*width or offset+offsets[1]>total):
                raise ValueError('Original Engram tensor encoding, shape or file size is invalid')
    return {'kind':'safetensors','path':str(path),'model':declaration,'files':receipts}


def inventory(request):
    path = Path(request['model_path']).expanduser().resolve()
    config_path = path/'config.json'
    config = json.loads(config_path.read_text())
    quant = config.get('quantization_config') or (config.get('text_config') or {}).get('quantization_config')
    index = path/'model.safetensors.index.json'
    result = {
        'platform': platform.system().lower()+'-'+{'arm64': 'aarch64', 'amd64': 'x86_64'}.get(platform.machine().lower(), platform.machine().lower()),
        'home': str(Path.home()), 'model_path': str(path), 'config': config,
        'config_sha256': digest(config_path), 'index_sha256': digest(index) if index.is_file() else None,
        'quantization_config_sha256': hashlib.sha256(json.dumps(quant, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'devices': [], 'active_drafting': [], 'network_interfaces': sorted(p.name for p in Path('/sys/class/net').iterdir()),
    }
    if request.get('engram_contract'):
        result['engram'] = inspect_engram(request, config)
    draft_path = request.get('draft_model')
    if draft_path:
        draft_path = Path(draft_path).expanduser().resolve()
        result['draft_model'] = str(draft_path)
        result['draft_config'] = json.loads((draft_path/'config.json').read_text())
    elif request.get('draft_architectures'):
        candidates = []
        cache = Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'hub'
        for parent in sorted(cache.glob('models--*'))[:128]:
            for file in sorted((parent/'snapshots').glob('*/config.json'))[:8]:
                try:
                    metadata = json.loads(file.read_text())
                    if set(metadata.get('architectures', [])) & set(request['draft_architectures']):
                        candidates.append((file.parent, metadata))
                except (OSError, ValueError, TypeError):
                    continue
        if len(candidates) == 1:
            result['draft_model'], result['draft_config'] = str(candidates[0][0]), candidates[0][1]
    result['devices'], result['cuda_architectures'] = cuda_inventory()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
            if not argv or Path(argv[0]).name not in ('tensorfold-native', 'tensorfold-dsv41') or 'serve' not in argv:
                continue
            at = argv.index('serve')
            cwd = (proc/'cwd').resolve()
            current = Path(argv[at+1]) if at+1 < len(argv) else Path('.')
            if not current.is_absolute():
                current = cwd/current
            if at+1 >= len(argv) or current.resolve() != path:
                continue
            def argument(name):
                for i, value in enumerate(argv):
                    if value.startswith(name+'='):
                        return value.split('=', 1)[1]
                    if value == name and i+1 < len(argv):
                        return argv[i+1]
                return None
            active_draft = argument('--drafter')
            if active_draft and not Path(active_draft).is_absolute():
                active_draft = str(cwd/active_draft)
            result['active_drafting'].append({
                'draft_model': active_draft, 'mtp_drafts': argument('--mtp-drafts'),
                'disabled': '--no-drafts' in argv, 'parallel': argument('--parallel'),
            })
        except (OSError, ValueError, UnicodeError):
            continue
    return result


def cuda_inventory():
    run = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,compute_cap', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if run.returncode:
        raise ValueError('NVIDIA driver inventory is unavailable on this model host')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    occupied = {line.split(',')[0].strip() for line in active.stdout.splitlines() if ',' in line}
    devices, architectures = [], {}
    for line in run.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 5:
            raise ValueError('NVIDIA compute capability is unavailable')
        architectures[fields[0]] = 'sm_'+fields[4].replace('.', '')
        try:
            free = float(fields[2]) < float(fields[3])*0.05
        except ValueError:
            # Unified-memory GB10 reports [N/A]; a successful process query is the available occupancy evidence.
            free = active.returncode == 0 and fields[1] not in occupied
        if free:
            devices.append(fields[0])
    return devices, architectures


def _rank_archive(payload, expected, directory):
    """Read only bounded USTAR headers and copy file bodies in small chunks."""
    source = io.BytesIO(payload) if isinstance(payload, bytes) else payload
    consumed = 0
    def read(size):
        nonlocal consumed
        result = source.read(size)
        consumed += len(result)
        if consumed > MAX_PAYLOAD:
            raise ValueError('Rank payload exceeds size limit')
        return result
    def exact(size):
        chunks = bytearray()
        while len(chunks) < size:
            chunk = read(size-len(chunks))
            if not chunk:
                raise ValueError('Truncated rank payload')
            chunks.extend(chunk)
        return bytes(chunks)
    seen = set()
    while True:
        header = exact(512)
        if header == b'\0'*512:
            if exact(512) != b'\0'*512:
                raise ValueError('Invalid rank archive terminator')
            while True:
                padding = read(8*1024*1024)
                if not padding:
                    return seen
                if any(padding):
                    raise ValueError('Unexpected data after rank archive')
        try:
            member = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        except (tarfile.HeaderError, UnicodeError) as error:
            raise ValueError('Invalid rank archive header') from error
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or member.name not in expected or member.name in seen:
            raise ValueError('Unexpected rank payload member')
        row = expected[member.name]
        if member.size != row['bytes']:
            raise ValueError('Rank payload size differs')
        path = member_path(directory, member.name)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        value = hashlib.sha256()
        remaining = member.size
        with path.open('xb') as output:
            while remaining:
                chunk = read(min(8*1024*1024, remaining))
                if not chunk:
                    raise ValueError('Truncated rank payload')
                output.write(chunk)
                value.update(chunk)
                remaining -= len(chunk)
        if member.size % 512:
            exact(512-member.size%512)
        path.chmod(0o700 if row.get('executable') else 0o600)
        if value.hexdigest() != row['sha256']:
            raise ValueError('Rank payload checksum differs')
        seen.add(member.name)


def stage(request, payload):
    parent = owned_directory(request['parent'], create=True)
    if request.get('owner_replay_dir') is not None:
        prepare_replay_directory(request['owner_replay_dir'])
    destination = parent/request['id']
    if not request['id'].isalnum() or len(request['id']) != 20:
        raise ValueError('Invalid rank deployment ID')
    expected = request['files']
    if not isinstance(expected, dict) or not expected or len(expected) > 8192:
        raise ValueError('Invalid rank inventory')
    total = 0
    for name, row in expected.items():
        safe_relative(name)
        if (not isinstance(row, dict) or set(row) - {'sha256', 'bytes', 'executable'}
                or not isinstance(row.get('sha256'), str)
                or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256'])
                or type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= MAX_PAYLOAD
                or type(row.get('executable', False)) is not bool):
            raise ValueError('Invalid rank file declaration')
        total += row['bytes']
    if total > MAX_PAYLOAD:
        raise ValueError('Rank files exceed payload limit')
    if destination.exists():
        owned_directory(destination)
        for name, row in expected.items():
            p = member_path(destination, name)
            info = p.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid() or digest(p) != row['sha256']:
                raise ValueError('Prepared rank files changed; preserve the deployment')
        return {'state': str(destination), 'status': 'prepared', 'reused': True}
    temporary = parent/('.'+request['id']+'-'+secrets.token_hex(8))
    temporary.mkdir(mode=0o700)
    try:
        seen = _rank_archive(payload, expected, temporary)
        if seen != set(expected):
            raise ValueError('Rank payload is incomplete')
        private_json(temporary/'stage-receipt.json', {'version': 1, 'files': expected})
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {'state': str(destination), 'status': 'prepared', 'reused': False}


def state_of(directory):
    path = owned_directory(directory)
    sys.path.insert(0, str(path))
    import engine_supervisor as supervisor
    live = path/'unit-live.json'
    if not live.exists():
        return {'status': 'stopped', 'all_children_reaped': True}
    recorded = json.loads(live.read_text())
    identity = recorded['supervisor']
    receipt = path/'supervisor.json'
    value = json.loads(receipt.read_text()) if receipt.is_file() else {}
    active = supervisor.identity(identity['pid']) == identity
    return {**value, 'supervisor': identity, 'supervisor_alive': active,
            'command_alive': bool(value.get('command') and supervisor.identity(value['command']['pid']) == value['command'])}


def start(request):
    path = owned_directory(request['state'])
    receipt = json.loads((path/'stage-receipt.json').read_text())
    for name, row in receipt['files'].items():
        p = member_path(path, name)
        info = p.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size != row['bytes'] or digest(p) != row['sha256']):
            raise ValueError('Prepared native rank changed; activation stopped')
    existing = state_of(path)
    if existing.get('supervisor_alive'):
        if existing.get('command_alive'):
            return existing
        raise ValueError('Owned rank is still starting or cleaning up')
    if (path/'unit-live.json').exists() and not existing.get('all_children_reaped'):
        raise ValueError('Prior rank cleanup is unverified')
    sys.path.insert(0, str(path))
    import engine_supervisor as supervisor
    unit = json.loads((path/'unit.json').read_text())
    if any(name in unit['env'] for name in ('HOME', 'home', 'CODEX_HOME')):
        raise ValueError('Rank environment cannot replace the user home')
    log = os.open(path/'engine.log', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        process = subprocess.Popen([sys.executable, str(path/'engine_supervisor.py'), '--state', str(path/'supervisor.json'), '--', *unit['argv']],
                                   cwd=path, env={**os.environ, **unit['env']}, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    finally:
        os.close(log)
    identity = supervisor.identity(process.pid, True)
    if identity is None:
        raise ValueError('Rank supervisor exited before its identity was recorded')
    private_json(path/'unit-live.json', {'supervisor': identity})
    _children[identity['pid']] = process
    return {'status': 'starting', 'supervisor': identity}


def stop(request):
    path = owned_directory(request['state'])
    sys.path.insert(0, str(path))
    import engine_supervisor as supervisor
    import signal
    state = state_of(path)
    if state.get('supervisor_alive'):
        supervisor.signal_identity(state['supervisor'], signal.SIGTERM)
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        state = state_of(path)
        if state.get('all_children_reaped') is True and not state.get('supervisor_alive'):
            process = _children.pop(state.get('supervisor', {}).get('pid'), None)
            if process is not None:
                process.wait(timeout=1)
            (path/'unit-live.json').unlink(missing_ok=True)
            return {'status': 'stopped', 'all_children_reaped': True}
        time.sleep(.05)
    return {'status': 'incomplete', 'reason': 'Rank cleanup is not confirmed; no unrelated process was signalled'}


def main():
    size = struct.unpack('>I', sys.stdin.buffer.read(4))[0]
    if size > 2*1024**2:
        raise ValueError('Rank request exceeds metadata limit')
    request = json.loads(sys.stdin.buffer.read(size))
    operation = request.pop('operation')
    if operation == 'inventory':
        result = inventory(request)
    elif operation == 'stage':
        result = stage(request, sys.stdin.buffer)
    elif operation == 'start':
        result = start(request)
    elif operation == 'status':
        result = state_of(request['state'])
    elif operation == 'stop':
        result = stop(request)
    else:
        raise ValueError('Unknown rank operation')
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error': str(error)}))
        raise SystemExit(1)
