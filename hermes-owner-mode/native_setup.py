# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Prepare TensorFold binaries, profiles and rank commands without starting them."""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import re
import subprocess
from urllib.parse import urlsplit

from .auth import ensure_key, read_key
from .drafting import active_drafting, draft_settings
from .model_metadata import declared_openclaw_compat, declared_openclaw_request_policy, declared_request_compatibility
from .native_bundle import check_bundle, check_profile_contract, validate_manifest, host_platform, sha256
from . import native_cluster
from .native_remote import member_path, owned_directory, cuda_inventory, prepare_replay_directory
from .profiles import read_json, validate_profile, stage_profile, _compatible_models, ProfileError, NATIVE_FORMAT, profile_fingerprint
from .setup_runtime import SetupError


def checkpoint_matches(info, profile, manifest=None):
    try:
        candidates = (_compatible_models((manifest or {}).get('compatible_profile_bases'))
                      if profile.get('format') == NATIVE_FORMAT else
                      [profile['model'], *_compatible_models(profile.get('compatible_models', []))])
    except ProfileError as error:
        raise SetupError('Invalid compatible checkpoint metadata') from error
    pinned = [{k: row[k] for k in ('config_sha256', 'index_sha256', 'quantization_config_sha256') if row.get(k)}
              for row in candidates]
    pinned = [row for row in pinned if row]
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
           for row in pinned for value in row.values()):
        raise SetupError('Invalid checkpoint fingerprint in the selected profile')
    if profile.get('format') == NATIVE_FORMAT and not pinned:
        raise SetupError('Native matrix setup requires an explicit compatible serving base')
    if pinned and not any(all(info.get(k) == value for k, value in row.items()) for row in pinned):
        raise SetupError('Checkpoint metadata differs from the selected profile')


def _local_inventory(model_path, devices):
    path = Path(model_path).expanduser().resolve()
    config = read_json(path/'config.json')
    quant = config.get('quantization_config') or (config.get('text_config') or {}).get('quantization_config')
    index = path/'model.safetensors.index.json'
    architectures = {}
    try:
        free_devices, architectures = cuda_inventory()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        free_devices = []
    return {'platform': host_platform(), 'home': str(Path.home()), 'model_path': str(path), 'config': config,
            'cuda_architectures': architectures,
            'devices': devices if devices is not None else free_devices,
            'network_interfaces': sorted(p.name for p in Path('/sys/class/net').iterdir()),
            'config_sha256': sha256(path/'config.json'), 'index_sha256': sha256(index) if index.is_file() else None,
            'quantization_config_sha256': hashlib.sha256(json.dumps(quant, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


def nodes(spec):
    declared = spec.get('hosts')
    if declared is None:
        declared = [{'host': 'local', 'model_path': spec.get('model_path'), 'devices': spec.get('devices')}]
    if (not isinstance(declared, list) or not 1 <= len(declared) <= 16
            or any(not isinstance(row, dict) or set(row)-{'host', 'model_path', 'devices', 'home', 'draft_model', 'engram_source', 'engram_dir', 'state_root'} for row in declared)):
        raise SetupError('engine.hosts must list one to sixteen native model hosts')
    result = []
    for row in declared:
        if row.get('state_root') is not None and (not isinstance(row['state_root'], str) or not Path(row['state_root']).is_absolute()):
            raise SetupError('Native state_root must be an absolute owned directory')
        host = row.get('host')
        model = row.get('model_path', spec.get('model_path'))
        if not isinstance(host, str) or not host or not isinstance(model, str) or not model:
            raise SetupError('Each native host requires a host and model_path')
        devices = row.get('devices', spec.get('devices'))
        if devices is not None and (not isinstance(devices, list) or not devices or any(not str(x).isdigit() for x in devices)
                                    or len(set(map(str, devices))) != len(devices)):
            raise SetupError('Native device IDs must be a nonempty, unique integer list')
        info = _local_inventory(model, devices) if host == 'local' else native_cluster.rpc(host, 'inventory', model_path=model, ssh_config=spec.get('ssh_config'))
        devices = devices if devices is not None else info['devices']
        if not devices:
            raise SetupError(f'No free NVIDIA devices on {host}; select devices explicitly without stopping other services')
        for device in devices:
            result.append({**row, **info, 'host': host, 'device': str(device), 'rank': len(result), 'ssh_config': spec.get('ssh_config')})
    if len(result) > 16:
        raise SetupError('TensorFold supports at most sixteen ranks in this setup')
    return result


def speculation(home, spec, manifest, model_id, inventory):
    methods = manifest['speculative_methods']
    active = (active_drafting(inventory['model_path'], model_id) if inventory['host'] == 'local' else
              next((row for row in inventory.get('active_drafting', []) if not row.get('disabled')), None))
    requested = spec.get('draft_model')
    active_path = active.get('draft_model') if active and not active.get('disabled') else None
    if active_path and requested and Path(requested) != Path(active_path):
        raise SetupError('The selected draft differs from the active native engine; its speculative setup must be preserved')
    method = 'dflash' if active_path or requested else manifest.get('default_speculative_method', methods[0])
    if method not in methods:
        raise SetupError('This native bundle cannot preserve the active speculative method')
    if method == 'dflash':
        if inventory['host'] != 'local':
            architectures = manifest.get('draft_model_architectures', ['DFlash2DraftModel', 'DFlashDraftModel'])
            found = native_cluster.rpc(inventory['host'], 'inventory', model_path=inventory['model_path'],
                                       draft_model=requested or active_path, draft_architectures=architectures,
                                       ssh_config=inventory.get('ssh_config'))
            if not found.get('draft_model') or not set(found.get('draft_config', {}).get('architectures', [])) & set(architectures):
                raise SetupError('A compatible DFlash model is required on the native model host')
            return {'method': method, 'draft_model': found['draft_model'],
                    **({'parallel': int(active['parallel'])} if active and active.get('parallel') else {})}
        adjusted = {**spec, 'model_path': inventory['model_path'], 'draft_model': requested or active_path}
        metadata = {**manifest, 'requires_draft_model': True,
                    'draft_model_architectures': manifest.get('draft_model_architectures', ['DFlash2DraftModel', 'DFlashDraftModel']),
                    'min_parallel': 1}
        result = draft_settings(home, adjusted, metadata, model_id)
        return {**result, 'method': method}
    if requested:
        raise SetupError('This native model uses its integrated drafter; an external draft is not supported')
    result = {'method': method}
    if method == 'mtp':
        depth = spec.get('mtp_drafts', active.get('mtp_drafts') if active else None)
        depth = int(depth) if depth is not None else manifest.get('default_mtp_drafts', 4)
        if type(depth) is not int or not 1 <= depth <= 16:
            raise SetupError('Native MTP depth must be between one and sixteen')
        if manifest.get('mtp_drafts_tunable', True) is False and depth != manifest.get('default_mtp_drafts', 3):
            raise SetupError('This native family has a fixed MTP depth; the selected setting cannot be preserved')
        result['mtp_drafts'] = depth
        result['tunable'] = manifest.get('mtp_drafts_tunable', True)
    if active and active.get('parallel'):
        result['parallel'] = int(active['parallel'])
    return result


def check(home, spec, profile, *, model_id):
    from .engine_setup import tensorfold_arguments, declared_input_modalities
    if spec.get('source') or spec.get('command'):
        raise SetupError('TensorFold setup accepts a verified binary bundle, not a Python source or custom command')
    if profile.get('format') == NATIVE_FORMAT:
        try:
            manifest = validate_manifest(read_json(Path(spec.get('bundle', ''))/'manifest.json'))
            check_profile_contract(manifest, profile)
        except (OSError, ValueError) as error:
            if isinstance(error, SetupError):raise
            raise SetupError('Native matrix profiles require an explicit compatible TensorFold bundle') from error
    if spec.get('api_transport', 'ssh') not in ('ssh', 'direct'):
        raise SetupError('Native api_transport must be ssh or direct')
    hosts = nodes(spec)
    first = hosts[0]
    if any(row['platform'] != first['platform'] for row in hosts):
        raise SetupError('All ranks in one native deployment must use the same bundle architecture')
    root, manifest = check_bundle(spec.get('bundle', ''), first['config'], profile, platform_id=first['platform'])
    environment = checked_environment(spec, manifest)
    startup = startup_timeout(spec, manifest)
    hardware_verified = True
    for node in hosts:
        if spec.get('network_interface') and spec['network_interface'] not in node.get('network_interfaces', []):
            raise SetupError(f"Native network interface is unavailable on {node['host']}")
        capability = node.get('cuda_architectures', {}).get(node['device'])
        if capability is None:
            hardware_verified = False
        elif not architecture_matches(capability, manifest['cuda_architectures']):
            raise SetupError(f"Native runtime CUDA architectures do not support {node['host']} device {node['device']} ({capability})")
    arguments = tensorfold_arguments(spec, manifest)
    if spec.get('hosts') is None and spec.get('devices') is None:
        count = max((n for n in manifest['tensor_parallel_sizes'] if n <= len(hosts)), default=0)
        hosts = hosts[:count]
        if not hosts:
            raise SetupError('No available device count matches the native bundle')
    world = len(hosts)
    if world not in manifest['tensor_parallel_sizes']:
        raise SetupError('This native bundle has not qualified the requested rank count')
    context = None
    for i, item in enumerate(arguments):
        if item == '--context' and i+1 < len(arguments):
            context = int(arguments[i+1])
        elif item.startswith('--context='):
            context = int(item.split('=', 1)[1])
    if context is None or context < 1:
        raise SetupError('Native bundles must declare their tested context size')
    for row in hosts:
        checkpoint_matches(row, profile, manifest)
        if (row['config'].get('model_type') != first['config'].get('model_type')
                or row['config_sha256'] != first['config_sha256'] or row['index_sha256'] != first['index_sha256']):
            raise SetupError('Native ranks must use matching checkpoint metadata')
        text = row['config'].get('text_config') or row['config']
        if text.get('hidden_size') != read_json(profile['manifest'])['hidden_size']:
            raise SetupError('Profile width does not match the native checkpoint')
    drafting = speculation(home, spec, manifest, model_id, first)
    shape = manifest.get('asset_geometry')
    if shape is not None and (not isinstance(shape, dict) or shape.get('context') != context or shape.get('tensor_parallel') != world):
        raise SetupError('Native generated assets do not match this context and rank count')
    if manifest['rank_transport'] == 'dsv41_env':
        contract = manifest.get('auxiliary_engram')
        if not isinstance(contract, dict) or not isinstance(contract.get('source'), dict):
            raise SetupError('DeepSeek native bundle must declare its pinned public Engram source')
        for node in hosts:
            request = {'model_path': node['model_path'], 'engram_contract': contract,
                       'engram_source': node.get('engram_source', spec.get('engram_source')),
                       'engram_dir': node.get('engram_dir', spec.get('engram_dir')),
                       'tp_rank': node['rank'], 'tp_world': world}
            if node['host'] == 'local':
                from .native_remote import inspect_engram
                try: receipt = inspect_engram(request, node['config'])
                except (OSError, ValueError, KeyError) as error: raise SetupError(str(error)) from error
            else:
                receipt = native_cluster.rpc(node['host'], 'inventory', ssh_config=spec.get('ssh_config'), **request)['engram']
            node['engram_source' if receipt['kind'] == 'safetensors' else 'engram_dir'] = receipt['path']
            node['engram_validation'] = receipt
        for key, field in [('TF_DSV41_CONTEXT','context'),('TF_DSV41_POOL_TOKENS','pool_tokens'),('TF_DSV41_SLOTS','slots')]:
            if str(environment.get(key)) != str(shape.get(field)):
                raise SetupError('DeepSeek runtime environment differs from its validated asset geometry')
    kind = first['config']['model_type']
    result = {'kind': 'tensorfold', 'runtime': 'native', 'runtime_version': manifest['runtime_version'],
              'source_revision': manifest['source_revision'], 'platform': first['platform'], 'hosts': hosts,
              'hardware_verified': hardware_verified,
              'bundle': str(root), 'manifest': manifest, 'arguments': arguments, 'drafting': drafting,
              'startup_timeout': startup,
              'context_window': {'tokens': context, 'source': 'TensorFold --context'},
              'openclaw_compat': declared_openclaw_compat(manifest, kind),
              'request_compatibility': declared_request_compatibility(manifest, kind),
              'openclaw_request_policy': declared_openclaw_request_policy(manifest, kind)}
    modalities = declared_input_modalities(manifest)
    if modalities is not None:
        result['input_modalities'] = modalities
    return result


def architecture_matches(capability, targets):
    actual = int(capability.removeprefix('sm_'))
    return any((target.startswith('sm_') or target.endswith('a')) and actual == int(re.sub(r'\D', '', target))
               or target.startswith('compute_') and not target.endswith('a') and actual >= int(target[8:])
               for target in targets)


def checked_environment(spec, manifest):
    if not isinstance(manifest.get('environment', {}), dict) or not isinstance(spec.get('environment', {}), dict):
        raise SetupError('Native runtime environment must be an object')
    env = {**manifest.get('environment', {}), **spec.get('environment', {})}
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()) or any(k in env for k in ('HOME', 'home', 'CODEX_HOME')):
        raise SetupError('Native runtime environment must contain safe string values')
    if any(k in env and env[k] == '0' for k in ('TF_DSV41_DRAFTS', 'TF_DSV41_GRAPHS')):
        raise SetupError('Native setup cannot disable speculative decoding or existing CUDA graphs')
    return env


def startup_timeout(spec, manifest):
    value = spec.get('startup_timeout', manifest.get('startup_timeout', 1800))
    if type(value) is not int or not 60 <= value <= 7200:
        raise SetupError('Native startup_timeout must be an integer from 60 to 7200 seconds')
    return value


def command(runtime, profile, key, node, checked, spec, model_id, port, master, master_port, *, replay_directory=None):
    manifest, draft = checked['manifest'], checked['drafting']
    world, rank = len(checked['hosts']), node['rank']
    argv = [str(runtime/manifest['binary']), 'serve', node['model_path'], '--name', model_id,
            '--host', spec.get('host', '127.0.0.1' if node['host'] == 'local' or spec.get('api_transport', 'ssh') == 'ssh' else '0.0.0.0'), '--port', str(port),
            *checked['arguments'], '--owner-profile', str(profile)]
    env = checked_environment(spec, manifest)
    env['LD_LIBRARY_PATH'] = str(runtime/'lib') + (os.pathsep+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['CUDA_VISIBLE_DEVICES'] = node['device']
    if spec.get('network_interface'):
        env['NCCL_SOCKET_IFNAME'] = spec['network_interface']
    transport = manifest['rank_transport']
    if transport == 'glm_env':
        env.update(TF_GLM_RANK=str(rank), TF_GLM_WORLD=str(world), NCCL_COMM_ID=f'{master}:{master_port}')
    elif transport == 'cli':
        argv += ['--tp', str(world), '--rank', str(rank)]
        if world > 1:
            argv += ['--master', master, '--master-port', str(master_port)]
    else:
        assets = runtime/manifest['assets_directory']
        env.update(TF_TP_RANK=str(rank), TF_TP_WORLD=str(world), TF_TP_DEVICE='0',
                   TF_TP_MASTER=master, TF_TP_PORT=str(master_port), TF_DSV41_ASSETS=str(assets),
                   TF_TP_PLAN_TIMEOUT=str(startup_timeout(spec, manifest)))
        source = node.get('engram_source', spec.get('engram_source'))
        directory = node.get('engram_dir', spec.get('engram_dir'))
        if bool(source) == bool(directory):
            raise SetupError('DeepSeek requires either engram_source or engram_dir on every native host')
        env['TF_DSV41_ENGRAM_SAFETENSORS' if source else 'TF_DSV41_ENGRAM_DIR'] = str(source or directory)
        if rank:
            argv = [str(runtime/manifest['follower_binary']), 'follow', node['model_path'], str(assets), '--owner-profile', str(profile)]
    if rank == 0 or transport == 'glm_env':
        argv += ['--owner-key-file', str(key)]
    if rank == 0 and manifest.get('durable_owner_replay', False):
        if replay_directory is None:
            raise SetupError('Durable native owner mode requires a persistent replay directory')
        argv += ['--owner-replay-dir', str(replay_directory)]
    if draft.get('draft_model'):
        argv += ['--drafter', node.get('draft_model', draft['draft_model'])]
    if 'mtp_drafts' in draft and draft.get('tunable', True):
        argv += ['--mtp-drafts', str(draft['mtp_drafts'])]
    if draft.get('parallel') and '--parallel' in argv:
        argv[argv.index('--parallel')+1] = str(draft['parallel'])
    return argv, env


def owner_replay_directory(home, node, key_material):
    """Same key on the same host keeps one journal across runtime/profile upgrades."""
    namespace = hashlib.sha256(b'flashalign-native-owner-replay-v1\0'+key_material).hexdigest()
    if node.get('state_root'):
        parent = Path(node['state_root'])/'native-replay'
    else:
        owner_home = Path(home) if node['host'] == 'local' else Path(node['home'])
        parent = owner_home/'.config/owner-mode/native-replay'
    if not parent.is_absolute():
        raise SetupError('Native owner replay parent must be an absolute private directory')
    return parent/namespace


def reject_untracked_replay_key(home, key_material):
    """An old in-memory gate cannot supply its consumed nonces to a new journal."""
    parent = Path(home)/'.config/owner-mode/engines'
    for path in parent.glob('*/deployment.json'):
        if path.parent.is_symlink():
            continue
        try:
            prior = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(prior, dict):
            continue
        replay = prior.get('owner_replay')
        if (prior.get('runtime') != 'native' or
                isinstance(replay, dict) and replay.get('durable') is True):
            continue
        candidates = [prior.get('key_file')]
        for command_row in prior.get('commands', []):
            if not isinstance(command_row, dict):
                continue
            argv = command_row.get('argv', [])
            if not isinstance(argv, list):
                continue
            if '--owner-key-file' in argv and argv.index('--owner-key-file')+1 < len(argv):
                candidates.append(argv[argv.index('--owner-key-file')+1])
        for name in candidates:
            if not isinstance(name, str):
                continue
            try:
                previous = read_key(name)
            except (OSError, ValueError):
                continue
            if previous == key_material:
                raise SetupError('This signing key was used by a native deployment without durable replay protection; create a fresh signing-key file before upgrading. The supplied key was not changed.')


def prepare(home, spec, profile_path, *, model_id, key_file=None):
    from .engine_setup import private_json, unused_port
    profile = validate_profile(profile_path, allow_missing_payload=True)
    checked = check(home, spec, profile, model_id=model_id)
    from .profile_assets import materialize_profile
    ready = materialize_profile(profile_path, cache_root=Path(home)/'.cache/flashalign/profiles')
    if ready['hashes'] != profile['hashes']:
        raise SetupError('Profile changed between compatibility validation and preparation')
    profile = ready
    bundle, manifest = Path(checked['bundle']), checked['manifest']
    if key_file is not None and manifest.get('durable_owner_replay', False):
        supplied = Path(key_file).expanduser()
        if supplied.exists():
            reject_untracked_replay_key(home, read_key(supplied))
    identity = hashlib.sha256((sha256(bundle/'manifest.json')+profile_fingerprint(profile)+
                              model_id+json.dumps(spec, sort_keys=True)).encode()).hexdigest()[:20]
    state = Path(home)/'.config/owner-mode/engines'/identity
    try:
        owned_directory(state, create=True)
    except ValueError as error:
        raise SetupError(str(error)) from error
    key = Path(key_file).expanduser().resolve() if key_file else state/'engine.key'
    ensure_key(key)
    key_material = read_key(key)
    staged_profile = state/'profile'
    stage_profile(profile['path'], staged_profile)
    hosts = checked['hosts']
    remote = any(row['host'] != 'local' for row in hosts)
    previous = read_json(state/'deployment.json') if (state/'deployment.json').is_file() else {}
    network_file = state/'network.json'
    network = read_json(network_file) if network_file.is_file() else {}
    port = spec.get('port') or urlsplit(previous.get('base_url', '')).port or network.get('port') or unused_port()
    master_port = spec.get('master_port') or previous.get('master_port') or network.get('master_port') or unused_port()
    if any(type(n) is not int or not 1 <= n <= 65535 for n in (port, master_port)):
        raise SetupError('Native API and rendezvous ports must be valid TCP ports')
    forward_port = (network.get('forward_port') or unused_port()) if remote and hosts[0]['host'] != 'local' and spec.get('api_transport', 'ssh') == 'ssh' else None
    private_json(network_file, {'port': port, 'master_port': master_port, 'forward_port': forward_port})
    master = spec.get('master_address', '127.0.0.1' if not remote else hosts[0]['host'].split('@')[-1])
    if remote:
        try:
            if ipaddress.ip_address(socket.gethostbyname(master)).is_loopback:
                raise ValueError()
        except (OSError, ValueError) as error:
            raise SetupError('Provide a reachable IPv4 master_address for a multi-host native deployment') from error
    commands, units = [], []
    replay_directory = None
    supervisor = Path(__file__).with_name('engine_supervisor.py').read_bytes()
    for node in hosts:
        remote_state = (state/('rank-'+str(node['rank'])) if node['host'] == 'local' else
                        Path(node.get('home', node['home']))/'.config/owner-mode/native-ranks'/(identity+'r'+str(node['rank'])))
        if remote:
            # A rank ID has a fixed width for the remote helper's owned-directory validation.
            rank_id = hashlib.sha256((identity+str(node['rank'])).encode()).hexdigest()[:20]
            parent = (state/'ranks' if node['host'] == 'local' else Path(node.get('state_root') or str(Path(node['home'])/'.config/owner-mode/native-ranks')))
            remote_state = parent/rank_id
        runtime = remote_state/'runtime'
        replay = (owner_replay_directory(home, node, key_material)
                  if node['rank'] == 0 and manifest.get('durable_owner_replay', False) else None)
        if replay is not None:
            replay_directory = replay
        shared_profile = profile.get('format') == NATIVE_FORMAT and node['host'] == 'local'
        argv, environment = command(runtime, staged_profile if shared_profile else remote_state/'profile', remote_state/'engine.key', node, checked, spec,
                                    model_id, port, master, master_port, replay_directory=replay)
        files = {'runtime/manifest.json': (bundle/'manifest.json', False),
                 'engine_supervisor.py': (supervisor, False)}
        if '--owner-key-file' in argv:
            files['engine.key'] = (key_material, False)
        for name, info in manifest['files'].items():
            files['runtime/'+name] = (bundle/name, info.get('executable', False))
        if not shared_profile:
            for path in staged_profile.iterdir():
                if path.is_file():
                    files['profile/'+path.name] = (path, False)
        files['unit.json'] = ((json.dumps({'argv': argv, 'env': environment})+'\n').encode(), False)
        if remote:
            native_cluster.stage_files(node['host'], files,
                               parent=str(parent), id=rank_id,
                               **({'owner_replay_dir': str(replay)} if replay is not None else {}),
                               ssh_config=spec.get('ssh_config'))
            units.append({'host': node['host'], 'state': str(remote_state), 'rank': node['rank'], 'ssh_config': spec.get('ssh_config')})
        else:
            try:
                owned_directory(remote_state, create=True)
                if replay is not None:
                    prepare_replay_directory(str(replay))
            except ValueError as error:
                raise SetupError(str(error)) from error
            for name, (data, executable) in files.items():
                try:
                    path = member_path(remote_state, name)
                except ValueError as error:
                    raise SetupError(str(error)) from error
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if path.is_symlink():
                    raise SetupError('Prepared native rank contains a symbolic link')
                with native_cluster.source_file(data) as (source, size):
                    if path.exists():
                        expected = hashlib.sha256()
                        for chunk in iter(lambda: source.read(8*1024*1024), b''):
                            expected.update(chunk)
                        if path.stat().st_size != size or sha256(path) != expected.hexdigest():
                            raise SetupError('Prepared native rank files changed; preserve the deployment and use a new plan')
                    else:
                        with path.open('xb') as output:
                            shutil.copyfileobj(source, output, length=8*1024*1024)
                path.chmod(0o700 if executable else 0o600)
            commands.append({'argv': argv, 'env': environment})
    leader = '127.0.0.1' if hosts[0]['host'] == 'local' else master
    remote_url = f'http://{leader}:{port}/v1'
    controller_units = []
    if forward_port is not None:
        from .native_cluster import ssh_options
        tunnel_id = hashlib.sha256((identity+'tunnel').encode()).hexdigest()[:20]
        parent = state/'tunnels'
        argv = ['ssh', *ssh_options(spec.get('ssh_config')), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ForwardAgent=no', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3', '-N', '-L', f'127.0.0.1:{forward_port}:127.0.0.1:{port}', hosts[0]['host']]
        files = {'engine_supervisor.py': (supervisor, False),
                 'unit.json': ((json.dumps({'argv': argv, 'env': {}})+'\n').encode(), False)}
        records = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'executable': exe}
                   for name, (data, exe) in files.items()}
        native_cluster.rpc('local', 'stage', payload=native_cluster.archive(files), parent=str(parent), id=tunnel_id, files=records)
        controller_units.append({'host': 'local', 'state': str(parent/tunnel_id), 'kind': 'ssh_forward'})
    result = {'version': 2, 'kind': 'tensorfold', 'runtime': 'native', 'state': str(state), 'source': str(state),
              'commands': commands, 'base_url': f'http://127.0.0.1:{forward_port}/v1' if forward_port else remote_url, 'model': model_id,
              'profile': profile['name'], 'profile_path': str(staged_profile), 'key_file': str(key),
              'runtime_version': manifest['runtime_version'], 'source_revision': manifest['source_revision'],
              'master_port': master_port,
              'bundle_sha256': sha256(bundle/'manifest.json'), 'drafting': checked['drafting'],
              'hardware_verified': checked['hardware_verified'], 'startup_timeout': checked['startup_timeout'],
              'status': 'prepared', 'install_dependencies': False, 'requirements': [], 'python': os.sys.executable}
    if replay_directory is not None:
        result['owner_replay'] = {'directory': str(replay_directory), 'durable': True}
    if remote:
        result['remote_units'] = units
        result['controller_units'] = controller_units
        result['remote_base_url'] = remote_url
    private_json(state/'deployment.json', result)
    return result
