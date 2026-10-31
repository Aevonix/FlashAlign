# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Default setup from local starter profiles and a compatible engine package."""
import os
from pathlib import Path

from .contacts import ApprovalStore
from .access_setup import choose_verified_contact
from .engine_setup import private_json
from .engine_assets import download_runtime_bundle
from .drafting import draft_settings, MissingDraftModel
from .model_metadata import inspect_model
from .profiles import bundled_profile_for, discover_profiles, read_json
from .setup_runtime import SetupError, check_harness_contexts, managed_context_window


def download_engine_bundle(home, model_type, *, platform_id=None):
    """Fetch a pinned public runtime; profiles stay in the local starter pack."""
    return download_runtime_bundle(home, model_type, platform_id=platform_id)


def backend_matches_checkpoint(metadata, model_id, model_path, quantization):
    """Choose an automatic backend from its declared checkpoint coverage."""
    from .backend_setup import _model_matches
    bindings = metadata.get('compatible_models')
    if bindings and not _model_matches(model_path, quantization, bindings):
        return False
    ids = metadata.get('qualified_model_ids') or [row.get('id') for row in metadata.get('tested_models', [])]
    return not ids or model_id in ids


def cached_models(home):
    cache = Path(os.environ.get('HF_HOME', str(Path(home)/'.cache/huggingface')))/'hub'
    found = []
    if cache.is_dir():
        for model in sorted(cache.glob('models--*'))[:128]:
            for config in sorted((model/'snapshots').glob('*/config.json'))[:8]:
                found.append(config.parent)
    return found


def make_default_plan(args, found, console):
    """Ask only for missing choices and write a private, reproducible setup plan."""
    console.banner('1.0.0')
    console.line('Default setup: detect a local model, prepare its engine and profile, then configure access.')
    console.step(1, 'Discover model and harness')
    direct = getattr(args, 'engine_only', False)
    selected = [] if direct else [name for name in ('hermes', 'openclaw') if found[name]['detected']]
    if not selected and not direct:
        raise SetupError('Install Hermes or OpenClaw before running default setup')
    console.context(harness=', '.join('Hermes' if name == 'hermes' else 'OpenClaw' for name in selected) or 'Direct engine')
    for name in selected:
        if not found[name]['runtime_compatible']:
            raise SetupError(f'{name} runtime is incompatible; update it before default setup')
        if name == 'openclaw' and found[name].get('session_scope_error'):
            raise SetupError(found[name]['session_scope_error'])
    model_path = args.model_path
    if model_path is None:
        models = cached_models(args.home)
        default = str(models[0]) if len(models) == 1 else ''
        if len(models) > 1:
            for index, path in enumerate(models, 1):
                console.line(f'  {index}. {path}')
        value = input('Local model folder'+(f' [{default}]' if default else '')+': ').strip() or default
        model_path = models[int(value)-1] if value.isdigit() and 0 < int(value) <= len(models) else Path(value).expanduser()
    remote_hosts = getattr(args, 'hosts', None)
    if remote_hosts:
        from .native_cluster import rpc
        remote_info = rpc(remote_hosts.split(',')[0].strip(), 'inventory', model_path=str(model_path), ssh_config=getattr(args, 'ssh_config', None))
        config = {**remote_info['config'], 'path': remote_info['model_path'], 'format': 'checkpoint'}
    else:
        config = inspect_model(model_path)
    console.step(2, 'Select engine and profile')
    profile_path, bundle_path = args.profile, args.engine_bundle
    backend = getattr(args, 'backend', 'auto')
    from .backend_setup import _model_matches, _select_kit, _bundled_draft_path, active_speculation, select_backend_python
    active = None if remote_hosts else active_speculation(model_path, '',
                                devices=args.devices.split(',') if args.devices else None)
    if backend == 'auto':
        backend = 'tensorfold' if remote_hosts else 'tensorfold' if bundle_path else active['kind'] if active else 'llamacpp' if config['format'] == 'GGUF' else None
    if not profile_path:
        starter = bundled_profile_for(config)
        if starter:
            profile_path = starter['path']
            console.line(f"Using included starter profile: {starter['name']} ({starter['status']}).")
    if not profile_path:
        roots = [*args.profiles, args.home/'.config/owner-mode/profiles']
        local, _ = discover_profiles(roots)
        if len(local) == 1:
            profile_path = local[0]['path']
        else:
            profile_path = input('Compatible profile folder: ').strip()
    local, errors = discover_profiles([Path(profile_path).expanduser()])
    if len(local) != 1:
        raise SetupError('A valid local profile is required for managed engine setup')
    profile = local[0]
    model_id = profile['model']['id']
    quantization = config.get('quantization_config') or (config.get('text_config') or {}).get('quantization_config')
    if config['format'] == 'checkpoint' and profile.get('compatible_models'):
        if remote_hosts:
            compatible = [row for row in profile['compatible_models'] if all(row.get(key) is None or row[key] == remote_info.get(key)
                          for key in ('config_sha256', 'index_sha256', 'quantization_config_sha256'))]
        else:
            compatible = _model_matches(Path(config['path']), quantization, profile['compatible_models'])
        if len(compatible) == 1:
            model_id = compatible[0]['id']
        elif len(compatible) > 1:
            raise SetupError('Several model identities match this checkpoint; choose one in a setup plan')
    backend_metadata = {}
    if backend is None:
        for candidate in ('vllm', 'sglang'):
            try:
                _, backend_metadata = _select_kit(candidate, config, profile)
                if not backend_matches_checkpoint(backend_metadata, model_id, Path(config['path']), quantization):
                    continue
                backend = candidate
                break
            except SetupError:
                continue
        backend = backend or 'tensorfold'
    elif backend != 'tensorfold':
        _, backend_metadata = _select_kit(backend, config, profile)
    if backend_metadata.get('requires_private_opt_in'):
        raise SetupError('This private unqualified engine requires an explicitly supplied setup plan with allow_research=true; default setup cannot opt in')
    if backend == 'tensorfold' and not bundle_path:
        console.line('Looking for a compatible engine package...')
        bundle_path = (download_engine_bundle(args.home, config.get('model_type'), platform_id=remote_info['platform'])
                       if remote_hosts else download_engine_bundle(args.home, config.get('model_type')))
    if backend == 'tensorfold' and not bundle_path:
        bundle_path = input('Aevonix TensorFold engine bundle folder: ').strip()
    if backend == 'tensorfold':
        backend_metadata = read_json(Path(bundle_path)/'manifest.json')
    python = select_backend_python({'python': args.engine_python} if args.engine_python else {}, backend_metadata) if backend != 'tensorfold' else {'executable': os.sys.executable}
    if backend != 'tensorfold':
        check_harness_contexts(selected, found, managed_context_window(backend, backend_metadata), managed=True)
    console.context(model=model_id, profile=profile['name'],
                    engine={'tensorfold': 'TensorFold', 'vllm': 'vLLM', 'sglang': 'SGLang', 'llamacpp': 'llama.cpp'}.get(backend, backend),
                    status='Selected; activation not checked yet')
    approvals = ApprovalStore(args.home/'.config/owner-mode/approvals.json')
    store = approvals.read()
    mode = getattr(args, 'mode', None) or store.get('mode', 'verified')
    console.step(3, 'Choose access')
    if (getattr(args, 'mode', None) is None and console.interactive
            and not getattr(args, 'yes', False)):
        index = console.select('Who gets the profile?', [
            ('Verified contacts', 'Approved contacts and authorized local sessions.'),
            ('Always On', 'Every request routed through FlashAlign, including tools and subagents.')],
            default=0 if mode == 'verified' else 1)
        mode = ('verified', 'always_on')[index]
    if mode not in ('verified', 'always_on'):
        raise SetupError('mode must be verified or always_on')
    console.context(access='Always On' if mode == 'always_on' else 'Verified contacts')
    contacts = [{key: row[key] for key in ('channel', 'account', 'contact')}
                for row in store['grants'] if row['enabled'] and approvals.channel_enabled(row['channel'])]
    if direct:
        contacts = []
    if not contacts and not direct and mode == 'verified':
        channel, account, contact = choose_verified_contact(console, selected, approvals)
        contacts = [{'channel': channel, 'contact': contact, 'account': account}]
    default_tools = 'on' if store['tools']['default'] else 'off'
    console.step(4, 'Set tool profile inheritance')
    if direct:
        tools = 'off'
    elif mode == 'always_on':
        tools = args.tools_default or default_tools
        console.line('Always On includes all routed tool and subagent requests. Verified-mode tool settings are preserved.')
    else:
        if args.tools_default:
            tools = args.tools_default
        elif console.interactive:
            index = console.select('Tools inherit the profile', [
                ('On', 'Apply the active profile to eligible tools and subagents.'),
                ('Off', 'Use the standard model for tool and subagent requests.')],
                default=0 if default_tools == 'on' else 1)
            tools = ('on', 'off')[index]
        else:
            tools = input(f'Tools default [on/off] [{default_tools}]: ').strip() or default_tools
    if tools not in ('on', 'off'):
        raise SetupError('Tools default must be on or off')
    console.context(tools='All routed calls' if mode == 'always_on' else tools.title())
    plan = {'harnesses': selected, 'model': model_id, 'profile': str(Path(profile_path).expanduser().resolve()),
            'contacts': contacts, 'tools_default': tools == 'on', 'mode': mode,
            'engine': {'kind': backend, 'model_path': str(Path(model_path).resolve()),
                       'python': python['executable'],
                       'install_dependencies': True}}
    if backend == 'tensorfold':
        plan['engine']['bundle'] = str(Path(bundle_path).expanduser().resolve())
        plan['engine']['install_dependencies'] = False
        for field in ('engram_source', 'engram_dir'):
            if getattr(args, field, None):
                plan['engine'][field] = str(getattr(args, field).expanduser())
        if getattr(args, 'hosts', None):
            plan['engine']['hosts'] = [{'host': host.strip(), 'model_path': str(model_path), 'devices': ['0']} for host in args.hosts.split(',')]
        if getattr(args, 'ssh_config', None):
            plan['engine']['ssh_config'] = str(args.ssh_config.expanduser().resolve())
        if getattr(args, 'master_address', None):
            plan['engine']['master_address'] = args.master_address
        if getattr(args, 'network_interface', None):
            plan['engine']['network_interface'] = args.network_interface
    if active and active['kind'] == backend:
        for key in ('draft_model', 'draft_algorithm', 'speculative_config', 'speculative_arguments'):
            if active.get(key) is not None:
                plan['engine'][key] = active[key]
    if direct:
        plan['engine_only'] = True
    if args.devices:
        plan['engine']['devices'] = args.devices.split(',')
    native_mtp = ('native_mtp_writer_groups' in backend_metadata.get('required_features', [])
                  or backend == 'tensorfold' and backend_metadata.get('default_speculative_method') in ('mtp', 'dspark', 'ngram'))
    if backend == 'sglang':
        methods = backend_metadata.get('speculative_algorithms', ['DFLASH'])
        selected_method = plan['engine'].get('draft_algorithm', methods[0])
        if not isinstance(selected_method, str):
            raise SetupError('A speculative algorithm must be a string')
        native_mtp = native_mtp or selected_method.upper() in backend_metadata.get('native_speculative_algorithms', [])
    if not native_mtp and not plan['engine'].get('draft_model') and backend_metadata.get('default_draft_subdirectory'):
        plan['engine']['draft_model'] = str(_bundled_draft_path(plan['engine']['model_path'], backend_metadata))
    if backend_metadata.get('requires_native_draft'):
        if getattr(args, 'drafter', None):
            raise SetupError('This model uses its native DSpark drafter; an external draft model is not qualified')
        default = dict(backend_metadata['default_speculative_config'])
        default['model'] = plan['engine']['model_path']
        plan['engine'].setdefault('speculative_config', default)
        console.line('Preserving native DSpark5. The first apply verifies local checkpoint bytes into private deployment state.')
        console.line('This experimental engine checks its GPU and host/cgroup memory policy before activation.')
    if getattr(args, 'drafter', None):
        plan['engine']['draft_model'] = str(args.drafter.expanduser().resolve())
    if native_mtp and backend != 'tensorfold':
        plan['engine'].setdefault('draft_model', plan['engine']['model_path'])
        plan['engine'].setdefault('draft_algorithm', backend_metadata['speculative_algorithms'][0])
    manifest = (read_json(Path(plan['engine']['bundle'])/'manifest.json') if backend == 'tensorfold'
                else {'requires_draft_model': backend == 'sglang' and not native_mtp, 'min_parallel': 1,
                      'draft_model_architectures': ['DFlash2DraftModel', 'DFlashDraftModel']})
    if manifest.get('requires_draft_model'):
        try:
            settings = draft_settings(args.home, plan['engine'], manifest, plan['model'])
        except MissingDraftModel:
            plan['engine']['draft_model'] = str(Path(input('Compatible draft model folder: ').strip()).expanduser().resolve())
            settings = draft_settings(args.home, plan['engine'], manifest, plan['model'])
        plan['engine'].update(settings)
        console.line('Preserving speculative decoding with the selected draft model.')
    path = args.home/'.config/owner-mode/default-setup.json'
    private_json(path, plan)
    console.step(5, 'Review setup')
    console.line(f"Engine: {backend} | Model: {plan['model']} | Profile: {profile['name']} | Access: {', '.join(selected) or 'direct engine'}")
    console.line('Mode: '+('Always On (all routed requests)' if mode == 'always_on' else 'Verified contacts'))
    if mode == 'always_on':
        console.line('The selected profile is required. Requests fail instead of falling back to stock.')
    if 'openclaw' in selected:
        scope = found['openclaw'].get('session_scope', {'selected': 'per-account-channel-peer'})
        console.line('OpenClaw direct-message session scope: '+scope['selected'])
    console.line('The engine will use a managed copy. Existing model files and unrelated services are left in place.')
    return path
