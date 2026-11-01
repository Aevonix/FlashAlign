# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Set up Aevonix FlashAlign for direct engine access, Hermes or OpenClaw."""
import argparse
import getpass
import hashlib
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owner_mode', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from owner_mode.contacts import ApprovalStore
from owner_mode.access_setup import choose_verified_contact, terminal_harnesses
from owner_mode.auth import read_key, ensure_key
from owner_mode.profiles import discover_profiles, match_profile, validate_profile, stage_profile
from owner_mode.profiles import resolve_profile, read_json
from owner_mode.automation import run_plan
from owner_mode.quickstart import make_default_plan
from owner_mode.engine_setup import stop_engine
from owner_mode.model_metadata import (configure_hermes_reasoning, configure_hermes_images,
                                       configure_hermes_provider, configure_openclaw_compat,
                                       checked_request_compatibility, checked_openclaw_request_policy,
                                       check_openclaw_request_config)
from owner_mode.setup_runtime import Console, SetupError, check_engine, engine_profiles, model_rows, restart_harnesses, register_openclaw, openclaw_session_scope, harness_context_requirements, check_harness_contexts, HERMES_MINIMUM_CONTEXT
from owner_mode.setup_transaction import current_transaction, pending_setup, recover_setup, transactional_setup

MODULES = ['__init__.py','auth.py','identity.py','hermes_ingress.py','native_authority.py',
           'content.py','hermes_observer.py',
           'contacts.py','channels.py','access_setup.py','terminal.py','native_terminal.py','policy.py','proxy.py','plugin.py','bridge.py','serve.py','manage.py','setup.py',
           'profiles.py','deepseek_grouped.py','native_profile.py','profile_assets.py','profile_inspector.py','setup_runtime.py','terminal_ui.py','terminal-logo.json','setup_transaction.py','automation.py','direct_gateway.py','engine_setup.py','engine_supervisor.py','engine_assets.py','native_bundle.py','native_setup.py','native_cluster.py','native_remote.py','engine_catalog.json','backend_setup.py','model_metadata.py','drafting.py','quickstart.py','client.py','harness_client.py','plugin.yaml','requirements.txt','README.md','LICENSE','NOTICE.md']


def load_config(path):
    if not path.exists():
        return {}
    text = path.read_text()
    try:
        value = json.loads(text)
    except ValueError:
        try:
            if path.suffix == '.json':
                import json5
                value = json5.loads(text)
            else:
                import yaml
                value = yaml.safe_load(text)
        except ImportError as error:
            raise ValueError('Run the setup launcher to install configuration readers') from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError('Harness configuration must be an object')
    return value


def discover(home, which=shutil.which, probe=None):
    found = {}
    for name, relative in [('hermes','.hermes/config.yaml'),('openclaw','.openclaw/openclaw.json')]:
        executable = which(name)
        path = home/relative
        config = load_config(path)
        urls = []
        if name == 'hermes':
            value = config.get('model',{}).get('base_url')
            owner = config.get('plugins',{}).get('entries',{}).get('owner-mode',{}).get('settings',{})
            if owner.get('routes',{}).get('head',{}).get('upstream'):
                value = owner['routes']['head']['upstream']
            if isinstance(value,str): urls.append(value)
        else:
            for provider_name, provider in config.get('models',{}).get('providers',{}).items():
                if provider_name == 'owner-mode':
                    bridge_path = config.get('plugins',{}).get('entries',{}).get('owner-mode',{}).get('config',{}).get('bridgeConfig')
                    if bridge_path:
                        upstream = load_config(Path(bridge_path)).get('routes',{}).get('head',{}).get('upstream')
                        if isinstance(upstream,str): urls.append(upstream)
                    continue
                if isinstance(provider,dict) and isinstance(provider.get('baseUrl'),str): urls.append(provider['baseUrl'])
        found[name] = {'detected':bool(executable or path.exists()), 'executable':executable,
                       'config':str(path), 'model_urls':list(dict.fromkeys(urls))}
        try:
            found[name]['context_requirements'] = harness_context_requirements(name, config)
        except SetupError as error:
            found[name]['context_requirements_error'] = str(error)
        if name == 'openclaw':
            try:
                found[name]['session_scope'] = openclaw_session_scope(config)
            except SetupError as error:
                found[name]['session_scope_error'] = str(error)
        found[name]['models'] = []
        if probe:
            for url in found[name]['model_urls']:
                try:
                    for model in probe(url):
                        found[name]['models'].append({'id':model,'profile':profile_for(model),
                            'compatibility':'known profile family; engine support unverified' if profile_for(model)
                                            else 'no supported profile found'})
                except Exception:
                    found[name].setdefault('unreachable_model_urls',[]).append(url)
    node=which('node')
    version='unknown'
    if node:
        try:version=subprocess.run([node,'--version'],capture_output=True,text=True,timeout=5).stdout.strip()
        except Exception:pass
    parts=version.lstrip('v').split('.')
    supported=False
    try:
        numeric=tuple(int(p) for p in parts[:3])
        supported=(numeric[0]==24 and numeric>=(24,22,1)) or (numeric[0]==26 and numeric>=(26,12,1))
    except (ValueError,IndexError):pass
    found['openclaw']['node_version']=version
    found['openclaw']['runtime_compatible']=supported
    found['hermes']['runtime_compatible']=sys.version_info >= (3,10)
    found['hermes']['supported_minimum_context_length']=HERMES_MINIMUM_CONTEXT
    return found


def model_list(base_url, timeout=5):
    return [row['id'] for row in model_rows(base_url)]


def profile_for(model, catalog=None):
    if catalog is None:
        try:
            catalog = read_json(ROOT.parent/'profiles/catalog.json')
        except (OSError, ValueError):
            return None
    matches = {item['profile'] for item in catalog.get('models', [])
               if isinstance(item, dict) and isinstance(item.get('profile'), str)
               and any(isinstance(alias, str) and alias.casefold() == model.casefold()
                       for alias in item.get('model_ids', []))}
    return next(iter(matches)) if len(matches) == 1 else None


def free_port(preferred):
    with socket.socket() as sock:
        try:sock.bind(('127.0.0.1',preferred))
        except OSError:sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def save_config(path, config):
    transaction = current_transaction()
    if transaction is not None:path = transaction.stage_file(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copy2(path, path.with_name(path.name+'.before-owner-mode-'+stamp))
    temp=path.with_name(path.name+'.owner-mode-new-'+secrets.token_hex(6))
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as stream:json.dump(config,stream,indent=2);stream.write('\n')
    os.replace(temp,path)


def init_signing_key(path):
    """Create once; reruns never rotate an existing deployment's key."""
    path = Path(path).expanduser()
    created = ensure_key(path)
    return {'key_file': str(path), 'status': 'created' if created else 'reused'}


@transactional_setup
def provision(home, selected, base_url, model, profile, contacts, tools_default, engine_key=None,
              *, profile_details=None, engine_check=None, model_info=None, api_key=None, mode='verified'):
    endpoint = urlsplit(base_url)
    if endpoint.scheme not in {'http','https'} or not endpoint.hostname or endpoint.username or endpoint.password:
        raise ValueError('Enter a model base URL without credentials')
    if not model or not profile or type(tools_default) is not bool:
        raise ValueError('A model, installed profile name and tool default are required')
    if mode not in ('verified', 'always_on'):
        raise ValueError('mode must be verified or always_on')
    state=home/'.config/owner-mode'
    state.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Parse every existing config before writing any of them.
    configs={name:load_config(home/('.hermes/config.yaml' if name=='hermes' else '.openclaw/openclaw.json'))
             for name in selected}
    request_policy = checked_openclaw_request_policy((model_info or {}).get('openclaw_request_policy'))
    if 'openclaw' in selected:
        check_openclaw_request_config(configs['openclaw'], model, request_policy)
    scope = openclaw_session_scope(configs['openclaw']) if 'openclaw' in selected else None
    key=Path(engine_key).expanduser() if engine_key else state/'engine.key'
    if engine_key and not key.is_file():
        raise ValueError('The existing engine signing-key file was not found')
    if not key.exists():
        init_signing_key(key)
    read_key(key)
    store=ApprovalStore(state/'approvals.json')
    pending_store=ApprovalStore(current_transaction().stage_file(store.path))
    def defaults(value):value['tools']['default']=tools_default
    pending_store.update(defaults)
    pending_store.set_mode(mode)
    pending_store.set_channels(sorted(set(store.channels()) | {channel for channel, _, _ in contacts}))
    for channel,account,contact in contacts:pending_store.set_contact(channel,account,contact,True)
    ports={'hermes':free_port(18034),'openclaw':free_port(18035)}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    release=state/'releases'/stamp
    release.mkdir(parents=True)
    for filename in MODULES:shutil.copy2(ROOT/filename,release/filename)
    route={'upstream':base_url,'profile':profile,'owner_mode_scale':1.0,'timeout':1800}
    if model == 'deepseek-ai/DeepSeek-V4.1-Flash-1017':
        route['request_compatibility'] = 'deepseek_v41'
    if (model_info or {}).get('request_compatibility') is not None:
        if 'request_compatibility' in route:
            raise SetupError('Conflicting model request compatibility declarations')
        declaration = checked_request_compatibility(model_info['request_compatibility'], require_model=False)
        route['request_compatibility'] = checked_request_compatibility({**declaration, 'model': model})
    if api_key:
        if '\r' in api_key or '\n' in api_key or len(api_key) > 8192:
            raise ValueError('Invalid upstream API key')
        secret = state/'upstream-api.key'
        temporary = current_transaction().stage_file(secret)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(api_key)
        temporary.chmod(0o600)
        route['api_key_file'] = str(secret)
    result={'harnesses':selected,'model':model,'profile':profile,'mode':mode,
            'enabled':pending_store.read()['enabled'],'approval_store':str(store.path),
            'engine_key_file':str(key),'engine_compatibility':'unverified until the engine reports the effective profile',
            'profile_installation':'Load a compatible Owner Mode profile in the selected engine', 'configs':[],
            'status':'awaiting_restart','applied':True,
            'recovery_directory':str(current_transaction().directory)}
    local_harnesses = sorted(set(selected) & terminal_harnesses(contacts))
    if local_harnesses:
        result['terminal_harnesses'] = local_harnesses
        result['terminal_commands'] = [['flashalign-client','terminal','--home',str(home),
                                         '--harness',name,'--authorize'] for name in local_harnesses]
    if route.get('request_compatibility'):
        result['model_request_compatibility'] = (
            'DeepSeek disabled reasoning uses enable_thinking=false before signing'
            if route['request_compatibility'] == 'deepseek_v41'
            else 'Declared thinking controls are preserved for the selected model')
    if (profile_details and profile_details.get('source') == 'local'
            and profile_details.get('artifact_valid') is True):
        checked = validate_profile(profile_details['manifest'])
        if checked['hashes'] != profile_details['hashes']:
            raise ValueError('The selected profile changed during setup; run setup again')
        identity = hashlib.sha256(''.join(checked['hashes'].values()).encode()).hexdigest()[:20]
        staged = state/'profiles'/identity
        stage_profile(checked['path'], staged)
        result['profile_path'] = str(staged)
        result['profile_hashes'] = checked['hashes']
        result['profile_artifact'] = 'validated locally; model quality is not certified'
    if engine_check:
        result['engine_compatibility'] = 'verified: engine confirmed standard mode and selected profile'
        result['engine_check'] = engine_check
        result['profile_installation'] = 'Selected profile is already loaded by the engine'
    if 'hermes' in selected:
        target=current_transaction().stage_tree(home/'.hermes/plugins/owner-mode')
        for filename in MODULES:shutil.copy2(release/filename,target/filename)
        cfg=configs['hermes']
        reasoning = configure_hermes_reasoning(cfg, model)
        if reasoning is not None:
            result['hermes_reasoning'] = reasoning
        result['hermes_images'] = configure_hermes_images(cfg, (model_info or {}).get('input'))
        result['hermes_provider'] = configure_hermes_provider(cfg, model,
            f"http://127.0.0.1:{ports['hermes']}/head/v1", (model_info or {}).get('input'))
        plugins=cfg.setdefault('plugins',{})
        enabled=plugins.setdefault('enabled',[])
        if not isinstance(enabled,list):
            raise ValueError('Hermes plugins.enabled must be an array')
        if 'owner-mode' not in enabled:
            enabled.append('owner-mode')
        if isinstance(plugins.get('disabled'),list):
            plugins['disabled']=[name for name in plugins['disabled'] if name!='owner-mode']
        cfg.setdefault('plugins',{}).setdefault('entries',{})['owner-mode']={'enabled':True,'settings':{
            'serve_argv':['gateway','run'],'port':ports['hermes'],'approvals_file':str(store.path),
            'managed_provider':result['hermes_provider']['name'],
            'key_file':str(key),'allowed_channels':sorted(pending_store.channels()),
            'trust':{'system_prompt':'host','tools':'host','turn_context':'host','history':'verified_owner'},
            'routes':{'head':route}}}
        cfg.setdefault('model',{}).update(default=model,base_url=f"http://127.0.0.1:{ports['hermes']}/head/v1")
        path=home/'.hermes/config.yaml';save_config(path,cfg);result['configs'].append(str(path))
    if 'openclaw' in selected:
        control=state/'openclaw-control.key'
        ensure_key(control)
        bridge_cfg=state/'openclaw-bridge.json'
        save_config(bridge_cfg,{'port':ports['openclaw'],'approvals_file':str(store.path),
                              'control_key_file':str(control),'key_file':str(key),'routes':{'head':route}})
        extension=current_transaction().stage_tree(home/'.openclaw/extensions/owner-mode')
        for filename in ['index.mjs','terminal.mjs','terminal-runner.mjs','model-params.mjs','package.json','openclaw.plugin.json','LICENSE','NOTICE.md']:
            shutil.copy2(ROOT.parent/'openclaw-owner-mode'/filename,extension/filename)
        descriptor={'id':model,'name':model,'reasoning':True,'input':['text'],
                    'contextWindow':32768,'maxTokens':8192,
                    'cost':{'input':0,'output':0,'cacheRead':0,'cacheWrite':0}}
        model_info = model_info or {}
        if isinstance(model_info.get('input'), list):
            modalities = [item for item in model_info['input'] if item in ('text', 'image')]
            if modalities:
                descriptor['input'] = modalities
        for field in ('contextWindow', 'maxTokens'):
            if type(model_info.get(field)) is int and model_info[field] > 0:
                descriptor[field] = model_info[field]
        if request_policy and request_policy.get('output_budget') == 'remaining_context':
            implicit_limit = (not (type(model_info.get('maxTokens')) is int and model_info['maxTokens'] > 0)
                              or model_info.get('maxTokensSource') == 'harness_output')
            if implicit_limit:
                # Native catalog/worker schemas need a positive capacity. The
                # provider wrapper removes this synthetic value before transport.
                descriptor['maxTokens'] = descriptor['contextWindow']
            request_policy = {**request_policy, 'implicit_model_limit': implicit_limit}
        cfg=configs['openclaw']
        descriptor, result['openclaw_model_compatibility'] = configure_openclaw_compat(
            cfg, model, descriptor, model_info)
        cfg.setdefault('session', {})['dmScope'] = scope['selected']
        result['openclaw_session_scope'] = scope
        cfg.setdefault('plugins',{}).setdefault('entries',{})['owner-mode']={
            'enabled':True,'hooks':{'allowConversationAccess':True},'config':{
            'bridgeUrl':f"http://127.0.0.1:{ports['openclaw']}",'controlKeyFile':str(control),
            'python':sys.executable,'corePath':str(release),'bridgeConfig':str(bridge_cfg),'model':descriptor,
            **({'requestPolicy':request_policy} if request_policy else {})}}
        cfg.setdefault('models',{}).setdefault('providers',{})['owner-mode']={
            'baseUrl':f"http://127.0.0.1:{ports['openclaw']}/head/v1",'api':'openai-completions',
            'apiKey':'local-owner-mode','models':[descriptor]}
        defaults=cfg.setdefault('agents',{}).setdefault('defaults',{})
        existing=defaults.get('model')
        defaults['model']={**(existing if isinstance(existing,dict) else {}),'primary':'owner-mode/'+model}
        path=home/'.openclaw/openclaw.json';save_config(path,cfg);result['configs'].append(str(path))
    save_config(state/'setup-result.json',result)
    return result


def select_profile(console, model, candidates, default=''):
    matches = []
    for candidate in candidates:
        ok, _ = match_profile(candidate, model)
        if ok and not any(item['name'] == candidate['name'] for item in matches):
            matches.append(candidate)
    def status(candidate):
        if candidate.get('artifact_valid'):
            return 'file checked', 'File integrity checked'
        if candidate.get('payload_status') == 'missing':
            return 'metadata checked; payload missing', 'Metadata checked; public payload downloads during managed setup'
        return 'engine reports loaded', 'Reported as loaded by the engine'
    for number, candidate in enumerate(matches, 1):
        kind = status(candidate)[0]
        console.line(f"  {number}. {candidate['name']} ({kind})")
    if console.interactive and matches:
        options = [(candidate['name'], status(candidate)[1]) for candidate in matches]
        index = console.select('Choose an alignment profile',
                               [*options, ('Another profile', 'Enter a profile folder or an installed name.')], default=0)
        if index < len(matches):
            return matches[index]
    suggested = '1' if matches else default
    answer = input(f'Profile number, folder or installed name [{suggested}]: ').strip() or suggested
    if answer.isdigit() and 0 < int(answer) <= len(matches):
        return matches[int(answer) - 1]
    if not answer:
        raise SetupError('Supply a profile folder or the name of a profile loaded in the engine')
    return resolve_profile(answer, model, candidates)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='flashalign',description=__doc__)
    parser.add_argument('--version', action='version', version=f'Aevonix FlashAlign {module.__version__}')
    parser.add_argument('--home',type=Path,default=Path.home())
    parser.add_argument('--detect',action='store_true')
    parser.add_argument('--profiles', type=Path, action='append', default=[], metavar='PATH',
                        help='Profile folder, catalog or directory of profiles; repeatable')
    parser.add_argument('--no-restart', action='store_true', help='Apply setup without restarting the selected gateways')
    parser.add_argument('--no-color', action='store_true', help='Disable terminal colors')
    parser.add_argument('--no-animation', action='store_true', help='Show the terminal interface without the startup animation')
    parser.add_argument('--config', type=Path, help='JSON setup plan for noninteractive checks, preparation or activation')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='Check a setup plan without changing configuration')
    action.add_argument('--apply', action='store_true', help='Prepare a setup plan without starting engines or restarting gateways')
    action.add_argument('--activate', action='store_true', help='Start a prepared engine, verify it, apply configuration and restart gateways')
    parser.add_argument('--init-key', nargs='?', const='', metavar='PATH', help='Create or reuse a deployment signing key')
    parser.add_argument('--auto', action='store_true', help='Default managed setup from a local model; asks only for missing choices')
    parser.add_argument('--engine-only', action='store_true', help='Prepare direct engine access without a harness or contact setup')
    parser.add_argument('--model-path', type=Path, help='Local checkpoint for default managed setup')
    parser.add_argument('--profile', type=Path, help='Local profile for default managed setup')
    parser.add_argument('--engine-bundle', type=Path, help='Local TensorFold runtime package for managed setup')
    parser.add_argument('--backend', choices=('auto', 'tensorfold', 'vllm', 'sglang', 'llamacpp'), default='auto',
                        help='Engine for default setup; auto chooses from the local model format')
    parser.add_argument('--engine-python', help='Python interpreter for the managed engine')
    parser.add_argument('--drafter', type=Path, help='Existing local draft model; active speculative decoding is preserved')
    parser.add_argument('--engram-source', type=Path, help='Original DeepSeek checkpoint folder containing its Engram tables')
    parser.add_argument('--engram-dir', type=Path, help='Prepared per-rank DeepSeek Engram table directory')
    parser.add_argument('--hosts', help='Comma-separated SSH host names for a TensorFold cluster')
    parser.add_argument('--ssh-config', type=Path, help='Owned SSH config file for native model hosts')
    parser.add_argument('--master-address', help='Private IPv4 address reachable by every native rank')
    parser.add_argument('--network-interface', help='NCCL network interface on the native model hosts')
    parser.add_argument('--devices', help='Comma-separated NVIDIA device IDs for managed setup')
    parser.add_argument('--tools-default', choices=('on', 'off'), help='Default profile inheritance for tool and subagent calls')
    parser.add_argument('--mode', choices=('verified', 'always_on'),
                        help='Verified contacts (default), or Always On for every request routed through FlashAlign')
    parser.add_argument('--yes', action='store_true', help='Apply default setup without the final confirmation prompt')
    parser.add_argument('--stop-engine', type=Path, metavar='DEPLOYMENT_JSON', help='Stop only a managed FlashAlign engine')
    args=parser.parse_args(argv)
    if args.config and args.mode is not None:
        parser.error('Set mode in the setup plan when using --config')
    if args.engine_only and not args.config and not args.detect:
        args.auto = True
    if (args.check or args.apply or args.activate) and not args.config:
        parser.error('--check, --apply and --activate require --config')
    if args.stop_engine:
        from owner_mode.profiles import read_json
        result = stop_engine(read_json(args.stop_engine))
        print(json.dumps(result, indent=2))
        if result['status'] != 'stopped':
            raise SystemExit(1)
        return result
    if args.init_key is not None:
        if args.config or args.detect:
            parser.error('--init-key is a separate operation')
        key_path = Path(args.init_key) if args.init_key else args.home/'.config/owner-mode/engine.key'
        print(json.dumps(init_signing_key(key_path), indent=2))
        return
    if args.apply or args.activate or (not args.config and not args.detect):
        recover_setup(args.home)
    elif pending_setup(args.home):
        raise SetupError('Interrupted setup needs recovery; rerun the setup plan with --apply')
    found=discover(args.home,probe=model_list)
    if args.auto:
        if args.config or args.detect:
            parser.error('--auto cannot be combined with --config or --detect')
        console = Console(color=not args.no_color, animation=not args.no_animation)
        plan = make_default_plan(args, found, console)
        if not args.yes and input('Prepare engine and apply this setup? [y/N]: ').strip().lower() not in ('y', 'yes'):
            return
        console.step(6, 'Prepare setup' if args.no_restart else 'Prepare and activate')
        console.line('Preparing the managed engine...' if args.no_restart else
                     'Preparing the managed engine and checking profile activation...')
        result = run_plan(plan, args.home, found, provision, activate=not args.no_restart, apply=True)
        if not args.no_restart and result['harnesses']:
            result['restart'] = restart_harnesses(args.home, result['harnesses'], found)
            success = all(row['status'] == 'restarted' for row in result['restart'].values())
            result['status'] = 'configured' if success else 'restart_failed'
        else:
            success = not args.no_restart
        save_config(args.home/'.config/owner-mode/setup-result.json', result)
        console.result('Setup prepared; activation pending' if args.no_restart else
                       'Direct engine verified' if args.engine_only else 'Engine verified and gateway restart requested',
                       ok=args.no_restart or success)
        return result
    if args.config:
        result = run_plan(args.config, args.home, found, provision, apply=args.apply, activate=args.activate, profile_paths=args.profiles)
        gateway_harnesses = [name for name in result['harnesses'] if name not in result.get('terminal_harnesses', [])]
        if args.activate and not args.no_restart and gateway_harnesses:
            result['restart'] = restart_harnesses(args.home, gateway_harnesses, found)
            result['status'] = 'configured' if all(row['status'] == 'restarted' for row in result['restart'].values()) else 'restart_failed'
        if args.apply or args.activate:
            save_config(args.home/'.config/owner-mode/setup-result.json', result)
        print(json.dumps(result, indent=2))
        return result
    if args.detect:
        from owner_mode.backend_setup import backend_catalog
        roots = [*args.profiles, args.home/'.config/owner-mode/profiles', ROOT.parent/'profiles']
        local, errors = discover_profiles(roots)
        found['profiles'] = {'valid': local, 'invalid': errors}
        found['backends'] = backend_catalog()
        print(json.dumps(found,indent=2));return
    console = Console(color=not args.no_color, animation=not args.no_animation)
    console.banner(module.__version__)
    console.step(1, 'Choose your harness')
    for name, info in found.items():
        label = 'Hermes' if name == 'hermes' else 'OpenClaw'
        console.line(f"  {label}: {'detected' if info['detected'] else 'not detected'}")
    detected=[name for name,value in found.items() if value['detected']]
    default_harness = 'both' if len(detected)==2 else detected[0] if detected else 'hermes'
    if console.interactive:
        index = console.select('Choose your harness', [
            ('Hermes', 'Use the installed Hermes integration.'),
            ('OpenClaw', 'Use the installed OpenClaw integration.'),
            ('Both', 'Configure Hermes and OpenClaw.')],
            default=('hermes', 'openclaw', 'both').index(default_harness))
        choice = ('hermes', 'openclaw', 'both')[index]
    else:
        choice=input('Harness [hermes/openclaw/both] ('+('/'.join(detected) or 'hermes')+'): ').strip().lower()
        choice=choice or default_harness
    if choice not in {'hermes','openclaw','both'}:raise SystemExit('Unknown harness')
    selected=['hermes','openclaw'] if choice=='both' else [choice]
    console.context(harness=', '.join('Hermes' if name == 'hermes' else 'OpenClaw' for name in selected))
    if 'openclaw' in selected and found['openclaw'].get('session_scope_error'):
        raise SetupError(found['openclaw']['session_scope_error'])
    if 'openclaw' in selected and not found['openclaw']['runtime_compatible']:
        raise SystemExit('This OpenClaw adapter requires Node 24.22.1+ (24.x) or Node 26.12.1+ (26.x).')
    console.step(2, 'Choose your model')
    defaults=[u for name in selected for u in found[name]['model_urls']]
    base=input('Model API base URL'+(' ['+defaults[0]+']' if defaults else '')+': ').strip() or (defaults[0] if defaults else '')
    api_key = os.environ.get('FLASHALIGN_API_KEY') or None
    try:
        rows = model_rows(base, api_key)
    except SetupError as error:
        if error.status not in (401, 403) or api_key:
            raise
        api_key = getpass.getpass('Model endpoint API key: ')
        rows = model_rows(base, api_key)
    if not rows:
        raise SetupError('The endpoint has no available models')
    for number, row in enumerate(rows, 1):
        console.line(f"  {number}. {row['id']}")
    if console.interactive:
        index = console.select('Choose your model', [(row['id'], 'Available from the selected endpoint.')
                                                    for row in rows], default=0)
        model_info = rows[index]
    else:
        answer = input('Model number or ID [1]: ').strip() or '1'
        if answer.isdigit() and 0 < int(answer) <= len(rows):
            model_info = rows[int(answer) - 1]
        else:
            model_info = next((row for row in rows if row['id'] == answer), None)
            if model_info is None:
                raise SetupError('Choose a model listed by the selected endpoint')
    model = model_info['id']
    console.context(model=model)
    check_harness_contexts(selected, found,
        {'tokens': model_info.get('contextWindow'), 'source': 'model endpoint metadata'})
    console.step(3, 'Find and verify a profile')
    roots = [*args.profiles, args.home/'.config/owner-mode/profiles', ROOT.parent/'profiles']
    roots += [Path(item) for item in os.environ.get('FLASHALIGN_PROFILE_PATH', '').split(os.pathsep) if item]
    local, errors = discover_profiles(roots)
    for error in errors:
        console.result(f"Skipped profile: {error['reason']}", ok=False)
    try:
        available = engine_profiles(base, api_key)
    except SetupError:
        available = []
    details = select_profile(console, model_info, local + available, default=profile_for(model) or '')
    profile = details['name']
    console.context(profile=profile, status='Selected; checking engine activation')
    existing_key = args.home/'.config/owner-mode/engine.key'
    default_key = str(existing_key) if existing_key.is_file() else ''
    key = input('Engine signing-key file'+(f' [{default_key}]' if default_key else '')+': ').strip() or default_key
    if not key:
        raise SetupError('The running engine and FlashAlign need the same signing-key file; obtain it from the engine operator')
    console.line('Checking standard mode and this profile with two one-token model requests...')
    check = check_engine(base, model, profile, key, api_key)
    console.result(f'Engine activated {profile}; standard mode also confirmed')
    console.context(status='Standard and selected-profile requests confirmed')
    if details.get('artifact_valid'):
        console.line('Profile file integrity checked. Numerical fit and model quality are separate checks.')
    console.step(4, 'Choose access')
    current_store = ApprovalStore(args.home/'.config/owner-mode/approvals.json').read()
    mode = args.mode
    if mode is None:
        default_mode = current_store.get('mode', 'verified')
        if console.interactive:
            index = console.select('Who gets the profile?', [
                ('Verified contacts', 'Approved contacts and authorized local sessions.'),
                ('Always On', 'Every request routed through FlashAlign, including tools and subagents.')],
                default=0 if default_mode == 'verified' else 1)
            mode = ('verified', 'always_on')[index]
        else:
            console.line('  1. Verified contacts: approved contacts on verified channels')
            console.line('  2. Always On: all requests routed through FlashAlign, including tools and subagents')
            choice = input(f'Access mode [1/2] [{1 if default_mode == "verified" else 2}]: ').strip()
            mode = {'1': 'verified', '2': 'always_on'}.get(choice, choice or default_mode)
    if mode not in ('verified', 'always_on'):
        raise SetupError('mode must be verified or always_on')
    console.context(access='Always On' if mode == 'always_on' else 'Verified contacts')
    contacts = []
    if mode == 'verified':
        contacts = [choose_verified_contact(console, selected, ApprovalStore(args.home/'.config/owner-mode/approvals.json'))]
    else:
        console.line('Always On uses the selected profile for every routed request. It requires no contact approval.')
    console.step(5, 'Set tool profile inheritance')
    if mode == 'always_on':
        tools = args.tools_default or ('on' if current_store['tools']['default'] else 'off')
        console.line('Always On includes all routed tool and subagent requests. Verified-mode tool settings are preserved.')
    else:
        if args.tools_default:
            tools = args.tools_default
        elif console.interactive:
            index = console.select('Tools inherit the profile', [
                ('On', 'Apply the active profile to eligible tools and subagents.'),
                ('Off', 'Use the standard model for tool and subagent requests.')], default=0)
            tools = ('on', 'off')[index]
        else:
            tools = input('Apply profile to tool calls by default [on/off] [on]: ').strip().lower() or 'on'
    if tools not in {'on','off'}:raise SystemExit('Tool profile default must be on or off')
    console.context(tools='All routed calls' if mode == 'always_on' else tools.title())
    console.step(6, 'Review and activate')
    console.line(f"Harness: {', '.join(selected)}")
    console.line(f'Model: {model} | Profile: {profile} | Tool profile default: {tools}')
    console.line('Access mode: '+('Always On (all routed requests)' if mode == 'always_on' else 'Verified contacts'))
    if mode == 'always_on':
        console.line('Requests fail if the selected profile cannot be applied; there is no per-request stock bypass.')
    console.line('Existing configuration will be backed up.')
    if 'openclaw' in selected:
        scope = found['openclaw'].get('session_scope') or openclaw_session_scope(
            load_config(args.home/'.openclaw/openclaw.json'))
        console.line('OpenClaw direct-message session scope: '+scope['selected']+
                     (' (updated during setup)' if scope['change_required'] else ''))
    gateway_harnesses = [name for name in selected if name not in terminal_harnesses(contacts)]
    console.line('Gateway restart: '+('skipped' if args.no_restart or not gateway_harnesses else ', '.join(gateway_harnesses)))
    if input('Apply setup? [y/N]: ').strip().lower() not in {'y','yes'}:return
    openclaw_install = None
    if 'openclaw' in selected:
        openclaw_install = register_openclaw(args.home, found['openclaw'].get('executable'), ROOT.parent/'openclaw-owner-mode')
    result=provision(args.home,selected,base,model,profile,contacts,tools=='on',key,
                     profile_details=details, engine_check=check, model_info=model_info, api_key=api_key, mode=mode)
    if openclaw_install is not None:result['openclaw_install'] = openclaw_install
    if args.no_restart:
        result['restart'] = {name: {'status': 'skipped'} for name in gateway_harnesses}
    else:
        result['restart'] = restart_harnesses(args.home, gateway_harnesses, found)
    restarted = not args.no_restart and all(item['status'] == 'restarted' for item in result['restart'].values())
    result['status'] = 'configured' if restarted else ('awaiting_restart' if args.no_restart else 'restart_failed')
    result['activation'] = (('gateway restart completed; Always On selected for all routed requests'
                             if mode == 'always_on' else 'gateway restart completed; native channel verification required')
                            if restarted else 'gateway restart required')
    save_config(args.home/'.config/owner-mode/setup-result.json', result)
    for name, item in result['restart'].items():
        label = 'Hermes' if name == 'hermes' else 'OpenClaw'
        console.result(f"{label}: {item['status']}", ok=item['status'] == 'restarted')
    if restarted:
        if not result.get('enabled', True):
            console.result('FlashAlign is configured and remains disabled.')
        elif mode == 'always_on':
            console.result('FlashAlign configured. Always On applies to every request routed through FlashAlign.')
        elif not result.get('terminal_harnesses'):
            console.result('FlashAlign configured. Send a message from your approved contact to verify the native channel.')
    else:
        console.line('Configuration saved. Start or restart the selected gateway before using FlashAlign.')
        for name in selected:
            console.line(f'  {name} gateway restart')
    for command in result.get('terminal_commands', []):
        console.line('Start an authorized native terminal: '+ ' '.join(command))
    return result


if __name__=='__main__':
    try:
        result = main()
        if result and any(item['status'] in ('failed', 'unavailable') for item in result.get('restart', {}).values()):
            raise SystemExit(1)
    except (SetupError, ValueError, OSError) as error:
        Console(color=False).result(str(error), ok=False)
        raise SystemExit(1)
    except (EOFError, KeyboardInterrupt):
        print('\nSetup cancelled.')
        raise SystemExit(1)
