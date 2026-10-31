# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Noninteractive agent setup. Configuration is applied without restarting gateways."""
import os
from pathlib import Path
import stat

from .auth import read_api_key
from .contacts import pair
from .access_setup import terminal_harnesses
from .profiles import bundled_profile_root, discover_profiles, read_json, resolve_profile
from .setup_runtime import SetupError, check_engine, engine_profiles, model_rows, register_openclaw, check_harness_contexts
from .engine_setup import check_engine_setup, prepare_engine, activate_engine, private_json
from .setup_transaction import pending_setup, recover_setup
from .model_metadata import check_openclaw_request_file
from .direct_gateway import prepare_gateway, activate_gateway, prior_client_gateway, retire_prior_gateway


def read_plan(path):
    path = Path(path).expanduser().absolute()
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid() or info.st_nlink != 1):
        raise SetupError('Setup plan must be an owned 0600 regular file')
    value = read_json(path)
    fields = {'harnesses', 'base_url', 'model', 'profile', 'engine_key_file',
              'contacts', 'tools_default', 'profile_paths', 'api_key_file', 'engine', 'engine_only', 'mode'}
    if not isinstance(value, dict) or set(value) - fields:
        raise SetupError('Setup plan contains unknown fields')
    direct = value.get('engine_only', False)
    if type(direct) is not bool:
        raise SetupError('engine_only must be true or false')
    mode = value.setdefault('mode', 'verified')
    if mode not in ('verified', 'always_on'):
        raise SetupError('mode must be verified or always_on')
    required = {'model', 'profile'}
    if not direct:
        required |= {'harnesses', 'tools_default'}
        if mode == 'verified':
            required.add('contacts')
        else:
            value.setdefault('contacts', [])
    else:
        value.setdefault('harnesses', [])
        value.setdefault('contacts', [])
        value.setdefault('tools_default', False)
    if 'engine' not in value:
        required |= {'base_url', 'engine_key_file'}
    if not required <= set(value):
        raise SetupError('Setup plan is missing required fields')
    selected = value['harnesses']
    if direct and (selected or value['contacts']):
        raise SetupError('Engine-only plans do not configure harnesses or contacts')
    if (not isinstance(selected, list) or (not selected and not direct)
            or any(not isinstance(name, str) or name not in ('hermes', 'openclaw') for name in selected)
            or len(set(selected)) != len(selected)):
        raise SetupError('Choose hermes, openclaw, or both in the setup plan')
    for name in required & {'base_url', 'model', 'profile', 'engine_key_file'}:
        if not isinstance(value[name], str) or not value[name].strip():
            raise SetupError('Setup plan requires nonempty endpoint, model, profile and key paths')
    if type(value['tools_default']) is not bool:
        raise SetupError('tools_default must be true or false')
    contacts = value['contacts']
    if not isinstance(contacts, list) or (not contacts and not direct and mode == 'verified'):
        raise SetupError('Setup plan must specify at least one approved contact')
    normalized = []
    for contact in contacts:
        if not isinstance(contact, dict) or set(contact) != {'channel', 'account', 'contact'}:
            raise SetupError('Each contact needs channel, account and contact fields')
        normalized.append(pair(contact['channel'], contact['account'], contact['contact']))
    paths = value.get('profile_paths', [])
    if not isinstance(paths, list) or any(not isinstance(item, str) for item in paths):
        raise SetupError('profile_paths must be an array of paths')
    def relative(item):
        candidate = Path(item).expanduser()
        return candidate if candidate.is_absolute() else path.parent/candidate
    if value.get('engine_key_file'):
        value['engine_key_file'] = relative(value['engine_key_file'])
    value['profile_paths'] = [relative(item) for item in paths]
    candidate = relative(value['profile'])
    if candidate.exists():
        value['profile'] = str(candidate)
    if 'api_key_file' in value:
        if not isinstance(value['api_key_file'], str) or not value['api_key_file'].strip():
            raise SetupError('api_key_file must be a path')
        value['api_key_file'] = relative(value['api_key_file'])
    if 'engine' in value:
        if not isinstance(value['engine'], dict):
            raise SetupError('engine must be an engine setup object')
        for name in ('bundle', 'model_path', 'source', 'draft_model', 'installation', 'ssh_config'):
            if name in value['engine']:
                value['engine'][name] = str(relative(value['engine'][name]))
    value['contacts'] = normalized
    return value


def run_plan(path, home, found, provision, *, apply=False, activate=False, profile_paths=()):
    if apply or activate:recover_setup(home)
    elif pending_setup(home):raise SetupError('Interrupted setup needs recovery; rerun the setup plan with --apply')
    plan = read_plan(path)
    plan.setdefault('mode', 'verified')
    prior_gateway = prior_client_gateway(home) if (apply or activate) and plan.get('engine_only') else None
    for name in plan['harnesses']:
        label = 'Hermes' if name == 'hermes' else 'OpenClaw'
        if not found[name]['detected']:
            raise SetupError(f'{label} is not installed; install it before applying FlashAlign setup')
        if not found[name]['runtime_compatible']:
            raise SetupError(f'{label} runtime is incompatible with this adapter')
        if name == 'openclaw' and found[name].get('session_scope_error'):
            raise SetupError(found[name]['session_scope_error'])
    token = read_api_key(plan['api_key_file']) if plan.get('api_key_file') else os.environ.get('FLASHALIGN_API_KEY')
    if plan.get('engine'):
        model = {'id': plan['model']}
    else:
        models = model_rows(plan['base_url'], token)
        model = next((item for item in models if item['id'] == plan['model']), None)
    if model is None:
        raise SetupError('The plan model is not listed by the selected endpoint')
    environment_paths = [Path(item) for item in os.environ.get('FLASHALIGN_PROFILE_PATH', '').split(os.pathsep) if item]
    local, errors = discover_profiles([*plan['profile_paths'], *profile_paths,
                                      *environment_paths, home/'.config/owner-mode/profiles',
                                      bundled_profile_root()])
    remote = []
    if not plan.get('engine'):
        try:
            remote = engine_profiles(plan['base_url'], token)
        except SetupError:
            pass
    details = resolve_profile(plan['profile'], model, local + remote)
    if plan.get('engine'):
        if details.get('source') != 'local':
            raise SetupError('Managed engine setup requires a local profile artifact')
        compatibility = check_engine_setup(home, plan['engine'], details, model_id=model['id'])
        if compatibility.get('input_modalities') is not None:
            model['input'] = compatibility['input_modalities']
        if compatibility.get('openclaw_compat') is not None:
            model['openclaw_compat'] = compatibility['openclaw_compat']
        if compatibility.get('request_compatibility') is not None:
            model['request_compatibility'] = compatibility['request_compatibility']
        if compatibility.get('openclaw_request_policy') is not None:
            model['openclaw_request_policy'] = compatibility['openclaw_request_policy']
        if 'openclaw' in plan['harnesses']:
            check_openclaw_request_file(home/'.openclaw/openclaw.json', model['id'], model.get('openclaw_request_policy'))
        context_check = check_harness_contexts(plan['harnesses'], found,
            compatibility.get('context_window', {'tokens': None, 'source': 'not declared by this managed backend'}),
            managed=True)
        if context_check['engine']['tokens']:
            model['contextWindow'] = context_check['engine']['tokens']
            outputs = [row['tokens'] for row in context_check['harnesses'].get('openclaw', []) if row['kind'] == 'output']
            if outputs:
                model['maxTokens'] = max(outputs)
                model['maxTokensSource'] = 'harness_output'
        result = {'status': 'engine_setup_verified', 'applied': False, 'harnesses': plan['harnesses'],
                  'model': model['id'], 'profile': details['name'], 'mode': plan['mode'], 'artifact_validated': True,
                  'backend': compatibility['kind'],
                  'activation': 'engine preparation and restart required', 'context_compatibility': context_check,
                  'input_modalities': compatibility.get('input_modalities'),
                  'native_image_setup': 'validated serving lane' if compatibility.get('input_modalities') is not None
                                        else 'unverified: serving lane has no input modality declaration'}
        requirements = compatibility.get('metadata', {})
        if compatibility.get('python'):
            result['engine_python'] = compatibility['python']
        if requirements.get('resource_policy'):
            result['engine_resource_policy'] = requirements['resource_policy']
        if requirements.get('model_integrity_verifier') or requirements.get('checkpoint_integrity'):
            result['model_integrity'] = 'full pinned payload verification during apply; no model writes'
        if 'openclaw' in plan['harnesses']:
            result['openclaw_session_scope'] = found['openclaw'].get('session_scope',
                {'current': 'main', 'selected': 'per-account-channel-peer', 'change_required': True})
        if not apply and not activate:
            return result
        deployment = prepare_engine(home, plan['engine'], details['manifest'], model_id=model['id'],
                                    key_file=plan.get('engine_key_file'))
        result.update(engine=deployment, status='engine_prepared', applied=True,
                      activation='engine launch pending' if plan.get('engine_only') else 'engine launch and gateway restart pending')
        result['next_command'] = ['bash', str(Path(__file__).resolve().parents[1]/'install.sh'),
                                  '--home', str(home), '--config', str(Path(path).absolute()), '--activate']
        if not activate:
            if plan.get('engine_only') and plan['mode'] == 'always_on':
                gateway = prepare_gateway(home, base_url=deployment['base_url'], model=model['id'],
                    profile=details['name'], key_file=deployment['key_file'], api_key_file=plan.get('api_key_file'),
                    api_key=token, request_compatibility=model.get('request_compatibility'))
                result.update(gateway=gateway, base_url=gateway['base_url'],
                              activation='engine and Always On gateway launch pending')
            return result
        deployment = activate_engine(deployment, timeout=plan['engine'].get('startup_timeout', 900))
        plan['base_url'], plan['engine_key_file'] = deployment['base_url'], deployment['key_file']
        result['engine'] = deployment
    else:
        context_check = check_harness_contexts(plan['harnesses'], found,
            {'tokens': model.get('contextWindow'), 'source': 'model endpoint metadata'})
    if plan.get('engine_only') and plan['mode'] == 'always_on' and apply and not activate:
        gateway = prepare_gateway(home, base_url=plan['base_url'], model=model['id'], profile=details['name'],
            key_file=plan['engine_key_file'], api_key_file=plan.get('api_key_file'), api_key=token,
            request_compatibility=model.get('request_compatibility'))
        return {'status':'gateway_prepared', 'applied':True, 'harnesses':[], 'engine_only':True,
                'mode':'always_on', 'model':model['id'], 'profile':details['name'],
                'artifact_validated':details.get('artifact_valid') is True,
                'gateway':gateway, 'base_url':gateway['base_url'], 'restart':{},
                'activation':'Always On gateway launch pending',
                'next_command':['bash',str(Path(__file__).resolve().parents[1]/'install.sh'),
                                '--home',str(home),'--config',str(Path(path).absolute()),'--activate']}
    check = check_engine(plan['base_url'], model['id'], details['name'], plan['engine_key_file'], token)
    result = {**(result if plan.get('engine') else {}), 'status': 'verified', 'applied': False, 'harnesses': plan['harnesses'],
              'model': model['id'], 'profile': details['name'], 'mode': plan['mode'], 'engine_check': check,
              'artifact_validated': details.get('artifact_valid') is True,
              'invalid_profile_count': len(errors), 'context_compatibility': context_check}
    if 'openclaw' in plan['harnesses']:
        result['openclaw_session_scope'] = found['openclaw'].get('session_scope',
            {'current': 'main', 'selected': 'per-account-channel-peer', 'change_required': True})
    if plan.get('engine_only'):
        result.update(restart={}, activation='direct engine verified', engine_only=True)
        if apply and not activate and prior_gateway is not None:
            # Keep the active gateway pointer until activation can retire it.
            # Publishing a direct client now would lose the owned stop handle.
            result.update(status='gateway_transition_prepared', applied=False,
                          activation='existing gateway remains active until --activate',
                          next_command=['bash', str(Path(__file__).resolve().parents[1]/'install.sh'),
                                        '--home', str(home), '--config', str(Path(path).absolute()), '--activate'])
            return result
        if apply or activate:
            if plan['mode'] == 'always_on':
                gateway = prepare_gateway(home, base_url=plan['base_url'], model=model['id'], profile=details['name'],
                    key_file=plan['engine_key_file'], api_key_file=plan.get('api_key_file'), api_key=token,
                    request_compatibility=model.get('request_compatibility'))
                gateway = activate_gateway(gateway)
                result.update(gateway=gateway, base_url=gateway['base_url'], activation='Always On gateway verified')
                client = {'base_url':gateway['base_url'], 'model':model['id'],
                          'profile':details['name'], 'mode':'always_on', 'gateway':True,
                          'gateway_deployment':gateway['deployment_file']}
            else:
                client = {'base_url': plan['base_url'], 'model': model['id'],
                          'profile': details['name'], 'key_file': str(plan['engine_key_file']), 'mode': plan['mode']}
                if plan.get('api_key_file'):
                    client['api_key_file'] = str(plan['api_key_file'])
            if activate:
                retired = retire_prior_gateway(prior_gateway, result.get('gateway'))
                if retired is not None:
                    result['previous_gateway'] = retired
            config = home/'.config/flashalign/client.json'
            private_json(config, client)
            result.update(status='engine_ready', applied=True, client_config=str(config))
        return result
    if apply or activate:
        openclaw_install = None
        if 'openclaw' in plan['harnesses']:
            openclaw_install = register_openclaw(home, found['openclaw'].get('executable'),
                Path(__file__).resolve().parents[1]/'openclaw-owner-mode')
        result.update(provision(home, plan['harnesses'], plan['base_url'], model['id'], details['name'],
            plan['contacts'], plan['tools_default'], plan['engine_key_file'],
            profile_details=details, engine_check=check, model_info=model, api_key=token, mode=plan['mode']))
        if openclaw_install is not None:result['openclaw_install'] = openclaw_install
        result.update(status='awaiting_restart', applied=True)
    local_harnesses = terminal_harnesses(plan['contacts'])
    result['restart'] = {name: {'status': 'pending', 'command': [name, 'gateway', 'restart'],
                              'env': {'HERMES_HOME': str(home/'.hermes')} if name == 'hermes' else
                                     {'OPENCLAW_STATE_DIR': str(home/'.openclaw'),
                                      'OPENCLAW_CONFIG_PATH': str(home/'.openclaw/openclaw.json')}}
                         for name in plan['harnesses'] if name not in local_harnesses}
    result['activation'] = ('authorized native terminal launch required' if apply and local_harnesses and not result['restart']
                            else 'operator-approved gateway restart required' if apply else 'preflight only; configuration unchanged')
    return result
