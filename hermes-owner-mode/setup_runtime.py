# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Bounded setup-time engine checks and harness service control."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request

from .auth import read_key, sign
from .terminal_ui import TerminalUI


# The supported Hermes core enforces this even when model.context_length is
# absent. This is a compatibility requirement, not a model capacity estimate.
HERMES_MINIMUM_CONTEXT = 64_000


class SetupError(ValueError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


def openclaw_session_scope(config):
    """Plan native direct-message isolation without reading sender identities."""
    session = config.get('session', {})
    if not isinstance(session, dict):
        raise SetupError('OpenClaw session configuration must be an object')
    current = session.get('dmScope', 'main')
    if current not in ('main', 'per-peer', 'per-channel-peer', 'per-account-channel-peer'):
        raise SetupError('OpenClaw has an unsupported direct-message session scope')
    selected = current if current in ('per-channel-peer', 'per-account-channel-peer') else 'per-account-channel-peer'
    return {'current': current, 'selected': selected, 'change_required': selected != current}


def harness_context_requirements(name, config):
    """Read only explicit limits which a route change must not silently reduce."""
    rows = []
    def add(field, value, kind):
        if value is None:
            return
        if type(value) is not int or value <= 0:
            raise SetupError(f'{name} {field} must be a positive token count before managed setup')
        rows.append({'field': field, 'tokens': value, 'kind': kind})
    if name == 'hermes':
        model = config.get('model', {})
        if isinstance(model, dict):
            add('model.context_length', model.get('context_length'), 'context')
            add('model.max_tokens', model.get('max_tokens'), 'output')
    elif name == 'openclaw':
        defaults = config.get('agents', {}).get('defaults', {})
        add('agents.defaults.contextTokens', defaults.get('contextTokens'), 'context')
        primary = defaults.get('model', {})
        primary = primary.get('primary') if isinstance(primary, dict) else primary
        if isinstance(primary, str) and '/' in primary:
            provider, _, model_id = primary.partition('/')
            descriptor = config.get('models', {}).get('providers', {}).get(provider, {})
            for model in descriptor.get('models', []) if isinstance(descriptor, dict) else []:
                if isinstance(model, dict) and model.get('id') == model_id:
                    add('selected model.contextWindow', model.get('contextWindow'), 'context')
                    add('selected model.maxTokens', model.get('maxTokens'), 'output')
                    break
    return rows


def normalize_engine_arguments(kind, arguments, *, llamacpp_method=None, llamacpp_draft_model=False):
    """Normalize known engine option aliases without changing their values."""
    if kind == 'llamacpp':
        return _normalize_llamacpp_arguments(arguments, llamacpp_method, llamacpp_draft_model)
    result = []
    for value in arguments:
        if kind == 'vllm' and value.startswith('--'):
            option, separator, inline = value.partition('=')
            name, dot, nested = option.partition('.')
            value = name.replace('_', '-') + dot + nested + separator + inline
        result.append(value)
    return result


def _normalize_llamacpp_arguments(arguments, method, has_draft):
    """Migrate removed token-count options only when their target is unambiguous."""
    arguments = list(arguments)
    legacy_max = {'--draft', '--draft-n', '--draft-max'}
    legacy_min = {'--draft-min', '--draft-n-min'}
    legacy_ngram = {'--spec-ngram-size-n', '--spec-ngram-size-m', '--spec-ngram-min-hits'}
    legacy = legacy_max | legacy_min | legacy_ngram
    if not any(value.split('=', 1)[0] in legacy for value in arguments):
        return arguments
    declared = []
    for index, value in enumerate(arguments):
        option, separator, inline = value.partition('=')
        if option == '--spec-type':
            selected = inline if separator else arguments[index+1] if index+1 < len(arguments) else ''
            declared.extend(selected.split(','))
        if option in ('--spec-draft-model', '--model-draft', '-md'):
            has_draft = True
    if method is not None and not isinstance(method, str):
        raise SetupError('A llama.cpp speculative method must be a string')
    if method is not None:
        declared.append(method)
    methods = set(declared)
    if len(methods) > 1 or '' in methods:
        raise SetupError('Legacy llama.cpp token counts are ambiguous with multiple speculative methods; use current method-specific options')
    selected = next(iter(methods), 'draft-simple' if has_draft else None)
    canonical_counts = {'--spec-draft-n-max', '--spec-draft-n-min', '--spec-ngram-mod-n-max',
                        '--spec-ngram-mod-n-min', '--spec-ngram-mod-n-match'}
    canonical_counts.update('--spec-'+kind+'-'+suffix
        for kind in ('ngram-simple', 'ngram-map-k', 'ngram-map-k4v')
        for suffix in ('size-n', 'size-m', 'min-hits'))
    result, seen = [], {}
    for index, value in enumerate(arguments):
        option, separator, inline = value.partition('=')
        target = option
        if option in legacy_max | legacy_min:
            suffix = 'n-max' if option in legacy_max else 'n-min'
            if selected and selected.startswith('draft-'):
                target = '--spec-draft-'+suffix
            elif selected == 'ngram-mod':
                target = '--spec-ngram-mod-'+suffix
            else:
                raise SetupError('Legacy llama.cpp draft token counts are ambiguous for this method; use --spec-draft-n-max or --spec-ngram-mod-n-max')
        elif option in legacy_ngram:
            suffix = option.removeprefix('--spec-ngram-')
            if selected in ('ngram-simple', 'ngram-map-k', 'ngram-map-k4v'):
                target = '--spec-'+selected+'-'+suffix
            elif selected == 'ngram-mod' and suffix == 'size-n':
                target = '--spec-ngram-mod-n-match'
            else:
                raise SetupError('Legacy llama.cpp ngram options are ambiguous for this method; use current method-specific options')
        if option in legacy or target in canonical_counts:
            setting = inline if separator else arguments[index+1] if index+1 < len(arguments) else ''
            try:
                number = int(setting)
            except ValueError as error:
                raise SetupError('A llama.cpp speculative token-count option requires an integer value') from error
            if target in seen and seen[target] != number:
                raise SetupError('Conflicting llama.cpp speculative token counts; use one explicit value for '+target)
            seen[target] = number
        result.append(target+separator+inline if separator else target)
    return result


def require_full_vllm_options(kind, arguments, names):
    if kind != 'vllm':
        return
    for value in arguments:
        name = value.split('=', 1)[0].split('.', 1)[0]
        if (name.startswith('--') and name != '--' and name not in names
                and any(option.startswith(name) for option in names)):
            raise SetupError('Use a complete vLLM option name instead of '+name)


def managed_context_window(kind, metadata, arguments=()):
    """Report the per-sequence cap actually passed to a managed launcher."""
    # Keep this independent from drafting.py, which also imports SetupError.
    def argument(argv, name, default=None):
        names = (name,) if isinstance(name, str) else name
        for index in range(len(argv)-1, -1, -1):
            value = argv[index]
            if any(value.startswith(flag+'=') for flag in names):
                return value.split('=', 1)[1]
            if value in names and index+1 < len(argv):
                return argv[index+1]
        return default
    defaults = metadata.get('default_arguments') or []
    combined = normalize_engine_arguments(kind, [*defaults, *arguments])
    require_full_vllm_options(kind, combined, {'--max-model-len'})
    if kind == 'vllm':
        value, source = argument(combined, '--max-model-len', '4096'), '--max-model-len'
    elif kind == 'sglang':
        fallback = metadata.get('qualification', {}).get('tested_context_length', 4096)
        value, source = argument(combined, '--context-length', str(fallback)), '--context-length'
    elif kind == 'llamacpp':
        value, source = argument(combined, ('--ctx-size', '-c'), '4096'), '--ctx-size / --parallel'
        parallel = argument(combined, ('--parallel', '-np'), '2')
        try:
            value = int(value) // int(parallel)
        except (ValueError, TypeError, ZeroDivisionError):
            value = None
    else:
        value, source = None, 'not declared by this managed backend'
    try:
        tokens = int(value)
    except (ValueError, TypeError):
        tokens = None
    return {'tokens': tokens if tokens and tokens > 0 else None, 'source': source}


def check_harness_contexts(selected, found, window, *, managed=False):
    """Reject a route change when declared harness limits exceed engine context."""
    requirements = {}
    for name in selected:
        if found[name].get('context_requirements_error'):
            raise SetupError(found[name]['context_requirements_error'])
        requirements[name] = list(found[name].get('context_requirements', []))
    tokens = window.get('tokens')
    if type(tokens) is not int or tokens <= 0:
        tokens = None
        window = {**window, 'tokens': None}
    if 'hermes' in selected and (managed or tokens is not None):
        rows = requirements['hermes']
        if not any(row['kind'] == 'context' and row['tokens'] >= HERMES_MINIMUM_CONTEXT for row in rows):
            rows.append({'field': 'runtime minimum context', 'tokens': HERMES_MINIMUM_CONTEXT,
                         'kind': 'context'})
    for name, rows in requirements.items():
        for row in rows:
            if tokens is None:
                raise SetupError(f'{name} declares {row["field"]}={row["tokens"]} tokens, but the selected '
                                 'engine context is unknown; verify its context before changing this route')
            if row['tokens'] > tokens:
                raise SetupError(f'{name} declares {row["field"]}={row["tokens"]} tokens, exceeding the '
                                 f'selected engine context of {tokens}; existing limits were not reduced. '
                                 'Choose an independently qualified compatible engine or explicitly revise the harness configuration')
    status = 'compatible' if any(requirements.values()) else 'no explicit harness limits'
    if 'hermes' in selected and tokens is None:
        status = 'unverified runtime context; endpoint does not advertise a context window'
    return {'status': status,
            'engine': window, 'harnesses': requirements}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint_url(base, suffix):
    parts = urlsplit(base)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise SetupError('Enter an HTTP(S) model base URL without credentials, query or fragment')
    return base.rstrip('/') + suffix


def request(base, suffix, *, payload=None, key=None, api_key=None, timeout=15):
    data = None if payload is None else json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
    headers = {'Accept': 'application/json'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    if key is not None:
        headers['X-Owner-Mode-Auth'] = sign(key, data or b'')
    if api_key:
        if not api_key.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in api_key) or len(api_key) > 8192:
            raise SetupError('Invalid model endpoint API key')
        headers['Authorization'] = 'Bearer ' + api_key
    message = Request(endpoint_url(base, suffix), data=data, headers=headers)
    try:
        response = build_opener(NoRedirect).open(message, timeout=timeout)
    except HTTPError as error:
        response = error
    except (URLError, TimeoutError, OSError) as error:
        raise SetupError('The model endpoint could not be reached') from error
    with response:
        body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise SetupError('The model endpoint returned an oversized response')
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError):
            value = None
        return response.status, response.headers, value


def model_rows(base, api_key=None, *, timeout=5):
    status, _, value = request(base, '/models', api_key=api_key, timeout=timeout)
    if status != 200 or not isinstance(value, dict) or not isinstance(value.get('data'), list):
        raise SetupError('The endpoint did not return an accessible model list', status=status)
    rows = []
    for row in value['data']:
        if isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'].strip():
            item = {field: row[field] for field in ('id', 'revision', 'format', 'input', 'contextWindow', 'maxTokens') if field in row}
            if 'contextWindow' not in item and type(row.get('max_model_len')) is int:
                item['contextWindow'] = row['max_model_len']
            rows.append(item)
    return rows


def engine_profiles(base, api_key=None):
    """Optional discovery extension. A 404 is normal for older engines."""
    status, _, value = request(base, '/owner-mode/profiles', api_key=api_key, timeout=5)
    if status in (404, 405):
        return []
    if status != 200 or not isinstance(value, dict) or not isinstance(value.get('data'), list):
        return []
    rows = []
    for row in value['data'][:4096]:
        if not isinstance(row, dict) or row.get('loaded') is not True:
            continue
        name, model = row.get('name'), row.get('model')
        if (not isinstance(name, str) or not name or name == 'off' or len(name) > 256
                or not isinstance(model, dict) or not isinstance(model.get('id'), str)):
            continue
        rows.append({'name': name, 'model': model, 'source': 'engine', 'loaded': True,
                     'artifact_valid': None, 'status': 'reported loaded by engine'})
    return rows


def remaining_startup_time(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise SetupError('Engine startup timed out before profile activation was confirmed')
    return remaining


def check_engine(base, model, profile, key_file, api_key=None, *, deadline=None):
    """Two one-token requests; responses are discarded, never logged."""
    try:
        key = read_key(key_file)
    except (OSError, ValueError) as error:
        raise SetupError('The engine signing key must be an owned 0600 file containing 32 raw bytes') from error
    body = {'model': model, 'messages': [{'role': 'user', 'content': 'Reply OK.'}],
            'max_tokens': 1, 'stream': False}
    def timeout():
        return 15 if deadline is None else remaining_startup_time(deadline)
    status, headers, _ = request(base, '/chat/completions', payload={**body, 'owner_mode': 'off'},
                                 api_key=api_key, timeout=timeout())
    if status != 200 or headers.get('X-Owner-Mode-Effective') != 'off':
        raise SetupError('The engine did not confirm standard mode; check the URL and engine support')
    status, headers, _ = request(base, '/chat/completions',
        payload={**body, 'owner_mode': profile, 'owner_mode_scale': 1.0}, key=key,
        api_key=api_key, timeout=timeout())
    if deadline is not None:
        remaining_startup_time(deadline)
    if status != 200 or headers.get('X-Owner-Mode-Effective') != profile:
        raise SetupError('The engine did not activate this profile; load the matching profile and signing key in the engine first')
    return {'status': 'active', 'profile': profile, 'standard_mode': 'off',
            'scope': 'request authentication and effective profile; not numerical or behavioral validation'}


def register_openclaw(home, executable, package, *, runner=subprocess.run):
    """Record a local plugin installation through OpenClaw's native manager."""
    if not executable or not Path(executable).is_file():
        return {'status': 'pending', 'reason': 'OpenClaw executable unavailable'}
    env = os.environ.copy()
    env['OPENCLAW_STATE_DIR'] = str(Path(home)/'.openclaw')
    env['OPENCLAW_CONFIG_PATH'] = str(Path(home)/'.openclaw/openclaw.json')
    command = [str(executable), 'plugins', 'install', '--force', '--accept-capabilities', str(package)]
    try:
        result = runner(command, env=env, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        raise SetupError('OpenClaw plugin registration failed; gateway restart was not requested') from error
    if result.returncode:
        raise SetupError('OpenClaw plugin registration failed; gateway restart was not requested')
    return {'status': 'registered', 'package': str(package)}


def restart_harnesses(home, selected, found, *, runner=subprocess.run):
    results = {}
    for name in selected:
        executable = found[name].get('executable')
        command = [executable or name, 'gateway', 'restart']
        if not executable:
            results[name] = {'status': 'unavailable', 'command': command}
            continue
        env = os.environ.copy()
        if name == 'hermes':
            env['HERMES_HOME'] = str(Path(home) / '.hermes')
        else:
            env['OPENCLAW_STATE_DIR'] = str(Path(home) / '.openclaw')
            env['OPENCLAW_CONFIG_PATH'] = str(Path(home) / '.openclaw/openclaw.json')
        try:
            result = runner(command, env=env, capture_output=True, text=True, timeout=45)
            results[name] = {'status': 'restarted' if result.returncode == 0 else 'failed', 'command': command}
        except (OSError, subprocess.TimeoutExpired):
            results[name] = {'status': 'failed', 'command': command}
        # CLI output may contain credentials or contact information. Never relay it.
    return results


class Console(TerminalUI):
    """Shared FlashAlign terminal presentation with the setup's existing API."""
