# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Read local model metadata without loading weights or importing an engine."""
from pathlib import Path
import copy
import hashlib
import json
import math
import re
import struct

from .profiles import read_json
from .setup_runtime import SetupError


OPENCLAW_THINKING_FORMATS = frozenset({
    'openai', 'openrouter', 'zai', 'deepseek', 'together', 'qwen', 'qwen-chat-template',
})
OPENCLAW_REASONING_EFFORTS = frozenset({'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'})


def _checked_openclaw_compat(value):
    if (not isinstance(value, dict) or 'thinkingFormat' not in value
            or set(value) - {'thinkingFormat', 'supportedReasoningEfforts'}
            or not isinstance(value.get('thinkingFormat'), str)
            or value.get('thinkingFormat') not in OPENCLAW_THINKING_FORMATS):
        raise SetupError('Unsupported OpenClaw thinking compatibility declaration')
    if 'supportedReasoningEfforts' in value:
        efforts = value['supportedReasoningEfforts']
        if (not isinstance(efforts, list) or not efforts
                or any(not isinstance(item, str) or item not in OPENCLAW_REASONING_EFFORTS for item in efforts)
                or len(set(efforts)) != len(efforts)):
            raise SetupError('Unsupported OpenClaw reasoning effort declaration')
    return copy.deepcopy(value)


def declared_openclaw_compat(metadata, model_type):
    """Use only an explicit declaration for the validated serving model type."""
    if 'openclaw_compat_by_model_type' not in metadata:
        return None
    declarations = metadata['openclaw_compat_by_model_type']
    supported = metadata.get('model_types', metadata.get('model_architectures', []))
    if (not isinstance(declarations, dict) or not isinstance(supported, list)
            or any(not isinstance(key, str) or key not in supported for key in declarations)):
        raise SetupError('Invalid model-scoped OpenClaw compatibility declaration')
    checked = {}
    for key, value in declarations.items():
        checked[key] = _checked_openclaw_compat(value)
    return checked.get(model_type)


def checked_request_compatibility(value, *, require_model=True):
    if require_model and (value is None or value == 'deepseek_v41'):
        return value
    if isinstance(value, dict) and value.get('kind') == 'gemma_boolean_thinking_v1':
        required = {'kind'} | ({'model'} if require_model else set())
        if (set(value) != required
                or require_model and (not isinstance(value.get('model'), str) or not value['model'])):
            raise SetupError('Unsupported Gemma boolean thinking declaration')
        return copy.deepcopy(value)
    required = {'kind', 'efforts'} | ({'model'} if require_model else set())
    if (not isinstance(value, dict) or set(value) != required
            or value.get('kind') != 'chat_template_reasoning_v1'
            or require_model and (not isinstance(value.get('model'), str) or not value['model'])):
        raise SetupError('Unsupported model request compatibility rule')
    efforts = value['efforts']
    if (not isinstance(efforts, list) or not efforts
            or any(not isinstance(item, str) or item not in OPENCLAW_REASONING_EFFORTS - {'none'} for item in efforts)
            or len(set(efforts)) != len(efforts)):
        raise SetupError('Unsupported template reasoning effort declaration')
    return copy.deepcopy(value)


def declared_request_compatibility(metadata, model_type):
    if 'request_compatibility_by_model_type' not in metadata:
        return None
    declarations = metadata['request_compatibility_by_model_type']
    supported = metadata.get('model_types', metadata.get('model_architectures', []))
    if (not isinstance(declarations, dict) or not isinstance(supported, list)
            or any(not isinstance(key, str) or key not in supported for key in declarations)):
        raise SetupError('Invalid model-scoped request compatibility declaration')
    checked = {key: checked_request_compatibility(value, require_model=False)
               for key, value in declarations.items()}
    return checked.get(model_type)


def checked_openclaw_request_policy(value):
    if value is None:
        return None
    if (not isinstance(value, dict) or not value
            or set(value) - {'output_budget', 'supports_thinking_off'}
            or 'output_budget' in value and value['output_budget'] != 'remaining_context'
            or 'supports_thinking_off' in value and type(value['supports_thinking_off']) is not bool):
        raise SetupError('Unsupported OpenClaw request policy')
    return copy.deepcopy(value)


def declared_openclaw_request_policy(metadata, model_type):
    if 'openclaw_request_policy_by_model_type' not in metadata:
        return None
    declarations = metadata['openclaw_request_policy_by_model_type']
    supported = metadata.get('model_types', metadata.get('model_architectures', []))
    if (not isinstance(declarations, dict) or not isinstance(supported, list)
            or any(not isinstance(key, str) or key not in supported for key in declarations)):
        raise SetupError('Invalid model-scoped OpenClaw request policy')
    if any(value is None for value in declarations.values()):
        raise SetupError('Invalid model-scoped OpenClaw request policy')
    checked = {key: checked_openclaw_request_policy(value) for key, value in declarations.items()}
    return checked.get(model_type)


def check_openclaw_thinking_selection(config, model, policy):
    """Reject an unsupported saved choice without replacing a user preference."""
    if not policy or policy.get('supports_thinking_off') is not False:
        return
    agents = config.get('agents', {})
    if not isinstance(agents, dict):
        raise SetupError('OpenClaw agents must be an object')
    defaults = agents.get('defaults', {})
    if not isinstance(defaults, dict):
        raise SetupError('OpenClaw agent defaults must be an object')
    key = 'owner-mode/'+model
    def setting(entry):
        models = entry.get('models', {})
        row = models.get(key, {}) if isinstance(models, dict) else {}
        params = row.get('params', {}) if isinstance(row, dict) else {}
        return params.get('thinking') if isinstance(params, dict) else None
    def disabled(value):
        return value is False or isinstance(value, str) and value.strip().lower() in {'off','none','disabled'}
    default = setting(defaults)
    if default is None:
        default = defaults.get('thinkingDefault')
    entries = agents.get('entries', agents.get('list', {}))
    values = entries.values() if isinstance(entries, dict) else entries if isinstance(entries, list) else []
    choices = [default]
    for entry in values:
        if not isinstance(entry, dict):
            continue
        selected = entry.get('model')
        selected = selected.get('primary') if isinstance(selected, dict) else selected
        if selected and selected != key:
            continue
        effective = entry.get('thinkingDefault')
        if effective is None:
            effective = setting(entry)
        choices.append(default if effective is None else effective)
    if any(disabled(value) for value in choices):
        raise SetupError('The selected model does not support thinking off; change the explicit OpenClaw choice before setup')


def check_openclaw_request_config(config, model, policy):
    check_openclaw_thinking_selection(config, model, policy)
    if not policy or policy.get('output_budget') != 'remaining_context':
        return
    workers = config.get('cloudWorkers', {})
    if not isinstance(workers, dict):
        return
    profiles = workers.get('profiles', {})
    selected = profiles.get(workers.get('requiredProfile')) if isinstance(profiles, dict) else None
    if (isinstance(selected, dict) and selected.get('provider') == 'device'
            and isinstance(selected.get('settings'), dict) and selected['settings'].get('inference') == 'worker'):
        raise SetupError('Remaining-context output is not supported by the required OpenClaw worker-local inference profile')


def check_openclaw_request_file(path, model, policy):
    """Preflight the same JSON/JSON5 config accepted by the setup writer."""
    if not policy:
        return
    path = Path(path)
    if not path.exists():
        return
    text = path.read_text()
    try:
        config = json.loads(text)
    except ValueError:
        try:
            import json5
            config = json5.loads(text)
        except (ImportError, ValueError) as error:
            raise SetupError('Cannot read OpenClaw request configuration') from error
    if not isinstance(config, dict):
        raise SetupError('OpenClaw configuration must be an object')
    check_openclaw_request_config(config, model, policy)


def _explicit_openclaw_compat(config, model):
    models = config.get('models', {})
    if not isinstance(models, dict) or not isinstance(models.get('providers', {}), dict):
        raise SetupError('OpenClaw model providers must be an object')
    providers = models.get('providers', {})
    agents = config.get('agents', {})
    defaults = agents.get('defaults', {}) if isinstance(agents, dict) else None
    if not isinstance(defaults, dict):
        raise SetupError('OpenClaw agent defaults must be an object')
    primary = defaults.get('model')
    primary = primary.get('primary') if isinstance(primary, dict) else primary
    selected = None
    if isinstance(primary, str):
        selected, separator, selected_model = primary.partition('/')
        if not separator or selected_model != model:
            selected = None
    groups = []
    for name, provider in providers.items():
        if not isinstance(provider, dict) or not isinstance(provider.get('models'), list):
            continue
        values = []
        for row in provider['models']:
            if not isinstance(row, dict) or row.get('id') != model or 'compat' not in row:
                continue
            if not isinstance(row['compat'], dict):
                raise SetupError('Explicit OpenClaw model compat must be an object')
            values.append(row['compat'])
        if values:
            groups.append((name, values))
    # The currently selected canonical route wins. On a rerun, this naturally
    # selects owner-mode; unrelated providers cannot replace that explicit choice.
    candidates = next((values for name, values in groups if name == selected), None)
    if candidates is None:
        candidates = [value for _, values in groups for value in values]
    if not candidates:
        return None
    if any(value != candidates[0] for value in candidates[1:]):
        raise SetupError('Conflicting explicit compatibility for the selected OpenClaw model')
    return copy.deepcopy(candidates[0])


def configure_openclaw_compat(config, model, descriptor, model_info=None):
    """Preserve explicit canonical compat before constructing owner-run aliases."""
    result = copy.deepcopy(descriptor)
    explicit = _explicit_openclaw_compat(config, model)
    if explicit is not None:
        result['compat'] = explicit
        return result, {'source': 'explicit canonical model', 'classified': True}
    declared = (model_info or {}).get('openclaw_compat')
    if declared is None:
        return result, {'source': 'undeclared serving lane', 'classified': False}
    result['compat'] = _checked_openclaw_compat(declared)
    return result, {'source': 'validated serving model type', 'classified': True,
                    'thinkingFormat': declared['thinkingFormat']}


def _portable_generation_settings(extra):
    """Copy request-generation options, excluding provider transport and identity."""
    result = {}
    numeric = {'temperature', 'top_p', 'top_k', 'min_p', 'max_tokens', 'max_completion_tokens',
               'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'length_penalty',
               'seed', 'min_tokens', 'n', 'top_logprobs'}
    for key in numeric:
        value = extra.get(key)
        if type(value) in (int, float) and math.isfinite(value):
            result[key] = value
    for key in ('ignore_eos', 'logprobs'):
        if type(extra.get(key)) is bool:
            result[key] = extra[key]
    if isinstance(extra.get('reasoning_effort'), str) or type(extra.get('reasoning_effort')) is int:
        result['reasoning_effort'] = extra['reasoning_effort']
    stop = extra.get('stop')
    if isinstance(stop, str) or isinstance(stop, list) and all(isinstance(item, str) for item in stop):
        result['stop'] = copy.deepcopy(stop)
    ids = extra.get('stop_token_ids')
    if isinstance(ids, list) and all(type(item) is int for item in ids):
        result['stop_token_ids'] = ids.copy()
    options = extra.get('chat_template_kwargs')
    if isinstance(options, dict):
        safe = {key: options[key] for key in ('enable_thinking', 'preserve_thinking') if type(options.get(key)) is bool}
        for key in ('reasoning_effort', 'thinking_budget'):
            if isinstance(options.get(key), str) or type(options.get(key)) is int:
                safe[key] = options[key]
        if safe:
            result['chat_template_kwargs'] = safe
    response = extra.get('response_format')
    if isinstance(response, dict) and response.get('type') in ('text', 'json_object', 'json_schema'):
        safe = {'type': response['type']}
        schema = response.get('json_schema')
        if response['type'] == 'json_schema' and isinstance(schema, dict):
            safe['json_schema'] = {key: copy.deepcopy(schema[key]) for key in ('name', 'description', 'schema', 'strict') if key in schema}
        result['response_format'] = safe
    return result


def configure_hermes_provider(config, model, base_url, modalities=None):
    """Select a dedicated local provider without editing the user's old provider."""
    model_config = config.setdefault('model', {})
    providers = config.setdefault('providers', {})
    if not isinstance(model_config, dict) or not isinstance(providers, dict):
        raise SetupError('Hermes model and providers configuration must be objects')
    previous = model_config.get('provider')
    name = previous.split(':', 1)[-1] if isinstance(previous, str) else ''
    old = providers.get(name, {})
    if not isinstance(old, dict):
        old = {}
    settings = config.get('plugins', {}).get('entries', {}).get('owner-mode', {}).get('settings', {})
    managed = settings.get('managed_provider') if isinstance(settings, dict) else None
    expected = 'http://127.0.0.1:'+str(settings.get('port'))+'/head/v1' if isinstance(settings, dict) else ''
    reusable = (isinstance(managed, str) and re.fullmatch(r'flashalign(?:-[0-9]+)?', managed)
                and name == managed and old.get('api_key') == 'flashalign-local'
                and old.get('base_url') == expected)
    if reusable:
        selected = managed
    else:
        selected = 'flashalign'
        number = 2
        while selected in providers:
            selected = 'flashalign-'+str(number)
            number += 1
            if number > 10000:
                raise SetupError('No free FlashAlign provider name')
    provider = {'base_url': base_url, 'default_model': model,
                'api_key': 'flashalign-local', 'api_mode': 'chat_completions'}
    extra = old.get('extra_body')
    if isinstance(extra, dict):
        provider['extra_body'] = _portable_generation_settings(extra)
    if modalities is not None:
        provider['models'] = {model: {'supports_vision': 'image' in modalities}}
    providers[selected] = provider
    model_config.update(provider=selected, default=model, base_url=base_url)
    return {'name': selected, 'previous_provider_preserved': not reusable,
            'generation_settings_preserved': isinstance(extra, dict),
            'generation_fields': sorted(provider.get('extra_body', {})),
            'omitted_provider_field_count': len(set(extra)-set(provider.get('extra_body', {}))) if isinstance(extra, dict) else 0,
            'scope': 'main model through the local FlashAlign proxy'}


def configure_hermes_images(config, modalities):
    """Bind native image routing only to declared serving-lane capabilities."""
    if modalities is None:
        return {'status': 'unverified', 'native_images': None,
                'reason': 'The serving lane did not declare its input modalities'}
    if (not isinstance(modalities, list) or 'text' not in modalities
            or any(value not in ('text', 'image') for value in modalities)):
        raise SetupError('Unsupported serving-lane input modalities')
    agent = config.setdefault('agent', {})
    model = config.setdefault('model', {})
    if not isinstance(agent, dict) or not isinstance(model, dict):
        raise SetupError('Hermes agent and model configuration must be objects')
    mode = agent.get('image_input_mode', 'auto')
    if mode not in ('auto', 'native_if_supported', 'native', 'text'):
        raise SetupError('Hermes image_input_mode is not supported by this setup')
    vision = 'image' in modalities
    if mode == 'native' and not vision:
        raise SetupError('Hermes explicitly requires native images but the selected serving lane is text-only')
    agent.setdefault('image_input_mode', 'auto')
    model['supports_vision'] = vision
    auxiliary_config = config.get('auxiliary', {})
    if not isinstance(auxiliary_config, dict):
        raise SetupError('Hermes auxiliary configuration must be an object')
    auxiliary = auxiliary_config.get('vision', {})
    explicit_auxiliary = isinstance(auxiliary, dict) and bool(auxiliary) and not (
        str(auxiliary.get('provider') or '').strip().lower() in ('', 'auto')
        and not auxiliary.get('model') and not auxiliary.get('base_url'))
    native = vision and mode != 'text' and not (mode == 'auto' and explicit_auxiliary)
    result = {'status': 'configured', 'modalities': modalities, 'image_input_mode': mode,
              'native_images': native, 'capability_field': 'model.supports_vision'}
    if mode == 'text' or (mode == 'auto' and explicit_auxiliary):
        result['status'] = 'preserved explicit image route'
    return result


def configure_hermes_reasoning(config, model):
    """Use Hermes' per-model override for the native DeepSeek effort vocabulary."""
    if model != 'deepseek-ai/DeepSeek-V4.1-Flash-1017':
        return None
    agent = config.get('agent', {})
    if not isinstance(agent, dict):
        raise SetupError('Hermes agent configuration must be an object')
    overrides = agent.get('reasoning_overrides', {})
    if not isinstance(overrides, dict):
        raise SetupError('Hermes agent.reasoning_overrides must be an object')
    bare = re.sub(r'[^a-z0-9]', '', model.rsplit('/', 1)[-1].lower())
    aliases = [key for key in overrides if isinstance(key, str) and key != model
               and re.sub(r'[^a-z0-9]', '', key.rsplit('/', 1)[-1].lower()) == bare]
    candidates = [('model override', overrides[model])] if model in overrides else []
    candidates += [('model alias override', overrides[key]) for key in aliases]
    candidates += [('global setting', agent.get('reasoning_effort'))]
    effective, source = None, 'provider default'
    known = {'none', 'false', 'disabled', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'}
    for origin, value in candidates:
        if value is False:
            effective, source = 'none', origin
            break
        if isinstance(value, str) and value.strip().lower() in known:
            effective, source = value.strip().lower(), origin
            break
    if effective in {'none', 'false', 'disabled', 'low', 'high', 'xhigh', 'max'}:
        return {'status': 'preserved', 'source': source, 'effort': effective}
    config.setdefault('agent', {})
    config['agent'].setdefault('reasoning_overrides', {})[model] = 'low'
    return {'status': 'adjusted', 'source': source, 'effort': 'low',
            'field': 'agent.reasoning_overrides.'+model,
            'reason': 'The native DeepSeek engine does not accept the generic medium/minimal/ultra effort defaults'}


def compatible_request(body, compatibility=None):
    """Translate only declared model-specific transport spellings, before signing."""
    compatibility = checked_request_compatibility(compatibility)
    if compatibility is None:
        return body
    if isinstance(compatibility, dict):
        if body.get('model') != compatibility['model'] or body.get('reasoning_effort') is None:
            return body
        effort = body['reasoning_effort']
        if compatibility['kind'] == 'gemma_boolean_thinking_v1':
            if effort != 'none':
                return body
            options = body.get('chat_template_kwargs', {})
            if not isinstance(options, dict):
                raise SetupError('Gemma chat_template_kwargs must be an object')
            # Hermes's one-shot recovery disable must override inherited
            # template defaults, or Gemma keeps generating hidden reasoning.
            result = {key: value for key, value in body.items() if key != 'reasoning_effort'}
            result['chat_template_kwargs'] = {**options, 'enable_thinking': False}
            return result
        if effort != 'none' and effort not in compatibility['efforts']:
            raise SetupError('Unsupported reasoning effort for the selected template')
        options = body.get('chat_template_kwargs', {})
        if not isinstance(options, dict):
            raise SetupError('chat_template_kwargs must be an object')
        options = dict(options)
        # Native template kwargs take precedence over top-level defaults.
        options.setdefault('enable_thinking', effort != 'none')
        if effort != 'none':
            options.setdefault('reasoning_effort', effort)
        result = {key: value for key, value in body.items() if key != 'reasoning_effort'}
        result['chat_template_kwargs'] = options
        return result
    if compatibility != 'deepseek_v41':
        raise SetupError('Unsupported model request compatibility rule')
    if body.get('model') != 'deepseek-ai/DeepSeek-V4.1-Flash-1017' or body.get('reasoning_effort') != 'none':
        return body
    options = body.get('chat_template_kwargs', {})
    if not isinstance(options, dict):
        raise SetupError('DeepSeek chat_template_kwargs must be an object')
    result = {key: value for key, value in body.items() if key != 'reasoning_effort'}
    result['chat_template_kwargs'] = {**options, 'enable_thinking': False}
    return result


def safetensors_header_fingerprint(path):
    """Fingerprint a single-shard layout without reading its weight payload."""
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SetupError('Truncated safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 64*1024*1024 or length+8 > size:
            raise SetupError('Invalid or oversized safetensors header')
        header = stream.read(length)
        if len(header) != length:
            raise SetupError('Truncated safetensors header')
    return {'file': path.name, 'sha256': hashlib.sha256(header).hexdigest(), 'size': size}


def gguf_metadata(path, *, fingerprint=False):
    """Read the few GGUF header fields needed for compatibility, with bounded reads."""
    path = Path(path)
    before = path.stat()
    size = before.st_size
    result = {}
    scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
              7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    with path.open('rb') as stream:
        def read(count):
            if count < 0 or stream.tell()+count > min(size, 512*1024*1024):
                raise SetupError('Invalid or oversized GGUF metadata')
            data = stream.read(count)
            if len(data) != count:
                raise SetupError('Truncated GGUF metadata')
            return data

        def number(fmt):
            return struct.unpack('<'+fmt, read(struct.calcsize('<'+fmt)))[0]

        def string(keep):
            length = number('Q')
            if length > 64*1024*1024:
                raise SetupError('Oversized GGUF metadata string')
            if keep:
                return read(length).decode('utf-8')
            if stream.tell()+length > min(size, 512*1024*1024):
                raise SetupError('Invalid GGUF string bounds')
            stream.seek(length, 1)

        def value(kind, keep=False, depth=0):
            if kind == 8:
                return string(keep)
            if kind == 9:
                if depth:
                    raise SetupError('Nested GGUF metadata arrays are unsupported')
                element, count = number('I'), number('Q')
                if count > 10_000_000:
                    raise SetupError('Oversized GGUF metadata array')
                if element in scalar:
                    length = struct.calcsize('<'+scalar[element])*count
                    if stream.tell()+length > min(size, 512*1024*1024):
                        raise SetupError('Invalid GGUF array bounds')
                    stream.seek(length, 1)
                else:
                    for _ in range(count):
                        value(element, False, depth+1)
                return None
            if kind not in scalar:
                raise SetupError('Unsupported GGUF metadata type')
            item = number(scalar[kind])
            return item if keep else None

        if read(4) != b'GGUF' or number('I') not in (2, 3):
            raise SetupError('Select a GGUF version 2 or 3 model')
        tensors = number('Q')
        if tensors > 1_000_000:
            raise SetupError('Oversized GGUF tensor table')
        count = number('Q')
        if count > 65536:
            raise SetupError('Oversized GGUF metadata table')
        for _ in range(count):
            name = string(True)
            keep = name in ('general.architecture', 'general.name') or name.endswith(
                ('.embedding_length', '.block_count', '.context_length'))
            item = value(number('I'), keep)
            if keep:
                if name in result:
                    raise SetupError('Duplicate GGUF compatibility field')
                result[name] = item
        if fingerprint:
            # Bind metadata and tensor names/shapes/types without reading model
            # payloads. The boundary deliberately excludes alignment padding.
            for _ in range(tensors):
                string(False)
                dimensions = number('I')
                if not 1 <= dimensions <= 4:
                    raise SetupError('Invalid GGUF tensor dimensions')
                for _ in range(dimensions):
                    if number('Q') < 1:
                        raise SetupError('Invalid GGUF tensor shape')
                number('I')  # Quantized tensor type.
                number('Q')  # Offset relative to the aligned payload start.
            header_bytes = stream.tell()
            stream.seek(0)
            digest = hashlib.sha256()
            remaining = header_bytes
            while remaining:
                chunk = read(min(1024*1024, remaining))
                digest.update(chunk)
                remaining -= len(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise SetupError('GGUF model changed during inspection')
            header = {'sha256': digest.hexdigest(), 'size': size, 'header_bytes': header_bytes}
    architecture = result.get('general.architecture')
    if not isinstance(architecture, str) or not architecture:
        raise SetupError('GGUF model does not declare its architecture')
    return (result, header) if fingerprint else result


def inspect_model(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == '.gguf':
        metadata, header = gguf_metadata(path, fingerprint=True)
        architecture = metadata['general.architecture']
        return {'path': str(path), 'format': 'GGUF', 'gguf_architecture': architecture,
                'model_type': {'qwen35': 'qwen3_5'}.get(architecture, architecture),
                'hidden_size': metadata.get(architecture+'.embedding_length'),
                'num_hidden_layers': metadata.get(architecture+'.block_count'),
                'metadata': metadata, 'gguf_header': header}
    if not (path/'config.json').is_file():
        raise SetupError('Select a local model folder containing config.json or a GGUF file')
    config = read_json(path/'config.json')
    if not isinstance(config, dict):
        raise SetupError('Invalid local model configuration')
    text = config.get('text_config') or config
    return {**config, 'path': str(path), 'format': 'checkpoint',
            'hidden_size': text.get('hidden_size'),
            'num_hidden_layers': text.get('num_hidden_layers')}
