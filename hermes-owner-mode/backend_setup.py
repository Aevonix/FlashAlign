# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Stage pinned engine integrations in a deployment owned by FlashAlign."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from .auth import read_key, ensure_key
from .drafting import MissingDraftModel, argument, draft_settings
from .engine_setup import available_devices, digest, prepare_python_headers, private_json, unused_port
from .model_metadata import (inspect_model, safetensors_header_fingerprint,
                             declared_openclaw_compat, declared_request_compatibility,
                             declared_openclaw_request_policy)
from .profiles import DEEPSEEK_GROUP_SCHEMA, match_profile, read_json, validate_profile, stage_profile
from .setup_runtime import (SetupError, managed_context_window, normalize_engine_arguments,
                            require_full_vllm_options)


KINDS = ('tensorfold', 'vllm', 'sglang', 'llamacpp')
KIT_ROOT = Path(__file__).resolve().parents[1]/'engines'


def _kits():
    for path in sorted(KIT_ROOT.glob('*/backend.json')):
        if not path.resolve().is_relative_to(KIT_ROOT.resolve()):
            continue
        metadata = read_json(path)
        kind = metadata.get('backend', metadata.get('engine'))
        if kind in KINDS[1:]:
            yield path.parent, metadata, kind


def backend_catalog():
    return [{'kind': kind, 'variant': metadata.get('variant'), 'status': metadata['status'],
             'model_types': metadata.get('model_types', metadata.get('model_architectures')),
             'model_format': metadata.get('model_format', 'checkpoint'), 'kit': str(path),
             **{key: metadata[key] for key in ('tensor_parallel_sizes', 'tested_speculative_methods', 'resource_policy', 'python_min_version', 'input_modalities', 'input_modalities_by_model_type', 'openclaw_compat_by_model_type', 'request_compatibility_by_model_type', 'openclaw_request_policy_by_model_type')
                if key in metadata},
             'full_payload_verification': bool(metadata.get('model_integrity_verifier'))}
            for path, metadata, kind in _kits()]


def _python_version(executable):
    """Inspect an interpreter without importing the engine or user site packages."""
    try:
        result = subprocess.run([executable, '-I', '-c',
            'import json,sys; print(json.dumps(list(sys.version_info[:3])))'],
            capture_output=True, text=True, check=True, timeout=5)
        value = json.loads(result.stdout)
        if (isinstance(value, list) and len(value) == 3
                and all(type(part) is int and part >= 0 for part in value)):
            return tuple(value)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def select_backend_python(spec, metadata):
    """Preserve explicit choices; auto-detect a compatible local interpreter."""
    explicit = spec.get('python')
    if explicit is not None and (not isinstance(explicit, str) or not explicit.strip()):
        raise SetupError('Engine python must name an interpreter')
    minimum = metadata.get('python_min_version')
    default = str(explicit or sys._base_executable)
    if minimum is None:
        return {'executable': default, 'version': None, 'minimum': None, 'automatic': explicit is None}
    if (not isinstance(minimum, list) or len(minimum) != 2
            or any(type(part) is not int or part < 0 for part in minimum) or minimum[0] != 3):
        raise SetupError('Engine integration has an invalid Python requirement')
    candidates = [default]
    if explicit is None:
        candidates += [sys.executable]
        candidates += [shutil.which('python'+str(minimum[0])+'.'+str(minor))
                       for minor in range(minimum[1], max(minimum[1], 15)+1)]
        candidates += [shutil.which('python3')]
    for executable in dict.fromkeys(candidate for candidate in candidates if candidate):
        version = _python_version(executable)
        if version is not None and version[0] == 3 and version[:2] >= tuple(minimum):
            return {'executable': executable, 'version': list(version), 'minimum': minimum,
                    'automatic': explicit is None}
    required = '.'.join(map(str, minimum))
    if explicit is not None:
        raise SetupError(f'This engine requires Python {required}+; the selected engine.python is incompatible or unavailable')
    raise SetupError(f'This engine requires Python {required}+; install a compatible local interpreter or set engine.python')


def _select_kit(kind, model, profile, variant=None):
    family = model.get('gguf_architecture') if kind == 'llamacpp' else model.get('model_type')
    matches = [(path, metadata) for path, metadata, backend in _kits()
               if backend == kind and family in metadata.get('model_types', metadata.get('model_architectures', []))
               and profile['transform'] in metadata.get('transforms', metadata.get('profile_transforms', []))
               and (profile.get('schema') or 'standard') in metadata.get('profile_schemas', ['standard'])
               and (variant is None or variant in (metadata.get('variant'), path.name))]
    if len(matches) != 1:
        raise SetupError('No unique engine integration supports this model family and profile transform')
    return matches[0]


def _input_modalities(metadata, model_type):
    """Use explicit backend capabilities for this type, never infer from its name."""
    def validate(value):
        if (not isinstance(value, list) or 'text' not in value
                or any(not isinstance(item, str) or item not in ('text', 'image') for item in value)
                or len(set(value)) != len(value)):
            raise SetupError('Invalid engine input modality declaration')
        return value.copy()

    if 'input_modalities_by_model_type' in metadata:
        declarations = metadata['input_modalities_by_model_type']
        supported = metadata.get('model_types', metadata.get('model_architectures', []))
        if (not isinstance(declarations, dict) or not declarations
                or not isinstance(supported, list) or any(not isinstance(item, str) for item in supported)
                or any(not isinstance(key, str) or not key or key not in supported for key in declarations)):
            raise SetupError('Invalid engine input_modalities_by_model_type declaration')
        checked = {key: validate(value) for key, value in declarations.items()}
        # A typed declaration is authoritative. Missing types remain unknown,
        # even when an older general capability declaration is also present.
        return checked.get(model_type)
    value = metadata.get('input_modalities', metadata.get('tested_modalities'))
    return None if value is None else validate(value)


def _model_matches(path, quantization, candidates):
    """Compare pinned metadata, never hash or load the full checkpoint here."""
    common = {'config_sha256': digest(path/'config.json'),
              'quantization_config_sha256': hashlib.sha256(json.dumps(
                  quantization or {}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    matched = []
    for entry in candidates:
        if (entry.get('config_sha256') != common['config_sha256'] or
                entry.get('quantization_config_sha256') is not None and
                entry['quantization_config_sha256'] != common['quantization_config_sha256']):
            continue
        if entry.get('index_sha256'):
            index = path/'model.safetensors.index.json'
            if index.is_file() and digest(index) == entry['index_sha256']:
                matched.append(entry)
        elif isinstance(entry.get('weights_header'), dict):
            header = entry['weights_header']
            name = header.get('file', '')
            if not name or Path(name).name != name or not name.endswith('.safetensors'):
                raise SetupError('Invalid pinned checkpoint filename')
            weights = path/name
            if weights.is_file() and safetensors_header_fingerprint(weights) == header:
                matched.append(entry)
    return matched


def _check_gguf_binding(metadata, model, manifest):
    """Match an explicitly qualified GGUF file to its compatible profile model."""
    required = metadata.get('gguf_required_model_architectures', [])
    architectures = metadata['model_architectures']
    if (not isinstance(required, list)
            or any(not isinstance(value, str) or value not in architectures for value in required)
            or len(set(required)) != len(required)):
        raise SetupError('Invalid required GGUF architecture declaration')
    bindings = metadata.get('gguf_models', [])
    if not isinstance(bindings, list):
        raise SetupError('Invalid GGUF checkpoint declarations')
    pins = []
    for row in bindings:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']
                or not isinstance(row.get('revision'), str) or not re.fullmatch('[0-9a-f]{40}', row['revision'])
                or row.get('architecture') not in architectures
                or 'profile_name' in row and (not isinstance(row['profile_name'], str) or not row['profile_name'])
                or not isinstance(row.get('header_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['header_sha256'])
                or type(row.get('size')) is not int or row['size'] <= 0
                or not isinstance(row.get('profile_model'), dict) or set(row['profile_model']) != {'id', 'revision'}
                or not isinstance(row['profile_model'].get('id'), str) or not row['profile_model']['id']
                or not isinstance(row['profile_model'].get('revision'), str)
                or not re.fullmatch('[0-9a-f]{40}', row['profile_model']['revision'])):
            raise SetupError('Invalid pinned GGUF checkpoint identity')
        if row['architecture'] == model['gguf_architecture']:
            pins.append(row)
    if not pins and model['gguf_architecture'] not in required:
        return None
    header = model['gguf_header']
    matched = [row for row in pins if row['header_sha256'] == header['sha256'] and row['size'] == header['size']]
    if len(matched) != 1:
        raise SetupError('GGUF checkpoint fingerprint does not match a qualified model for this architecture')
    pin = matched[0]
    source = manifest.get('source', {})
    source = source if isinstance(source, dict) else {}
    identities = [manifest['model'], *manifest.get('compatible_models', []),
                  source.get('base'), source.get('stock')]
    if not any(isinstance(identity, dict) and all(identity.get(key) == pin['profile_model'][key]
               for key in ('id', 'revision')) for identity in identities):
        raise SetupError('GGUF profile model or revision does not match the selected profile')
    return {key: pin[key] for key in ('id', 'revision', 'profile_model', 'header_sha256', 'size')}


def _check_llamacpp_strengths(metadata, sites):
    """Accept only strengths explicitly supported by the installed adapter."""
    rank_limit = metadata.get('maximum_rank', 128)
    if type(rank_limit) is not int or not 1 <= rank_limit <= 128:
        raise SetupError('Invalid llama.cpp profile rank declaration')
    if any(type(site.get('rank')) is not int or not 1 <= site['rank'] <= rank_limit for site in sites):
        raise SetupError('This llama.cpp integration requires profile rank at most '+str(rank_limit))
    mode = metadata.get('site_scale_mode')
    if mode is None:
        expected = metadata.get('site_scale', 1.0)
        if type(expected) not in (int, float) or not math.isfinite(expected) or expected <= 0:
            raise SetupError('Invalid llama.cpp site strength declaration')
        if any(site['scale'] != expected for site in sites):
            raise SetupError('This llama.cpp integration requires site strengths of '+str(expected))
    elif mode == 'positive_finite':
        for site in sites:
            scale = site.get('scale')
            valid = type(scale) in (int, float) and math.isfinite(scale) and scale > 0
            try:
                rounded = struct.unpack('<f', struct.pack('<f', scale))[0] if valid else 0
            except (OverflowError, struct.error):
                rounded = 0
            if not math.isfinite(rounded) or not 0 < rounded <= float.fromhex('0x1.fffffep+127')/2:
                raise SetupError('This llama.cpp integration requires finite positive FP32 site strengths at most half the FP32 maximum')
    else:
        raise SetupError('Unsupported llama.cpp site strength declaration')


def _without_options(arguments, names):
    """Keep managed launcher options out of a variant's upstream defaults."""
    result = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value.split('=', 1)[0] in names:
            index += 1 if '=' in value else 2
        else:
            result.append(value)
            index += 1
    return result


def _sglang_arguments(defaults, explicit):
    """Replace backend defaults while leaving duplicate caller options visible."""
    def canonical(arguments):
        result = []
        for value in arguments:
            option, separator, setting = value.partition('=')
            if value.startswith('--') and separator and not setting.startswith('--'):
                result.extend((option, setting))
            else:
                result.append(value)
        return result
    defaults, explicit = canonical(defaults), canonical(explicit)
    overridden = {value.split('=', 1)[0] for value in explicit
                  if value.startswith('--') and value != '--'}
    result, index = [], 0
    while index < len(defaults):
        value = defaults[index]
        index += 1
        if value.split('=', 1)[0] in overridden:
            if '=' not in value and index < len(defaults) and not defaults[index].startswith('--'):
                index += 1
        else:
            result.append(value)
    return [*result, *explicit]


def _physical_devices_by_process(proc_root):
    """Read a bounded GPU ownership snapshot; unknown ownership stays unknown."""
    if proc_root != Path('/proc'):
        return {}
    try:
        def query(fields, kind):
            result = subprocess.run(['nvidia-smi', '--query-'+kind+'='+fields,
                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
            if len(result.stdout) > 1024*1024:
                raise ValueError('GPU ownership report is too large')
            return [row.split(',') for row in result.stdout.splitlines() if row.strip()]
        indices = {}
        seen_indices = set()
        for index, uuid in query('index,uuid', 'gpu'):
            index, uuid = index.strip(), uuid.strip()
            if (not re.fullmatch(r'0|[1-9][0-9]*', index) or not uuid.startswith('GPU-')
                    or index in seen_indices or uuid in indices):
                return {}
            indices[uuid] = index
            seen_indices.add(index)
        direct = {}
        for pid, uuid in query('pid,gpu_uuid', 'compute-apps'):
            direct.setdefault(int(pid.strip()), set()).add(indices[uuid.strip()])
        result = {pid: set(values) for pid, values in direct.items()}
        for pid, values in direct.items():
            seen = {pid}
            for _ in range(256):
                try:
                    parent = int((proc_root/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    break
                if parent <= 1 or parent in seen:
                    break
                seen.add(parent)
                result.setdefault(parent, set()).update(values)
                pid = parent
        return result
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {}


def _disjoint_visible_devices(process, devices, placements):
    """Use only unambiguous host-index visibility to exclude another deployment."""
    if not devices or any(not re.fullmatch(r'0|[1-9][0-9]*', str(value)) for value in devices):
        return False
    try:
        # Read only the two visibility fields. Never return or log environment
        # contents; unreadable, remapped or unknown placements stay conservative.
        fields = {}
        for entry in (process/'environ').read_bytes().split(b'\0'):
            name, separator, value = entry.partition(b'=')
            if separator and name in (b'CUDA_VISIBLE_DEVICES', b'NVIDIA_VISIBLE_DEVICES'):
                if name in fields:
                    return False
                fields[name] = value.decode('ascii')
        if fields.get(b'NVIDIA_VISIBLE_DEVICES', 'all') != 'all':
            return False
        visible = fields.get(b'CUDA_VISIBLE_DEVICES', '').split(',')
        if not visible or any(not re.fullmatch(r'0|[1-9][0-9]*', value) for value in visible):
            return False
        # Confirm the complete declared placement using driver ownership, so
        # CUDA ordinal remapping or an incompletely started worker cannot hide
        # an overlapping deployment.
        if placements.get(int(process.name)) != set(visible):
            return False
        return set(visible).isdisjoint(map(str, devices))
    except (OSError, UnicodeError):
        return False


def active_speculation(model_path, model_id, proc_root=Path('/proc'), *, devices=None):
    """Inspect owned command lines only, retaining decoding settings but no secrets."""
    result = []
    placements = None
    for process in proc_root.glob('[0-9]*'):
        matched = False
        try:
            if process.stat().st_uid != os.getuid():
                continue
            argv = [part for part in (process/'cmdline').read_bytes().decode().split('\0') if part]
            names = [Path(part).name for part in argv]
            if any('sglang' in name for name in names):
                kind = 'sglang'
            elif any(name in ('vllm', 'flashalign_vllm', 'flashalign_vllm.cli',
                              'flashalign_deepseek', 'flashalign_deepseek.cli') for name in names):
                kind = 'vllm'
            elif any(name == 'llama-server' or name == 'llama-server.exe' for name in names):
                kind = 'llamacpp'
            elif 'tensorfold' in names and 'serve' in argv:
                kind = 'tensorfold'
            else:
                continue
            if kind != 'llamacpp':
                argv = normalize_engine_arguments(kind, argv)
            candidate = argument(argv, '--model-path') or argument(argv, '--model') or argument(argv, '-m')
            # Python's -m names the module, not the model.
            if candidate and candidate in ('vllm', 'flashalign_vllm', 'flashalign_deepseek', 'sglang.launch_server', 'tensorfold'):
                candidate = None
            if not candidate and 'serve' in argv:
                index = argv.index('serve')
                candidate = argv[index+1] if index+1 < len(argv) else None
            if not candidate:
                continue
            cwd = (process/'cwd').resolve()
            current = Path(candidate).expanduser()
            current = current if current.is_absolute() else cwd/current
            if current.resolve() != Path(model_path).resolve() and candidate != model_id:
                continue
            if devices is not None:
                if placements is None:
                    placements = _physical_devices_by_process(proc_root)
                # Separate TensorFold rank processes expose only their own
                # card. That cannot establish the placement of a whole TP job.
                complete_rank_scope = kind != 'tensorfold' or argument(argv, '--tp') == '1'
                if complete_rank_scope and _disjoint_visible_devices(process, devices, placements):
                    continue
            matched = True
            if kind == 'llamacpp':
                argv = normalize_engine_arguments(kind, argv)
            entry = {'kind': kind}
            if kind == 'vllm':
                require_full_vllm_options(kind, argv, {'--speculative-config'})
                encoded = argument(argv, '--speculative-config')
                if not encoded:
                    if any(name in ('flashalign_deepseek', 'flashalign_deepseek.cli') for name in names):
                        raise SetupError('The active native engine does not expose its speculative configuration')
                    continue
                entry['speculative_config'] = json.loads(encoded)
            elif kind == 'sglang':
                method = argument(argv, '--speculative-algorithm')
                if not method:
                    continue
                entry['draft_algorithm'] = method.upper()
                entry['draft_model'] = argument(argv, '--speculative-draft-model-path')
                extra = []
                for index, value in enumerate(argv):
                    flag = value.split('=', 1)[0]
                    if flag.startswith('--speculative-') and flag not in (
                            '--speculative-algorithm', '--speculative-draft-model-path'):
                        extra.append(value)
                        if '=' not in value and index+1 < len(argv) and not argv[index+1].startswith('--'):
                            extra.append(argv[index+1])
                entry['speculative_arguments'] = extra
            elif kind == 'llamacpp':
                draft_paths = [argument(argv, flag) for flag in ('--spec-draft-model', '--model-draft', '-md')]
                draft_paths = [value for value in draft_paths if value is not None]
                if len(set(draft_paths)) > 1:
                    raise SetupError('The active engine has conflicting draft model options; preserve its settings until resolved')
                draft = draft_paths[0] if draft_paths else None
                method = argument(argv, '--spec-type')
                if not draft and not method:
                    continue
                entry.update(draft_model=draft, draft_algorithm=method or 'draft-simple')
                extra = []
                for index, value in enumerate(argv):
                    flag = value.split('=', 1)[0]
                    if flag.startswith('--spec-') and flag not in ('--spec-type', '--spec-draft-model'):
                        extra.append(value)
                        if '=' not in value and index+1 < len(argv) and not argv[index+1].startswith('--'):
                            extra.append(argv[index+1])
                entry['speculative_arguments'] = extra
            else:
                if '--no-drafts' in argv:
                    continue
                entry['draft_model'] = argument(argv, '--drafter')
                entry['draft_algorithm'] = 'tensorfold'
            if entry.get('draft_model') and entry['draft_model'] not in ('auto', 'none'):
                draft = Path(entry['draft_model']).expanduser()
                entry['draft_model'] = str(draft if draft.is_absolute() else cwd/draft)
            if entry not in result:
                result.append(entry)
        except (OSError, UnicodeError, ValueError) as error:
            if matched:
                if kind == 'llamacpp' and isinstance(error, SetupError):
                    raise
                raise SetupError('The active speculative settings could not be read; they will not be replaced') from error
            continue
    if len(result) > 1:
        raise SetupError('Matching engines have different speculative settings; select one deployment before setup')
    return result[0] if result else None


def _bundled_draft_path(model_path, metadata):
    folder = metadata.get('default_draft_subdirectory')
    if (not isinstance(folder, str) or folder in ('', '.', '..')
            or any(separator in folder for separator in ('/', chr(92), chr(0)))):
        raise SetupError('Invalid bundled draft directory')
    root = Path(model_path).expanduser().resolve()
    try:
        target = (root/folder).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SetupError('The declared bundled draft directory is unavailable') from error
    if target == root or not target.is_relative_to(root) or not target.is_dir():
        raise SetupError('The bundled draft directory escapes the model folder')
    return target


def _llamacpp_speculative_capabilities(metadata, architecture):
    """Select only methods declared for this GGUF graph family."""
    declarations = metadata.get('speculative_types_by_model_architecture')
    if declarations is None:
        methods = metadata.get('speculative_types_qualified', ['draft-simple'])
    else:
        if (not isinstance(declarations, dict)
                or any(key not in metadata['model_architectures'] for key in declarations)):
            raise SetupError('Invalid model-scoped llama.cpp speculative declaration')
        methods = declarations.get(architecture, [])
    if (not isinstance(methods, list) or not methods
            or any(not isinstance(value, str) or not re.fullmatch('[a-z][a-z0-9-]*', value) for value in methods)
            or len(set(methods)) != len(methods)):
        raise SetupError('No valid llama.cpp speculative methods are declared for this model architecture')
    draftless = metadata.get('draftless_speculative_types', [])
    if (not isinstance(draftless, list)
            or any(not isinstance(value, str) or not re.fullmatch('[a-z][a-z0-9-]*', value) for value in draftless)
            or len(set(draftless)) != len(draftless)):
        raise SetupError('Invalid llama.cpp draftless speculative declaration')
    defaults = metadata.get('default_speculative_type_by_model_architecture', {})
    if (not isinstance(defaults, dict)
            or any(key not in metadata['model_architectures'] or not isinstance(value, str)
                   for key, value in defaults.items())):
        raise SetupError('Invalid llama.cpp default speculative declaration')
    default = defaults.get(architecture)
    if default is not None and default not in methods:
        raise SetupError('The default llama.cpp speculative method is not qualified for this model')
    return methods, draftless, default


def _speculation(home, spec, kind, model_id, *, metadata=None, proc_root=Path('/proc'), devices=None, model=None):
    metadata = metadata or {}
    active = active_speculation(spec['model_path'], model_id, proc_root, devices=devices)
    if active and active['kind'] != kind:
        raise SetupError('The active engine uses a different speculative backend; its decoding settings will not be replaced or disabled')
    result = dict(active or {})
    method_hint = None
    if kind == 'llamacpp':
        model = model or inspect_model(spec['model_path'])
        methods, draftless, default = _llamacpp_speculative_capabilities(metadata, model['gguf_architecture'])
        method_hint = spec.get('draft_algorithm', result.get('draft_algorithm'))
        if method_hint is None:
            method_hint = (next((value for value in methods if value not in draftless), None)
                           if spec.get('draft_model') or result.get('draft_model') else default)
        if 'speculative_arguments' in result:
            value = result['speculative_arguments']
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise SetupError('Speculative arguments must be a string array')
            result['speculative_arguments'] = normalize_engine_arguments(
                kind, value, llamacpp_method=method_hint,
                llamacpp_draft_model=bool(result.get('draft_model')))
    for field in ('draft_model', 'draft_algorithm', 'speculative_config', 'speculative_arguments'):
        if field in spec:
            value = spec[field]
            if field == 'draft_model':
                value = str(Path(value).expanduser().resolve())
            if kind == 'llamacpp' and field == 'speculative_arguments':
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise SetupError('Speculative arguments must be a string array')
                value = normalize_engine_arguments(kind, value, llamacpp_method=method_hint,
                                                  llamacpp_draft_model=bool(result.get('draft_model')))
            if active and field in active and value != active[field]:
                raise SetupError('The selected speculative settings differ from the active engine; preserve its existing settings')
            result[field] = value
    extra = result.get('speculative_arguments', [])
    if not isinstance(extra, list) or any(not isinstance(value, str) for value in extra):
        raise SetupError('Speculative arguments must be a string array')
    for value in extra:
        allowed_prefixes = ('--spec-',) if kind == 'llamacpp' else ('--speculative-', '--spec-draft-')
        if value.startswith('-') and (not value.startswith(allowed_prefixes)
                or value.split('=', 1)[0] in ('--speculative-algorithm', '--speculative-draft-model-path',
                                             '--spec-type', '--spec-draft-model', '--model-draft', '-md')):
            raise SetupError('Speculative arguments cannot override the managed algorithm or model')
    if kind == 'vllm':
        if metadata.get('requires_native_draft'):
            if extra:
                raise SetupError('This native vLLM engine requires all speculative settings in speculative_config')
            if 'draft_model' in result:
                raise SetupError('This vLLM integration uses the stock model native DSpark; external draft models are not qualified')
            default = dict(metadata['default_speculative_config'])
            default['model'] = str(Path(spec['model_path']).expanduser().resolve())
            config = result.get('speculative_config', default)
            if (not isinstance(config, dict) or set(config) != set(default)
                    or config.get('method') != default['method']
                    or type(config.get('num_speculative_tokens')) is not int
                    or config['num_speculative_tokens'] != default['num_speculative_tokens']
                    or not isinstance(config.get('model'), str)
                    or str(Path(config['model']).expanduser().resolve()) != default['model']):
                raise SetupError('This DeepSeek variant qualifies native DSpark5 with the pinned stock model; '
                                 'active speculative settings will not be replaced or disabled')
            result['speculative_config'] = config
            return {key: value for key, value in result.items() if key != 'kind'}
        if 'draft_model' in result:
            raise SetupError('Use engine.speculative_config for a vLLM draft model; only ngram has passed the launch checks')
        config = result.get('speculative_config', {'method': 'ngram', 'num_speculative_tokens': 3,
                                                  'prompt_lookup_min': 2, 'prompt_lookup_max': 3})
        if not isinstance(config, dict) or config.get('method') != 'ngram':
            raise SetupError('This vLLM setup has qualified ngram speculation only; the active method will not be disabled')
        allowed = {'method', 'num_speculative_tokens', 'prompt_lookup_min', 'prompt_lookup_max'}
        if set(config)-allowed or any(type(value) is not int or value <= 0 for key, value in config.items() if key != 'method'):
            raise SetupError('Unsupported ngram settings; the existing speculative configuration will not be replaced')
        if config.get('prompt_lookup_min', 1) > config.get('prompt_lookup_max', 3):
            raise SetupError('Ngram lookup minimum exceeds its maximum')
        result['speculative_config'] = config
    elif kind == 'sglang':
        methods = metadata.get('speculative_algorithms', ['DFLASH'])
        draftless = metadata.get('draftless_speculative_algorithms', [])
        if (not isinstance(draftless, list) or any(not isinstance(item, str) for item in draftless)
                or len(draftless) != len(set(draftless))
                or any(method != 'NGRAM' or method not in methods for method in draftless)):
            raise SetupError('Invalid SGLang draftless speculative capability')
        selected_method = result.get('draft_algorithm', methods[0])
        if not isinstance(selected_method, str):
            raise SetupError('A speculative algorithm must be a string')
        method = selected_method.upper()
        if method not in methods:
            raise SetupError('The active speculative method is unsupported by this SGLang integration; it will not be disabled')
        if method in draftless:
            if result.get('draft_model'):
                raise SetupError('NGRAM speculation does not use an external draft model')
            result['draft_algorithm'] = method
        elif ('native_mtp_writer_groups' in metadata.get('required_features', [])
                or method in metadata.get('native_speculative_algorithms', [])):
            model = str(Path(spec['model_path']).expanduser().resolve())
            if result.get('draft_model') and str(Path(result['draft_model']).resolve()) != model:
                raise SetupError('This integration requires the model native MTP; the active draft model will not be replaced')
            result.update(draft_model=model, draft_algorithm=method)
        else:
            manifest = {'requires_draft_model': True, 'min_parallel': 1,
                        'draft_model_architectures': ['DFlash2DraftModel', 'DFlashDraftModel']}
            if not result.get('draft_model') and metadata.get('default_draft_subdirectory'):
                result['draft_model'] = str(_bundled_draft_path(spec['model_path'], metadata))
            selected = draft_settings(home, {**spec, **result}, manifest, model_id, proc_root=Path('/nonexistent'))
            result.update(selected, draft_algorithm=method)
            if metadata.get('requires_bundled_draft'):
                expected = _bundled_draft_path(spec['model_path'], metadata)
                if Path(result['draft_model']).resolve() != expected:
                    raise SetupError('This engine requires its pinned bundled draft; the selected draft will not be replaced')
    elif kind == 'llamacpp':
        model = model or inspect_model(spec['model_path'])
        methods, draftless, default = _llamacpp_speculative_capabilities(metadata, model['gguf_architecture'])
        method = result.get('draft_algorithm')
        if method is None:
            method = (next((value for value in methods if value not in draftless), None)
                      if result.get('draft_model') else default)
            if result.get('draft_model') and method is None:
                raise SetupError('This llama.cpp model has no qualified method for an external draft model')
        if method is not None:
            if method not in methods:
                raise SetupError('The active speculative method is unsupported by this llama.cpp model integration; it will not be disabled')
            result['draft_algorithm'] = method
        if method in draftless:
            if result.get('draft_model'):
                raise SetupError('This llama.cpp speculative method does not use an external draft model')
            result.pop('draft_model', None)
        elif method is not None:
            if not result.get('draft_model'):
                raise SetupError('This llama.cpp speculative method requires a local GGUF draft model; it will not be disabled')
            draft = Path(result['draft_model'])
            if not draft.is_file() or draft.suffix.lower() != '.gguf':
                raise SetupError('llama.cpp requires an existing GGUF draft model file')
    return {key: value for key, value in result.items() if key != 'kind'}


def _native_profile_layers(kit, metadata, model, sites):
    """Validate explicit native-MTP writers without broadening model bounds."""
    native_layers = metadata.get('profile_native_mtp_layers', [])
    if not isinstance(native_layers, list):
        raise SetupError('Engine native MTP layer declaration must be an array')
    if native_layers:
        base_layers = model.get('num_hidden_layers')
        nextn = (model.get('text_config') or model).get('num_nextn_predict_layers')
        if (type(base_layers) is not int or base_layers <= 0 or type(nextn) is not int or nextn <= 0
                or not isinstance(native_layers, list)
                or any(type(layer) is not int for layer in native_layers)
                or len(set(native_layers)) != len(native_layers)
                or any(layer not in range(base_layers, base_layers + nextn) for layer in native_layers)
                or 'native_mtp_writer_groups' not in metadata.get('required_features', [])):
            raise SetupError('Engine native MTP layer declaration does not match the checkpoint')
    name = metadata.get('profile_site_inventory')
    if name is not None:
        if (not isinstance(name, str) or not name or name in ('.', '..')
                or any(c in name for c in ('/', '\\', '\x00'))
                or (kit/name).resolve().parent != kit.resolve()):
            raise SetupError('Invalid engine profile-site inventory path')
        inventory = read_json(kit/name)
        if (not isinstance(inventory, list) or not inventory
                or any(not isinstance(row, dict) or set(row) != {'layer', 'site', 'tensor'}
                       or type(row['layer']) is not int or row['layer'] < 0
                       or not isinstance(row['site'], str) or not row['site']
                       or not isinstance(row['tensor'], str) or not row['tensor'] for row in inventory)):
            raise SetupError('Invalid engine profile-site inventory')
        expected = {(row['layer'], row['site'], row['tensor']) for row in inventory}
        actual = {(row['layer'], row['site'], row['tensor']) for row in sites}
        if len(inventory) != len(expected) or len(sites) != len(actual) or actual != expected:
            raise SetupError('Profile does not cover the exact declared target and native MTP writers')
    return native_layers


def check_backend(home, spec, profile, *, model_id, proc_root=Path('/proc')):
    kind = spec.get('kind')
    if kind not in KINDS[1:]:
        raise SetupError('Choose tensorfold, vllm, sglang or llamacpp as the engine kind')
    allowed = {'kind', 'variant', 'model_path', 'devices', 'arguments', 'draft_model', 'draft_algorithm',
               'speculative_config', 'speculative_arguments', 'python', 'install_dependencies', 'installation', 'source',
               'host', 'port', 'environment', 'startup_timeout', 'allow_untested_memory', 'allow_research', 'integrity_receipts'}
    if set(spec)-allowed:
        raise SetupError('Engine setup contains unknown fields')
    environment = spec.get('environment', {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
        raise SetupError('Engine environment must contain string values')
    if any(name in environment for name in ('HOME', 'home', 'CODEX_HOME')):
        raise SetupError('Engine setup cannot replace the user home environment')
    if spec.get('host', '127.0.0.1') not in ('127.0.0.1', 'localhost', '0.0.0.0'):
        raise SetupError('Managed engines accept a loopback or 0.0.0.0 listen address')
    if 'port' in spec and (type(spec['port']) is not int or not 1 <= spec['port'] <= 65535):
        raise SetupError('Invalid engine port')
    model = inspect_model(spec.get('model_path', ''))
    kit, metadata = _select_kit(kind, model, profile, spec.get('variant'))
    python = select_backend_python(spec, metadata)
    if metadata.get('requires_private_opt_in') and spec.get('allow_research') is not True:
        raise SetupError('This CPU-prepared private engine requires explicit allow_research=true; no GPU qualification is claimed')
    if 'allow_research' in spec and type(spec['allow_research']) is not bool:
        raise SetupError('allow_research must be boolean')
    required_hashes = metadata.get('required_profile_hashes')
    if required_hashes is not None and profile['hashes'] != required_hashes:
        raise SetupError('This private engine requires its exact pinned profile artifact')
    if 'allow_untested_memory' in spec and (type(spec['allow_untested_memory']) is not bool
            or not metadata.get('requires_runtime_memory_guard')):
        raise SetupError('allow_untested_memory is only a boolean option for a guarded experimental engine')
    manifest = read_json(profile['manifest'])
    if model.get('hidden_size') != manifest['hidden_size']:
        raise SetupError('Profile width does not match the model checkpoint')
    if not match_profile(profile, {'id': model_id})[0]:
        raise SetupError('Profile model identifier does not match this deployment')
    if kind == 'llamacpp':
        family = model.get('gguf_architecture')
        supported = metadata['model_architectures']
        transforms = metadata['profile_transforms']
        sites = metadata['profile_sites']
        gguf_binding = _check_gguf_binding(metadata, model, manifest)
    else:
        if model['format'] != 'checkpoint':
            raise SetupError('This engine setup requires a local BF16 checkpoint folder')
        family = model.get('model_type')
        supported = metadata['model_types']
        transforms = metadata['transforms']
        sites = metadata.get('sites', ['embedding_out', 'attn_out', 'mlp_out'])
        quantization = model.get('quantization_config') or (model.get('text_config') or {}).get('quantization_config')
        compatible = metadata.get('compatible_models', [])
        if compatible:
            matching = _model_matches(Path(model['path']), quantization, compatible)
            if len(matching) != 1:
                raise SetupError('Model fingerprints do not match the tested engine variant')
            declared = profile.get('compatible_models', [])
            primary = profile['model']
            primary_matches = (all(primary.get(key) == matching[0].get(key) for key in ('id', 'revision'))
                and all(primary.get(key) is None or primary[key] == matching[0].get(key)
                        for key in ('config_sha256', 'index_sha256', 'quantization_config_sha256')))
            if declared and not primary_matches and not _model_matches(Path(model['path']), quantization, declared):
                raise SetupError('Profile does not declare this serving checkpoint')
        elif quantization:
            raise SetupError('This engine integration has qualified BF16 checkpoints only; this quantization needs separate validation')
    if family not in supported or profile['transform'] not in transforms:
        raise SetupError('This engine integration does not support the selected model family or profile transform')
    grouped = profile['transform'] == 'residual_writer_group_projection_v1'
    typed_grouped = profile.get('schema') == DEEPSEEK_GROUP_SCHEMA
    native_layers = _native_profile_layers(kit, metadata, model, manifest['sites'])
    if grouped:
        sites = ['attn_out', 'combined_mlp_out', 'text_embedding_out', 'ple_value_out']
    if any(site['site'] not in sites for site in manifest['sites']):
        raise SetupError('This engine integration does not support every profile site')
    if metadata.get('model_integrity_verifier'):
        pin = metadata['compatible_models'][0]
        if manifest['model'] != {key: pin[key] for key in ('id', 'revision')}:
            raise SetupError('The native DeepSeek profile must target the exact pinned stock revision')
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', profile['name']) is None:
            raise SetupError('This native DeepSeek loader requires a short ASCII profile identifier')
        if (Path(profile['path'])/'profile.safetensors').stat().st_size > 16*1024*1024:
            raise SetupError('This native DeepSeek loader accepts profile payloads up to 16 MiB')
    for site in ([] if grouped or typed_grouped else manifest['sites']):
        layer, target = site['layer'], site['tensor']
        stem = f'model.language_model.layers.{layer}'
        expected = {'embedding_out': ['model.language_model.embed_tokens.weight'],
                    'attn_out': [stem+'.self_attn.o_proj.weight', stem+'.linear_attn.out_proj.weight'],
                    'mlp_out': [stem+'.mlp.down_proj.weight']}
        patterns = metadata.get('site_tensor_patterns')
        pattern = patterns.get(site['site']) if patterns else None
        valid = (bool(pattern) and re.fullmatch(pattern.format(layer=layer), target) is not None
                 if patterns else target in expected.get(site['site'], []))
        if not valid:
            raise SetupError('This engine integration does not recognize a profile target tensor')
        if (model.get('num_hidden_layers') is not None and layer >= model['num_hidden_layers']
                and layer not in native_layers):
            raise SetupError('Profile refers to a layer outside the model')
    if kind == 'llamacpp':
        _check_llamacpp_strengths(metadata, manifest['sites'])
    devices = spec.get('devices')
    parallel_sizes = metadata.get('tensor_parallel_sizes', [1])
    if devices is None:
        free = available_devices()
        count = max((size for size in parallel_sizes if size <= len(free)), default=0)
        devices = free[:count]
    if (not isinstance(devices, list) or len(devices) not in parallel_sizes
            or any(type(value) not in (str, int) for value in devices)
            or len({str(value) for value in devices}) != len(devices)):
        counts = ', '.join(str(value) for value in parallel_sizes)
        raise SetupError('This engine integration requires a supported GPU count ('+counts+'); select free devices')
    arguments = spec.get('arguments', [])
    if not isinstance(arguments, list) or any(not isinstance(value, str) for value in arguments):
        raise SetupError('Engine arguments must be a string array')
    arguments = normalize_engine_arguments(kind, arguments,
        llamacpp_method=spec.get('draft_algorithm'), llamacpp_draft_model=bool(spec.get('draft_model')))
    defaults = metadata.get('default_arguments')
    if defaults is not None and (not isinstance(defaults, list)
            or any(not isinstance(value, str) for value in defaults)):
        raise SetupError('Engine default arguments must be a string array')
    reserved = {'--model', '--model-path', '-m', '--host', '--port', '--served-model-name', '--alias',
                '--no-drafts', '--disable-speculative-decoding', '--speculative-config',
                '--speculative-algorithm', '--speculative-draft-model-path', '--spec-draft-model', '--model-draft', '-md', '--spec-type',
                '--tp-size', '--tp', '--tensor-parallel-size', '--ep-size', '--expert-parallel-size'}
    require_full_vllm_options(kind, arguments, reserved | {'--max-model-len'})
    if any(value.split('=', 1)[0].split('.', 1)[0] in reserved
           or value.startswith(('--speculative-', '--spec-')) for value in arguments):
        raise SetupError('Engine arguments cannot override managed model, listener, identity or speculative settings')
    if metadata.get('model_integrity_verifier'):
        controlled = {'--allow-research', '--allow-untested-memory', '--dry-run', '--replay-file', '--model-receipt',
                      '--quantization', '-q', '--dtype', '--hf-overrides', '--hf-config-path',
                      '--enforce-eager', '--no-enforce-eager'}
        require_full_vllm_options(kind, arguments, controlled)
        if any(value.split('=', 1)[0].split('.', 1)[0] in controlled for value in arguments):
            raise SetupError('Experimental engine safety and checkpoint settings must use their managed plan fields')
    drafting = _speculation(home, spec, kind, model_id, metadata=metadata, proc_root=proc_root, devices=devices, model=model)
    modalities = _input_modalities(metadata, family)
    return {'kind': kind, 'kit': str(kit), 'model': model, 'devices': devices,
            'metadata': metadata, 'python': python, 'drafting': drafting, 'arguments': arguments,
            **({'gguf_binding': gguf_binding} if kind == 'llamacpp' and gguf_binding else {}),
            'input_modalities': modalities,
            'openclaw_compat': declared_openclaw_compat(metadata, family),
            'request_compatibility': declared_request_compatibility(metadata, family),
            'openclaw_request_policy': declared_openclaw_request_policy(metadata, family),
            'context_window': managed_context_window(kind, metadata, arguments), 'status': metadata['status']}


def _run(command, log_path, *, cwd=None, env=None):
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, 'ab') as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SetupError('Engine preparation failed; inspect the private install.log in its deployment directory')
    except OSError as error:
        raise SetupError('Engine preparation could not start; check the required Python, Git and build tools') from error


def _copy_kit(original, destination):
    files = {str(path.relative_to(original)): digest(path) for path in original.rglob('*')
             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    if destination.exists():
        actual = {str(path.relative_to(destination)): digest(path) for path in destination.rglob('*')
                  if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
        if actual != files:
            raise SetupError('Managed integration files changed; use a fresh deployment instead of overwriting them')
    else:
        temporary = destination.with_name(destination.name+'.new-'+secrets.token_hex(8))
        try:
            shutil.copytree(original, temporary, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            actual = {str(path.relative_to(temporary)): digest(path) for path in temporary.rglob('*')
                      if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
            if actual != files:
                raise SetupError('Integration source changed while it was being copied')
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return files


def _prepare_runtime(kind, spec, kit, state):
    log = state/'install.log'
    metadata = read_json(kit/'backend.json') if (kit/'backend.json').is_file() else {}
    python = select_backend_python(spec, metadata)['executable']
    install = spec.get('install_dependencies', False)
    if type(install) is not bool:
        raise SetupError('install_dependencies must be true or false')
    if kind == 'llamacpp':
        installation = Path(spec['installation']).expanduser().resolve() if spec.get('installation') else state/'native/installation.json'
        if not installation.is_file():
            if not install:
                raise SetupError('llama.cpp needs engine.install_dependencies=true or a prepared installation.json')
            _run([python, str(kit/'install.py'), '--state', str(state/'native'), '--cuda', 'on'], log)
        runtime = read_json(installation)
        expected = read_json(kit/'backend.json')['revision']
        patch = read_json(kit/'patch.json')['patch_sha256']
        if (runtime.get('revision') != expected or runtime.get('status') != 'prepared'
                or runtime.get('patch_sha256') != patch):
            raise SetupError('llama.cpp installation does not match the pinned FlashAlign runtime')
        for name in ('binary', 'gguf_python', 'python'):
            if not Path(runtime.get(name, '')).exists():
                raise SetupError('llama.cpp installation is incomplete')
        if runtime.get('binary_sha256') != digest(runtime['binary']):
            raise SetupError('llama.cpp binary differs from its installation receipt')
        return runtime
    if install:
        environment = state/'venv'
        if not (environment/'bin/python').is_file():
            _run([python, '-m', 'venv', str(environment)], log)
        python = str(environment/'bin/python')
        receipt = state/'dependencies.json'
        signature = digest(kit/'requirements.txt')
        if not receipt.is_file() or read_json(receipt).get('requirements_sha256') != signature:
            _run([python, '-m', 'pip', '--isolated', 'install', '-r', str(kit/'requirements.txt')], log)
            if kind == 'vllm':
                # Build tooling can write egg-info/build folders into its input tree.
                # Keep the runtime copy immutable across prepare and activation.
                with tempfile.TemporaryDirectory(prefix='package-build-', dir=state) as temporary:
                    package = Path(temporary)/'source'
                    shutil.copytree(kit, package)
                    _run([python, '-m', 'pip', '--isolated', 'install', '--no-deps', str(package)], log)
            private_json(receipt, {'requirements_sha256': signature})
    if kind == 'vllm':
        # PYTHONPATH alone cannot install vLLM's general-plugin entry point.
        plugin = metadata.get('plugin_name', 'flashalign')
        if not isinstance(plugin, str) or not re.fullmatch(r'[A-Za-z0-9_]+', plugin):
            raise SetupError('Invalid pinned engine plugin name')
        target = metadata.get('plugin_entrypoint', 'flashalign_vllm.plugin:register')
        if not isinstance(target, str) or re.fullmatch(r'[A-Za-z0-9_.]+:[A-Za-z0-9_]+', target) is None:
            raise SetupError('Invalid pinned engine plugin entry point')
        supplied_environment = {'env': {**os.environ, **spec['environment']}} if spec.get('environment') else {}
        _run([python, '-c', "import importlib.metadata as m; assert m.version('vllm') == '0.32.0'; entries=[e for e in m.entry_points(group='vllm.general_plugins') if e.name == "+repr(plugin)+"]; assert entries and all(e.value == "+repr(target)+" for e in entries)"], log, **supplied_environment)
        runtime = {'python': python, 'build_environment': prepare_python_headers(
            python, state, install=install, environment=spec.get('environment'))}
        overlay_requirements = metadata.get('isolated_overlay_requirements')
        if overlay_requirements:
            if Path(overlay_requirements).name != overlay_requirements or not (kit/overlay_requirements).is_file():
                raise SetupError('Invalid isolated runtime overlay')
            overlay = state/'runtime-overlay'
            receipt = overlay/'flashalign-overlay.json'
            signature = digest(kit/overlay_requirements)
            if not receipt.is_file() or read_json(receipt).get('requirements_sha256') != signature:
                if not install:
                    distribution = metadata.get('overlay_distribution')
                    version = metadata.get('overlay_version')
                    if not isinstance(distribution, str) or not isinstance(version, str):
                        raise SetupError('This engine needs its isolated runtime overlay; enable dependency installation')
                    _run([python, '-c', 'import importlib.metadata as m; assert m.version('+repr(distribution)+') == '+repr(version)],
                         log, env={**os.environ, **spec.get('environment', {})})
                    runtime['provided_pythonpath'] = spec.get('environment', {}).get('PYTHONPATH', '')
                    runtime['overlay_source'] = 'operator-provided compatible environment'
                    return runtime
                if overlay.exists():
                    raise SetupError('The existing runtime overlay lacks its matching receipt; preserve it and use a fresh deployment')
                with tempfile.TemporaryDirectory(prefix='overlay-build-', dir=state) as temporary:
                    prepared = Path(temporary)/'overlay'
                    _run([python, '-m', 'pip', '--isolated', 'install', '--no-deps', '--require-hashes',
                          '--target', str(prepared), '-r', str(kit/overlay_requirements)], log)
                    private_json(prepared/'flashalign-overlay.json', {'requirements_sha256': signature})
                    os.replace(prepared, overlay)
            if not overlay.is_dir():
                raise SetupError('The isolated runtime overlay is missing')
            runtime['overlay'] = str(overlay)
        return runtime
    _run([python, '-c', "import importlib.metadata as m; assert m.version('sglang') == '0.5.20'"], log)
    source = state/'sglang'
    if not source.exists():
        original = Path(spec['source']).expanduser().resolve() if spec.get('source') else state/'upstream'
        if not original.exists():
            if not install:
                raise SetupError('Provide a pinned SGLang source checkout or enable dependency installation')
            _run(['git', 'init', str(original)], log)
            _run(['git', '-C', str(original), 'remote', 'add', 'origin', 'https://github.com/sgl-project/sglang.git'], log)
            _run(['git', '-C', str(original), 'fetch', '--depth', '1', 'origin', read_json(kit/'backend.json')['upstream_revision']], log)
            _run(['git', '-C', str(original), 'checkout', '--detach', 'FETCH_HEAD'], log)
        _run([python, str(kit/'prepare.py'), '--source', str(original), '--output', str(source)], log)
    receipt = read_json(source/'flashalign-build.json')
    if receipt.get('commit') != read_json(kit/'backend.json')['upstream_revision']:
        raise SetupError('SGLang source differs from the pinned integration')
    final_edits = {row['path']: row for row in receipt.get('files', [])}
    for row in final_edits.values():
        path = source/row['path']
        if not path.resolve().is_relative_to(source.resolve()) or digest(path) != row['after_sha256']:
            raise SetupError('Prepared SGLang source changed; use a fresh deployment')
    for path in (kit/'flashalign_sglang').rglob('*.py'):
        target = source/'python/sglang/srt/flashalign'/path.relative_to(kit/'flashalign_sglang')
        if digest(path) != digest(target):
            raise SetupError('Prepared SGLang integration changed; use a fresh deployment')
    return {'python': python, 'source': str(source), 'build_environment': prepare_python_headers(
        python, state, install=install, environment=spec.get('environment'))}


def _prepare_checkpoint_receipts(spec, checked, runtime, kit, state):
    config = checked['metadata'].get('checkpoint_integrity')
    provided = spec.get('integrity_receipts', {})
    if config is None:
        if provided:
            raise SetupError('This engine does not accept external checkpoint receipts')
        return {}
    if not isinstance(provided, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in provided.items()):
        raise SetupError('integrity_receipts must map roles to trusted local receipt paths')
    roles = config.get('roles_by_algorithm', {}).get(checked['drafting']['draft_algorithm'])
    if not isinstance(roles, list) or not roles or any(role not in ('stock', 'dflash') for role in roles):
        raise SetupError('Invalid checkpoint role declaration')
    if set(provided)-set(roles):
        raise SetupError('Unexpected checkpoint receipt role')
    name = config.get('script', '')
    entry = (kit/name).resolve()
    if not isinstance(name, str) or not entry.is_relative_to(kit.resolve()) or not entry.is_file():
        raise SetupError('Invalid checkpoint verifier')
    paths = {}
    for role in roles:
        destination = state/'model-integrity'/(role+'.json')
        if role in provided and not destination.exists():
            source = Path(provided[role]).expanduser().resolve()
            info = source.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise SetupError('Checkpoint receipt must be owned and not writable by other users')
            private_json(destination, read_json(source))
        _run([runtime['python'], str(entry), '--model', checked['model']['path'],
              '--receipt', str(destination), '--role', role, '--verify'], state/'install.log')
        paths[role] = str(destination)
    return paths


def prepare_backend(home, spec, profile_path, *, model_id, key_file=None):
    profile = validate_profile(profile_path)
    checked = check_backend(home, spec, profile, model_id=model_id)
    # The selected interpreter is part of the immutable deployment identity.
    spec = {**spec, 'python': checked['python']['executable']}
    kind = checked['kind']
    original = Path(checked['kit'])
    kit_hash = hashlib.sha256(''.join(digest(path) for path in sorted(original.rglob('*'))
                                    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc').encode()).hexdigest()
    identity = hashlib.sha256((kit_hash+profile['hashes']['profile.json']+profile['hashes']['profile.safetensors']+str(Path(spec['model_path']).resolve())+
                               model_id+json.dumps(spec, sort_keys=True, default=str)).encode()).hexdigest()[:20]
    state = Path(home)/'.config/owner-mode/engines'/identity
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    kit = state/'integration'
    files = _copy_kit(original, kit)
    runtime = _prepare_runtime(kind, spec, kit, state)
    verifier = checked['metadata'].get('model_integrity_verifier')
    if verifier:
        entry = (kit/verifier).resolve()
        if not entry.is_relative_to(kit.resolve()) or not entry.is_file():
            raise SetupError('Invalid model integrity verifier')
        receipt = state/'model-integrity/checkpoint.json'
        _run([runtime['python'], str(entry), '--model', checked['model']['path'],
              '--receipt', str(receipt), '--verify'], state/'install.log')
        runtime['model_receipt'] = str(receipt)
    checkpoint_receipts = _prepare_checkpoint_receipts(spec, checked, runtime, kit, state)
    key = Path(key_file).expanduser().resolve() if key_file else state/'engine.key'
    ensure_key(key)
    read_key(key)
    staged = state/'profile'
    stage_profile(profile['path'], staged)
    previous = read_json(state/'deployment.json') if (state/'deployment.json').is_file() else {}
    port = spec.get('port') or urlsplit(previous.get('base_url', '')).port or unused_port()
    if type(port) is not int or not 1 <= port <= 65535:
        raise SetupError('Invalid engine port')
    host = spec.get('host', '127.0.0.1')
    if host not in ('127.0.0.1', 'localhost', '0.0.0.0'):
        raise SetupError('Managed engines accept a loopback or 0.0.0.0 listen address')
    model = checked['model']['path']
    drafting = checked['drafting']
    extra = checked['arguments']
    python = runtime['python']
    common = ['--model', model, '--profile', str(staged), '--key-file', str(key), '--host', host, '--port', str(port)]
    if kind == 'vllm':
        launcher = checked['metadata'].get('launch')
        if launcher:
            if Path(launcher).name != launcher or not (kit/launcher).is_file():
                raise SetupError('Invalid engine variant launcher')
            argv = [python, str(kit/launcher), *common, '--speculative-config',
                    json.dumps(drafting['speculative_config'], separators=(',', ':'))]
            if verifier:
                argv += ['--model-receipt', runtime['model_receipt'], '--replay-file', str(state/'replay/nonces.sqlite3')]
                if spec.get('allow_untested_memory'):
                    argv += ['--allow-untested-memory']
            else:
                argv += ['--enable-prefix-caching']
            argv += extra
        else:
            argv = [python, '-m', 'flashalign_vllm', *common, '--speculative-config', json.dumps(drafting['speculative_config'], separators=(',', ':'))]
            defaults = checked['metadata'].get('default_arguments')
            if defaults is None:
                defaults = ['--max-model-len', '4096', '--gpu-memory-utilization', '0.85', '--enable-prefix-caching']
            argv += [*normalize_engine_arguments(kind, defaults), *extra]
    elif kind == 'sglang':
        argv = [python, str(kit/'launch.py'), '--source', runtime['source'], *common,
                '--draft-algorithm', drafting['draft_algorithm']]
        if drafting.get('draft_model'):
            argv += ['--draft-model', drafting['draft_model']]
        if checked['metadata'].get('managed_stateful_launcher'):
            argv += ['--state', str(state/'runtime')]
            if checked['metadata'].get('requires_private_opt_in'):
                argv += ['--allow-unqualified-candidate']
            for role, path in checkpoint_receipts.items():
                option = {'stock': '--stock-receipt', 'dflash': '--draft-receipt'}.get(role)
                if option is None:
                    raise SetupError('Unsupported managed checkpoint role')
                argv += [option, path]
        argv += ['--', '--served-model-name', model_id]
        defaults = checked['metadata'].get('default_arguments')
        if defaults is None:
            if checked['metadata'].get('variant'):
                raise SetupError('Engine variant must declare tested default arguments')
            defaults = ['--context-length', '4096', '--mem-fraction-static', '0.82', '--max-running-requests', '2',
                        '--attention-backend', 'triton', '--enable-cache-report']
        defaults = _without_options(defaults, {'--speculative-algorithm', '--speculative-draft-model-path'})
        if not argument(defaults, '--context-length'):
            context = checked['metadata'].get('qualification', {}).get('tested_context_length', 4096)
            defaults += ['--context-length', str(context)]
        argv += _sglang_arguments(defaults, extra)
        argv += drafting.get('speculative_arguments', []) or ([] if 'native_mtp_writer_groups' in checked['metadata'].get('required_features', [])
                                                            or drafting['draft_algorithm'] in checked['metadata'].get('native_speculative_algorithms', [])
                                                            or drafting['draft_algorithm'] in checked['metadata'].get('draftless_speculative_algorithms', [])
                                                            else ['--speculative-num-draft-tokens', '8'])
    else:
        defaults = checked['metadata'].get('default_arguments')
        if defaults is None:
            defaults = ['--ctx-size', '4096', '--parallel', '2', '--cache-ram', '512',
                        '--n-gpu-layers', '99', '--no-webui', '--metrics']
        native_args = [*defaults, *extra]
        if drafting.get('draft_algorithm'):
            native_args += ['--spec-type', drafting['draft_algorithm']]
        if drafting.get('draft_model'):
            native_args = [*native_args,
                           *(drafting.get('speculative_arguments') or ['--spec-draft-n-max', '4', '--spec-draft-ngl', '99'])]
        else:
            native_args += drafting.get('speculative_arguments', [])
        private_json(state/'native-arguments.json', native_args)
        argv = [python, str(kit/'launcher.py'), '--binary', runtime['binary'], '--gguf-python', runtime['gguf_python'],
                '--state', str(state/'native-runtime'), *common, '--engine-args', str(state/'native-arguments.json')]
        if drafting.get('draft_model'):
            argv += ['--draft-model', drafting['draft_model']]
    environment = spec.get('environment', {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in environment.items()):
        raise SetupError('Engine environment must contain string values')
    if any(name in environment for name in ('HOME', 'home', 'CODEX_HOME')):
        raise SetupError('Engine setup cannot replace the user home environment')
    env = {**environment, **runtime.get('build_environment', {}), 'CUDA_VISIBLE_DEVICES': ','.join(map(str, checked['devices'])),
           'TORCH_EXTENSIONS_DIR': str(state/'torch-extensions'), 'TRITON_CACHE_DIR': str(state/'triton-cache'),
           'HF_HUB_DISABLE_TELEMETRY': '1'}
    if kind == 'vllm':
        runtime_path = runtime.get('overlay') or runtime.get('provided_pythonpath')
        env['PYTHONPATH'] = os.pathsep.join([str(kit), runtime_path]) if runtime_path else str(kit)
    result = {'version': 1, 'kind': kind, 'state': str(state), 'source': str(kit),
              'commands': [{'argv': argv, 'env': env}], 'base_url': f'http://127.0.0.1:{port}/v1',
              'model': model_id, 'profile': profile['name'], 'key_file': str(key), 'profile_path': str(staged),
              'integration_files': files, 'drafting': drafting, 'status': 'prepared', 'install_dependencies': False,
              'python': python, 'requirements': []}
    if checked.get('gguf_binding'):
        result['gguf_model'] = checked['gguf_binding']
    if verifier:
        result.update(model_receipt=runtime['model_receipt'], resource_policy=checked['metadata']['resource_policy'],
                      allow_untested_memory=spec.get('allow_untested_memory', False),
                      hardware_activation_check='pending', replay_file=str(state/'replay/nonces.sqlite3'))
    if checkpoint_receipts:
        result.update(checkpoint_receipts=checkpoint_receipts, resource_policy=checked['metadata'].get('resource_policy'),
                      replay_file=str(state/'runtime/replay/nonces.sqlite3'), hardware_activation_check='pending')
    private_json(state/'deployment.json', result)
    return result
