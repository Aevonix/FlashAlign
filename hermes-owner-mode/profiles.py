# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Validate bundled and operator-supplied runtime profile artifacts."""
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

from .deepseek_grouped import (DEEPSEEK_GROUP_SCHEMA, DEEPSEEK_MODEL,
                               validate_grouped_profile)


class ProfileError(ValueError):
    pass


def stage_profile(source, destination):
    """Publish a verified profile directory atomically, without overwriting one."""
    checked = validate_profile(source)
    destination = Path(destination)
    if os.path.lexists(destination):
        if destination.is_symlink() or validate_profile(destination)['hashes'] != checked['hashes']:
            raise ProfileError('Staged profile changed; use a fresh deployment')
        return checked
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name+'.new-', dir=destination.parent))
    try:
        for name in ('profile.json', 'profile.safetensors', 'sha256.json'):
            original = Path(checked['path'])/name
            if original.is_file():
                shutil.copyfile(original, temporary/name)
                (temporary/name).chmod(0o600)
        receipt = read_json(temporary/'sha256.json') if (temporary/'sha256.json').exists() else {}
        if any(receipt.get(name) != value for name, value in checked['hashes'].items()):
            (temporary/'sha256.json').write_text(json.dumps({**receipt, **checked['hashes']}, indent=2)+'\n')
            (temporary/'sha256.json').chmod(0o600)
        copy_profile_notices(checked['path'], temporary)
        if validate_profile(temporary)['hashes'] != checked['hashes']:
            raise ProfileError('Profile changed while it was being staged')
        os.replace(temporary, destination)
        return checked
    finally:
        if temporary.exists():shutil.rmtree(temporary)


FORMS = {
    'projection': set(),
    'row_gain_projection_v1': {'row_gain'},
    'row_normalized_projection_v1': {'row_gain', 'row_norm_preconditioning'},
    'input_low_rank_delta_v1': {'matrix_input'},
    'weighted_output_projection_v1': {'per_direction_strength'},
    'residual_writer_group_projection_v1': {'writer_groups', 'text_embedding_before_multimodal_merge',
                                           'ple_value_projection', 'native_mtp_writer_groups'},
}
MAX_HEADER = 4 * 1024 * 1024
MAX_PROFILE = 256 * 1024 * 1024
def _deepseek_group_tensors(profile, hidden, sites):
    return validate_grouped_profile(profile, hidden, sites, error=ProfileError)


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ProfileError('Duplicate JSON field')
        value[key] = item
    return value


def _nonfinite(_):
    raise ProfileError('Non-finite JSON number')


def read_json(path, limit=MAX_HEADER):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise ProfileError('Missing or oversized profile metadata')
    try:
        return json.loads(path.read_text(), object_pairs_hook=_object, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError('Invalid profile JSON') from error


def _positive_int(value):
    return type(value) is int and value > 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _group_tensors(profile, hidden, sites):
    if type(profile.get('schema_version')) is not int or profile['schema_version'] != 1:
        raise ProfileError('Unsupported writer-group schema version')
    if (type(profile.get('rank')) is not int or profile['rank'] != 1
            or not _number(profile.get('scale')) or profile['scale'] != 1 or profile.get('tensor') != 'shared.Q'):
        raise ProfileError('Unsupported shared writer-group geometry')
    seen = set()
    kinds = set()
    for site in sites:
        if not isinstance(site, dict):
            raise ProfileError('Invalid writer group')
        layer, kind, role = site.get('layer'), site.get('site'), site.get('model_role')
        if (type(layer) is not int or layer < 0
                or kind not in ('attn_out', 'combined_mlp_out', 'text_embedding_out', 'ple_value_out')
                or role not in ('target', 'native_mtp', 'target_and_native_mtp')):
            raise ProfileError('Unsupported writer-group role or site')
        key = (role, layer, kind)
        if key in seen:
            raise ProfileError('Duplicate writer group')
        seen.add(key);kinds.add(kind)
        sources = site.get('source_tensors')
        if (not isinstance(sources, list) or not sources or len(sources) > 1024
                or any(not isinstance(value, str) or not value or len(value) > 1024 for value in sources)
                or len(set(sources)) != len(sources)):
            raise ProfileError('Invalid writer-group source tensors')
        if kind == 'combined_mlp_out' and (not _positive_int(site.get('expert_count'))
                or site.get('fused_expert_tensor_axes') != ['expert', 'output', 'input']):
            raise ProfileError('Invalid expert writer-group declaration')
        if kind == 'text_embedding_out' and (layer != 0 or role != 'target_and_native_mtp'):
            raise ProfileError('Invalid shared text embedding group')
    if not {'text_embedding_out', 'ple_value_out'} <= kinds or not any(key[0] == 'native_mtp' for key in seen):
        raise ProfileError('Writer groups do not declare their required feature coverage')
    if not profile.get('compatible_models'):
        raise ProfileError('Writer-group profiles require pinned compatible models')
    return {'shared.Q': [1, hidden]}


def _compatible_models(value):
    if not isinstance(value, list):
        raise ProfileError('Compatible models must be a list')
    for model in value:
        if (not isinstance(model, dict) or not isinstance(model.get('id'), str) or not model['id']
                or not isinstance(model.get('model_type'), str) or not model['model_type']
                or not isinstance(model.get('revision'), str)
                or not re.fullmatch('[0-9a-f]{40}', model['revision'])
                or any(not isinstance(model.get(field), str) or not re.fullmatch('[0-9a-f]{64}', model[field])
                       for field in ('config_sha256',))
                or ('quantization_config_sha256' in model
                    and (not isinstance(model['quantization_config_sha256'], str)
                         or not re.fullmatch('[0-9a-f]{64}', model['quantization_config_sha256'])))):
            raise ProfileError('Invalid compatible-model fingerprints')
        index = model.get('index_sha256')
        header = model.get('weights_header')
        if index is not None:
            if header is not None or not isinstance(index, str) or not re.fullmatch('[0-9a-f]{64}', index):
                raise ProfileError('Invalid compatible-model weight fingerprints')
        elif (not isinstance(header, dict) or set(header) != {'file', 'sha256', 'size'}
                or not isinstance(header.get('file'), str)
                or not re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors', header['file'])
                or not isinstance(header.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', header['sha256'])
                or not _positive_int(header.get('size'))):
            raise ProfileError('Invalid compatible-model weight fingerprints')
    return value


def validate_profile(path):
    """Validate schema, float payloads and supplied hashes, not model quality."""
    path = Path(path).expanduser()
    manifest = path / 'profile.json' if path.is_dir() else path
    root = manifest.parent.resolve()
    profile = read_json(manifest)
    if not isinstance(profile, dict):
        raise ProfileError('Profile metadata must be an object')
    name = profile.get('name')
    model = profile.get('model')
    if (not isinstance(name, str) or not name.strip() or name == 'off' or len(name) > 256
            or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in name)):
        raise ProfileError('Invalid profile name')
    if not isinstance(model, dict) or not isinstance(model.get('id'), str) or not model['id']:
        raise ProfileError('Profile must identify its model')
    if not isinstance(model.get('revision'), str) or not model['revision']:
        raise ProfileError('Profile must pin a model revision')
    compatible = _compatible_models(profile.get('compatible_models', []))
    hidden = profile.get('hidden_size')
    if not _positive_int(hidden):
        raise ProfileError('Invalid profile hidden size')
    transform = profile.get('transform', 'projection')
    requires = profile.get('requires', [])
    schema = profile.get('schema')
    deepseek_grouped = schema == DEEPSEEK_GROUP_SCHEMA
    if schema is not None and not deepseek_grouped:
        raise ProfileError('Unsupported profile schema')
    if (not isinstance(transform, str) or transform not in FORMS or not isinstance(requires, list)
            or any(not isinstance(item, str) for item in requires)
            or len(set(requires)) != len(requires)
            or (not deepseek_grouped and set(requires) != FORMS[transform])
            or (deepseek_grouped and transform != 'projection')):
        raise ProfileError('Unsupported or inconsistent profile transform')
    sites = profile.get('sites')
    if not isinstance(sites, list) or not sites or len(sites) > 4096:
        raise ProfileError('Profile must contain valid transform sites')
    grouped = transform == 'residual_writer_group_projection_v1'
    expected = (_deepseek_group_tensors(profile, hidden, sites) if deepseek_grouped
                else _group_tensors(profile, hidden, sites) if grouped else {})
    basis_keys = set(expected) if deepseek_grouped else set()
    seen = set()
    for site in ([] if grouped or deepseek_grouped else sites):
        if not isinstance(site, dict):
            raise ProfileError('Invalid profile site')
        layer, output, rank = site.get('layer'), site.get('site'), site.get('rank')
        if (type(layer) is not int or layer < 0 or output not in ('attn_out', 'shared_expert_out', 'mlp_out', 'embedding_out')
                or not _positive_int(rank) or rank > hidden or not _number(site.get('scale'))
                or not isinstance(site.get('tensor'), str) or not site['tensor']):
            raise ProfileError('Invalid profile site, rank or scale')
        if output == 'embedding_out' and layer != 0:
            raise ProfileError('Embedding output sites use layer zero')
        prefix = f'L{layer}.{output}'
        if prefix in seen:
            raise ProfileError('Duplicate profile site')
        seen.add(prefix)
        expected[prefix + '.Q'] = [rank, hidden]
        if transform == 'weighted_output_projection_v1':
            if site['scale'] != 1:
                raise ProfileError('Weighted projections require unit site scale')
            expected[prefix + '.strengths'] = [rank]
        if 'row_gain' in requires:
            expected[prefix + '.g'] = [hidden]
        if 'row_norm_preconditioning' in requires:
            expected[prefix + '.m'] = [hidden]
        if transform == 'input_low_rank_delta_v1':
            width = site.get('input_size')
            if not _positive_int(width):
                raise ProfileError('Invalid input-delta width')
            expected[prefix + '.B'] = [rank, width]
    tensors = root / 'profile.safetensors'
    size = tensors.stat().st_size if tensors.is_file() else 0
    if size < 10 or size > MAX_PROFILE:
        raise ProfileError('Missing or oversized profile tensor file')
    with tensors.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if header_size > MAX_HEADER or header_size + 8 > size:
            raise ProfileError('Invalid tensor header length')
        try:
            header = json.loads(stream.read(header_size), object_pairs_hook=_object, parse_constant=_nonfinite)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ProfileError('Invalid tensor header') from error
        if not isinstance(header, dict) or set(header) - {'__metadata__'} != set(expected):
            raise ProfileError('Missing or unexpected profile tensors')
        intervals = []
        for key, shape in expected.items():
            info = header[key]
            if (not isinstance(info, dict) or info.get('dtype') != 'F32' or info.get('shape') != shape
                    or any(type(n) is not int for n in info.get('shape', []))):
                raise ProfileError('Profile tensors must have the declared FP32 shapes')
            offsets = info.get('data_offsets')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ProfileError('Invalid tensor offsets')
            start, end = offsets
            if start < 0 or end - start != math.prod(shape) * 4 or end > size - 8 - header_size:
                raise ProfileError('Tensor payload length does not match its shape')
            intervals.append((start, end))
            stream.seek(8 + header_size + start)
            values = array('f')
            values.frombytes(stream.read(end - start))
            if sys.byteorder != 'little':
                values.byteswap()
            if any(not math.isfinite(value) for value in values):
                raise ProfileError('Profile contains non-finite tensor values')
            if key.endswith(('.g', '.m', '.strengths')) and any(value <= 0 for value in values):
                raise ProfileError('Profile gain and norm values must be positive')
            if key.endswith('.Q') or key in basis_keys:
                rank, width = shape
                if rank > 128:
                    raise ProfileError('Profile rank exceeds the setup validator limit of 128')
                rows = [values[i * width:(i + 1) * width] for i in range(rank)]
                for i, row in enumerate(rows):
                    for j in range(i + 1):
                        dot = math.fsum(a * b for a, b in zip(row, rows[j]))
                        if abs(dot - (1 if i == j else 0)) > 2e-5:
                            raise ProfileError('Profile basis rows are not orthonormal')
        end = 0
        for start, stop in sorted(intervals):
            if start != end:
                raise ProfileError('Tensor payloads overlap or contain gaps')
            end = stop
        if end != size - 8 - header_size:
            raise ProfileError('Unexpected bytes in profile tensor payload')
    hashes = {'profile.json': _digest(manifest), 'profile.safetensors': _digest(tensors)}
    receipt = root / 'sha256.json'
    if receipt.is_file():
        recorded = read_json(receipt)
        if not isinstance(recorded, dict):
            raise ProfileError('Invalid checksum record')
        for filename, digest in hashes.items():
            supplied = recorded.get(filename)
            if supplied is not None and supplied != digest:
                raise ProfileError('Profile checksum mismatch')
    return {'name': name, 'model': model, 'transform': transform, 'requires': requires,
            'schema': schema,
            'compatible_models': compatible,
            'manifest': str(manifest.resolve()),
            'path': str(root), 'hashes': hashes, 'artifact_valid': True,
            'status': profile.get('status', 'behavior not certified'), 'source': 'local'}


def discover_profiles(paths):
    profiles, errors, seen = [], [], set()
    for location in paths:
        root = Path(location).expanduser()
        if not root.exists():
            continue
        catalog = {}
        catalog_path = root / 'catalog.json' if root.is_dir() else root
        if catalog_path.name == 'catalog.json' and catalog_path.is_file():
            try:
                entries = read_json(catalog_path).get('models', [])
                for entry in entries:
                    if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
                        continue
                    folder = (catalog_path.parent / entry['path']).resolve()
                    folder.relative_to(catalog_path.parent.resolve())
                    aliases = entry.get('model_ids', [])
                    if isinstance(aliases, list) and all(isinstance(item, str) for item in aliases):
                        catalog[str(folder)] = aliases
            except (OSError, ValueError, TypeError, AttributeError):
                errors.append({'path': str(catalog_path), 'reason': 'Invalid profile catalog'})
        if root.name == 'catalog.json':
            candidates = [Path(folder) / 'profile.json' for folder in catalog]
        else:
            candidates = [root] if root.is_file() or (root / 'profile.json').is_file() else sorted(root.glob('*/profile.json'))
        for candidate in candidates:
            marker = str(candidate.resolve())
            if marker in seen:
                continue
            seen.add(marker)
            try:
                item = validate_profile(candidate)
                item['model_ids'] = catalog.get(item['path'], [])
                profiles.append(item)
            except (OSError, ValueError, TypeError, OverflowError) as error:
                errors.append({'path': str(candidate), 'reason': str(error)})
    return profiles, errors


def bundled_profile_root():
    return Path(__file__).resolve().parents[1]/'profiles'


def copy_profile_notices(source, destination):
    """Preserve supplied legal notices alongside installed runtime parameters."""
    for name in ('LICENSE', 'LICENSE.txt', 'LICENSE.source', 'NOTICE.md'):
        path = Path(source)/name
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_HEADER:
            target = Path(destination)/name
            if path.resolve() != target.resolve():
                shutil.copyfile(path, target)
            target.chmod(0o600)


def bundled_profile_for(config, root=None):
    """Select a starter by declared family and dimensions; no network or model load."""
    root = Path(root) if root is not None else bundled_profile_root()
    if not (root/'catalog.json').is_file():
        return None
    catalog = read_json(root/'catalog.json')
    text = config.get('text_config') or config
    family = config.get('model_type') or text.get('model_type')
    hidden = text.get('hidden_size')
    revision = config.get('_commit_hash')
    available, errors = discover_profiles([root])
    by_path = {item['path']: item for item in available}
    for entry in catalog.get('models', []):
        if family not in entry.get('model_types', []) or hidden != entry.get('hidden_size'):
            continue
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ProfileError('Starter profile path escapes its package')
        profile = by_path.get(str(path))
        if profile is None:
            raise ProfileError('The matching starter profile failed file validation')
        revisions = {item.get('revision') for item in [profile['model'], *profile.get('compatible_models', [])]}
        if revision and revision not in revisions:
            raise ProfileError('The model revision differs from its starter profile')
        status = profile.get('status', '')
        if isinstance(status, str) and (status == 'failed' or status.startswith('failed_')):
            raise ProfileError('The matching starter profile has a failed validation result; '
                               'provide a separately validated profile explicitly')
        return profile
    return None


def match_profile(profile, model):
    """Use explicit IDs/aliases and reject known revision/format conflicts."""
    reason = 'different model'
    for target in [profile['model'], *profile.get('compatible_models', [])]:
        aliases = target.get('aliases', [])
        aliases = aliases if isinstance(aliases, list) else []
        model_ids = profile.get('model_ids', []) if target is profile['model'] else []
        model_ids = model_ids if isinstance(model_ids, list) else []
        if model['id'] not in [target.get('id'), *aliases, *model_ids]:
            continue
        mismatch = next((field for field in ('revision', 'format')
                         if target.get(field) and model.get(field) and target[field] != model[field]), None)
        if mismatch:
            reason = f'different model {mismatch}'
            continue
        return True, 'model match; live engine check required'
    return False, reason


def resolve_profile(value, model, candidates=()):
    """Resolve an operator-selected artifact or installed name without guessing IDs."""
    path = Path(value).expanduser()
    if path.exists():
        candidate = validate_profile(path)
    else:
        candidate = next((item for item in candidates if item['name'] == value), None)
        if candidate is None:
            if (not value or value == 'off' or len(value) > 256
                    or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value)):
                raise ProfileError('Invalid profile name')
            candidate = {'name': value, 'model': {'id': model['id']}, 'source': 'manual', 'artifact_valid': None}
    ok, reason = match_profile(candidate, model)
    if not ok:
        raise ProfileError('The profile does not match the selected model: ' + reason)
    return candidate
