# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Validate a pinned TensorFold runtime without executing it."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

from .profiles import read_json, NATIVE_FORMAT, NATIVE_REQUIRES, _compatible_models, ProfileError
from .setup_runtime import SetupError

VERSIONS = frozenset(('1.1.9', '1.2.0'))
PLATFORMS = {'linux-x86_64': 62, 'linux-aarch64': 183}
MAX_FILE = 2 * 1024**3
MAX_TOTAL = 4 * 1024**3


def host_platform():
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine().lower(), platform.machine().lower())
    return platform.system().lower() + '-' + machine


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(name):
    if not isinstance(name, str) or not name or '\\' in name or any(ord(c) < 32 for c in name):
        raise SetupError('Invalid native runtime path')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise SetupError('Invalid native runtime path')
    if path.name == 'profile.json' or path.suffix == '.safetensors' or 'libcuda.so' in path.name:
        raise SetupError('Runtime bundles cannot contain profiles or GPU drivers')
    if path.parts[0] not in ('bin', 'lib', 'assets', 'licenses', 'sources') and name not in ('LICENSE', 'NOTICE.md', 'COMMERCIAL.md'):
        raise SetupError('Native runtime files must be in bin, lib, assets, licenses or sources')
    return name


def records(manifest):
    if type(manifest.get('startup_timeout', 1800)) is not int or not 60 <= manifest.get('startup_timeout', 1800) <= 7200:
        raise SetupError('Native startup_timeout must be between 60 and 7200 seconds')
    files = manifest.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= 4096:
        raise SetupError('Native runtime needs a complete file inventory')
    total = 0
    for name, row in files.items():
        relative(name)
        if (not isinstance(row, dict) or set(row) - {'sha256', 'bytes', 'executable'}
                or not isinstance(row.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])
                or type(row.get('bytes')) is not int or not 0 <= row['bytes'] <= MAX_FILE
                or type(row.get('executable', False)) is not bool):
            raise SetupError('Invalid native runtime file checksum or size')
        total += row['bytes']
    if total > MAX_TOTAL or not {'LICENSE', 'NOTICE.md'} <= files.keys():
        raise SetupError('Native runtime inventory exceeds limits or omits license notices')
    return files


def validate_manifest(manifest, *, platform_id=None):
    if (not isinstance(manifest, dict) or manifest.get('version') != 2
            or manifest.get('engine') != 'tensorfold-native' or manifest.get('runtime_version') not in VERSIONS
            or not isinstance(manifest.get('source_revision'), str)
            or not re.fullmatch('[0-9a-f]{40}', manifest['source_revision'])):
        raise SetupError('Use a TensorFold 1.1.9 or 1.2.0 runtime bundle with its exact source revision')
    if manifest.get('durable_owner_replay') is not True:
        raise SetupError('Native runtime requires durable_owner_replay: true')
    target = manifest.get('platform')
    if target not in PLATFORMS or platform_id is not None and target != platform_id:
        raise SetupError('Native runtime architecture does not match the selected host')
    architectures = manifest.get('cuda_architectures')
    if (not isinstance(architectures, list) or not architectures
            or any(not isinstance(x, str) or not re.fullmatch(r'(sm|compute)_\d{2,3}a?', x) for x in architectures)):
        raise SetupError('Native runtime must declare its CUDA architectures')
    files = records(manifest)
    source = manifest.get('source_archive')
    if not isinstance(source, str) or not source.startswith('sources/') or source not in files:
        raise SetupError('Native runtime must include the source archive for its exact source revision')
    auxiliary = manifest.get('auxiliary_sources', [])
    if (not isinstance(auxiliary, list) or any(not isinstance(item, dict)
            or set(item) != {'archive', 'license'}
            or any(not isinstance(item.get(key), str) or not item[key].startswith('sources/')
                   or item[key] not in files for key in ('archive', 'license')) for item in auxiliary)):
        raise SetupError('Auxiliary runtime sources must include inventoried source and license files')
    models, transforms = manifest.get('model_types'), manifest.get('transforms')
    if (not isinstance(models, list) or not models or any(not isinstance(x, str) or not x for x in models)
            or len(set(models)) != len(models) or not isinstance(transforms, list) or not transforms
            or any(x not in ('projection', 'row_gain_projection_v1', 'row_normalized_projection_v1',
                             'input_low_rank_delta_v1', 'native_matrix_override_v1') for x in transforms)):
        raise SetupError('Native runtime must declare model and operator compatibility')
    schemas = manifest.get('profile_schemas')
    if schemas is not None and (not isinstance(schemas, list) or not schemas
            or any(not isinstance(item, str) or not item for item in schemas)
            or len(set(schemas)) != len(schemas)):
        raise SetupError('Native runtime profile schemas must be an explicit unique list')
    if 'native_matrix_override_v1' in transforms or NATIVE_FORMAT in (schemas or []):
        _native_contract(manifest)
    for name in ('binary', 'follower_binary'):
        if name not in manifest and name == 'follower_binary':
            continue
        path = manifest.get(name)
        if path not in files or not path.startswith('bin/') or not files[path].get('executable'):
            raise SetupError('Native runtime executable is missing from its file inventory')
    if manifest.get('rank_transport') not in ('cli', 'glm_env', 'dsv41_env'):
        raise SetupError('Native runtime must declare its rank transport')
    if manifest['rank_transport'] == 'dsv41_env':
        contract = manifest.get('auxiliary_engram')
        source = contract.get('source') if isinstance(contract, dict) else None
        if (not isinstance(source, dict) or not isinstance(source.get('id'), str) or not source['id']
                or not isinstance(source.get('revision'), str) or not re.fullmatch('[0-9a-f]{40}', source['revision'])
                or any(not isinstance(source.get(k), str) or not re.fullmatch('[0-9a-f]{64}', source[k]) for k in ('config_sha256','index_sha256'))):
            raise SetupError('DeepSeek bundle requires its pinned public auxiliary Engram source')
        if 'follower_binary' not in manifest or not isinstance(manifest.get('assets_directory'), str):
            raise SetupError('DeepSeek native runtime requires its follower binary and immutable assets')
        geometry = manifest.get('asset_geometry')
        if (not isinstance(geometry,dict) or any(type(geometry.get(k)) is not int or geometry[k] < 1
                for k in ('context','tensor_parallel','pool_tokens','slots')) or geometry['pool_tokens'] < geometry['context']):
            raise SetupError('DeepSeek native bundle requires explicit generated-asset geometry')
        prefix = manifest['assets_directory'].rstrip('/') + '/'
        if not prefix.startswith('assets/') or not any(name.startswith(prefix) for name in files):
            raise SetupError('DeepSeek native assets are missing')
    sizes = manifest.get('tensor_parallel_sizes')
    if (not isinstance(sizes, list) or not sizes or any(type(n) is not int or n not in (1, 2, 4, 8, 16) for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise SetupError('Native runtime must declare qualified tensor-parallel sizes')
    if any(n > 1 for n in sizes) and not any(name.startswith('lib/libnccl.so') for name in files):
        raise SetupError('Tensor-parallel bundles must include the licensed NCCL runtime')
    if 'lib/libtfgrammar.so' not in files:
        raise SetupError('Native runtime requires libtfgrammar.so for structured output and required tools')
    for name in files:
        if name.startswith('lib/') and '.so' in Path(name).name:
            if 'licenses/'+Path(name).name+'.txt' not in files:
                raise SetupError('Every bundled runtime library requires its license and notices')
    arguments = manifest.get('arguments', [])
    if not isinstance(arguments, list) or any(not isinstance(x, str) for x in arguments):
        raise SetupError('Native runtime arguments must be an argument array')
    if any(x.split('=', 1)[0] in ('--no-drafts', '--no-cuda-graphs') for x in arguments):
        raise SetupError('Native runtime defaults must preserve speculative decoding and CUDA graphs')
    if type(manifest.get('mtp_drafts_tunable', True)) is not bool:
        raise SetupError('Native MTP tuning capability must be boolean')
    methods = manifest.get('speculative_methods')
    if (not isinstance(methods, list) or not methods or
            any(x not in ('mtp', 'dflash', 'dspark', 'ngram') for x in methods)):
        raise SetupError('Native runtime must declare supported speculative methods')
    return manifest


def _native_contract(manifest):
    """The profile target and the TensorFold serving base are separate pins."""
    if (manifest.get('runtime_version') != '1.2.0'
            or NATIVE_FORMAT not in manifest.get('profile_schemas', [])
            or 'native_matrix_override_v1' not in manifest.get('transforms', [])
            or manifest.get('supported_profile_requires') != list(NATIVE_REQUIRES)
            or manifest.get('supported_profile_strengths') != [1.0]
            or type(manifest['supported_profile_strengths'][0]) not in (int, float)
            or manifest.get('tensor_parallel_sizes') != [4]):
        raise SetupError('Native matrix profiles require the explicit TensorFold 1.2.0 contract')
    target = manifest.get('profile_target_model')
    if (not isinstance(target, dict) or set(target) != {'id', 'revision'}
            or not isinstance(target['id'], str) or not target['id']
            or not isinstance(target['revision'], str) or not re.fullmatch('[0-9a-f]{40}', target['revision'])):
        raise SetupError('Native matrix runtime must pin the original profile target model')
    try:
        bases = _compatible_models(manifest.get('compatible_profile_bases'))
    except ProfileError as error:
        raise SetupError('Native matrix runtime must declare complete compatible serving base fingerprints') from error
    if not bases or any(row.get('format') != 'EXL3' for row in bases):
        raise SetupError('Native matrix runtime must explicitly declare its compatible EXL3 serving bases')
    return bases


def check_profile_contract(manifest, profile):
    schema = profile.get('schema') or 'standard'
    schemas = manifest.get('profile_schemas')
    if schemas is not None and schema not in schemas:
        raise SetupError('This native runtime does not support the selected profile schema')
    if profile.get('format') == NATIVE_FORMAT:
        _native_contract(manifest)
        if profile.get('model') != manifest['profile_target_model']:
            raise SetupError('The native profile target differs from the runtime declaration')
        if profile.get('requires') != manifest['supported_profile_requires']:
            raise SetupError('The native runtime does not implement every required profile capability')
        spec = read_json(profile['manifest'])
        if (spec.get('strengths') != manifest['supported_profile_strengths']
                or spec.get('tp_world_size') not in manifest['tensor_parallel_sizes']):
            raise SetupError('The native profile strength or tensor-parallel layout is unsupported')


def check_bundle(path, model, profile, *, platform_id=None):
    root = Path(path).expanduser().resolve()
    manifest = validate_manifest(read_json(root/'manifest.json'), platform_id=platform_id or host_platform())
    if model.get('model_type') not in manifest['model_types']:
        raise SetupError('This native runtime does not support the selected model family')
    if profile['transform'] not in manifest['transforms']:
        raise SetupError('This native runtime does not support the selected profile transform')
    check_profile_contract(manifest, profile)
    files = manifest['files']
    actual = set()
    for item in root.rglob('*'):
        info = item.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        name = str(item.relative_to(root))
        if name in ('manifest.json', 'asset-receipt.json'):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SetupError('Native runtime accepts only regular files, without links')
        actual.add(name)
        row = files.get(name)
        if row is None or info.st_size != row['bytes'] or sha256(item) != row['sha256']:
            raise SetupError('Native runtime file inventory or checksum differs')
    if actual != set(files):
        raise SetupError('Native runtime file inventory is incomplete')
    binaries = [manifest[name] for name in ('binary', 'follower_binary') if name in manifest]
    binaries += [name for name in files if name.startswith('lib/') and '.so' in Path(name).name]
    for name in binaries:
        with (root/name).open('rb') as f:
            elf = f.read(20)
        if (len(elf) < 20 or elf[:6] != b'\x7fELF\x02\x01'
                or int.from_bytes(elf[18:20], 'little') != PLATFORMS[manifest['platform']]):
            raise SetupError('Native executable ELF architecture differs from its manifest')
    return root, manifest
