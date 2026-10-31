# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Read-only selection of a compatible draft model without disabling active decoding."""
import os
from pathlib import Path

from .profiles import read_json
from .setup_runtime import SetupError


class MissingDraftModel(SetupError):
    pass


def argument(argv, name, default=None):
    for index in range(len(argv)-1, -1, -1):
        value = argv[index]
        if value.startswith(name+'='):
            return value.split('=', 1)[1]
        if value == name and index+1 < len(argv):
            return argv[index+1]
    return default


def active_drafting(model_path, model_id, proc_root=Path('/proc')):
    """Inspect matching owned TensorFold command lines; never read environment secrets."""
    found = []
    for process in proc_root.glob('[0-9]*'):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            argv = (process/'cmdline').read_bytes().decode().split('\0')
            if 'serve' not in argv or not any(Path(part).name in ('tensorfold', 'tensorfold-native', 'tensorfold-dsv41') for part in argv):
                continue
            index = argv.index('serve')
            if index+1 >= len(argv):
                continue
            current = Path(argv[index+1]).expanduser()
            cwd = (process/'cwd').resolve()
            if not current.is_absolute():
                current = cwd/current
            if current.resolve() != Path(model_path).resolve() and argv[index+1] != model_id:
                continue
            drafter = argument(argv, '--drafter')
            if drafter and drafter not in ('auto', 'none'):
                path = Path(drafter).expanduser()
                drafter = str(path if path.is_absolute() else cwd/path)
            parallel = argument(argv, '--parallel', '1')
            entry = {'draft_model': drafter, 'disabled': '--no-drafts' in argv,
                     'parallel': int('1' if parallel == 'auto' else parallel),
                     'tensor_parallel': int(argument(argv, '--tp', '1')),
                     'mtp_drafts': argument(argv, '--mtp-drafts')}
            if entry not in found:
                found.append(entry)
        except (OSError, UnicodeError, ValueError):
            continue
    if len(found) > 1:
        raise SetupError('Matching engines have different speculative settings; select the intended deployment explicitly')
    return found[0] if found else None


def draft_settings(home, spec, manifest, model_id, *, proc_root=Path('/proc')):
    if not manifest.get('requires_draft_model'):
        return {}
    active = active_drafting(spec['model_path'], model_id, proc_root)
    architectures = manifest.get('draft_model_architectures', [])
    def compatible(metadata):
        if not isinstance(metadata, dict):
            return False
        if architectures:
            declared = metadata.get('architectures', [])
            return (isinstance(declared, list) and all(isinstance(x, str) for x in declared)
                    and bool(set(architectures) & set(declared)))
        return metadata.get('speculators_model_type') == 'dspark'
    if active and not active['disabled']:
        if (active['parallel'] < manifest.get('min_parallel', 2)
                or active['draft_model'] in (None, '', 'auto', 'none')
                or (manifest.get('tensor_parallel_sizes')
                    and active['tensor_parallel'] not in manifest['tensor_parallel_sizes'])):
            raise SetupError('The active speculative configuration is unsupported by this profile package; it will not be disabled or replaced')
        if spec.get('draft_model') and Path(spec['draft_model']).resolve() != Path(active['draft_model']).resolve():
            raise SetupError('The selected draft model differs from the active engine; preserve its draft model')
        draft = active['draft_model']
    else:
        draft = spec.get('draft_model')
    if not draft:
        cache = Path(os.environ.get('HF_HOME', str(Path(home)/'.cache/huggingface')))/'hub'
        candidates = []
        for directory in sorted(cache.glob('models--*'))[:128]:
            for config in sorted((directory/'snapshots').glob('*/config.json'))[:8]:
                try:
                    metadata = read_json(config)
                    if compatible(metadata):
                        candidates.append(config.parent)
                except (OSError, ValueError):
                    continue
        if len(candidates) == 1:
            draft = candidates[0]
    if not draft:
        raise MissingDraftModel('A compatible local draft model is required; provide --drafter or engine.draft_model. Speculative decoding will not be disabled')
    path = Path(draft).expanduser().resolve()
    metadata = read_json(path/'config.json') if (path/'config.json').is_file() else None
    if not compatible(metadata):
        raise SetupError('This profile package requires a compatible draft model; existing speculation will not be disabled')
    result = {'draft_model': str(path)}
    if active and not active['disabled']:
        result['parallel'] = active['parallel']
        if active['mtp_drafts'] is not None:
            result['mtp_drafts'] = int(active['mtp_drafts'])
    return result
