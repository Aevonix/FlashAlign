# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launch the qualified Gemma4 FP8 backend from a matching local checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from runpy import run_path
import struct
import sys

ROOT = Path(__file__).resolve().parent
_arguments = run_path(str(ROOT/'flashalign_vllm/arguments.py'))
normalize_arguments, option_name = _arguments['normalize_arguments'], _arguments['option_name']
require_full_options = _arguments['require_full_options']


def fingerprint(model, entry):
    raw = (model/'config.json').read_bytes()
    config = json.loads(raw)
    if config.get('model_type') != entry['model_type']:
        return False
    if hashlib.sha256(raw).hexdigest() != entry['config_sha256']:
        return False
    quant = json.dumps(config.get('quantization_config'), sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(quant).hexdigest() != entry['quantization_config_sha256']:
        return False
    spec = entry['weights_header']
    if Path(spec['file']).name != spec['file']:
        return False
    path = model/spec['file']
    if path.stat().st_size != spec['size'] or sorted(p.name for p in model.glob('*.safetensors')) != [spec['file']]:
        return False
    if list(model.glob('*.safetensors.index.json')):
        return False
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 2 <= length <= 32*1024*1024:
            return False
        header = stream.read(length)
    return len(header) == length and hashlib.sha256(header).hexdigest() == spec['sha256']


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--profile', type=Path, action='append', required=True)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8030)
    p.add_argument('--speculative-config')
    p.add_argument('--draft-model')
    p.add_argument('--draft-method')
    p.add_argument('--dry-run', action='store_true')
    args, extra = p.parse_known_args(normalize_arguments(sys.argv[1:]))
    if extra[:1] == ['--']: extra = extra[1:]
    reserved = {'--hf-overrides', '--hf-config-path', '--quantization', '-q', '--dtype', '--config',
                '--model', '--profile', '--key-file', '--speculative-config', '--draft-model', '--draft-method'}
    try: require_full_options(extra, reserved | set(p._option_string_actions) | {'--max-model-len'})
    except ValueError as error: p.error(str(error))
    if any(option_name(x) in reserved for x in extra):
        p.error('Model metadata/quantization overrides are not qualified by this backend')
    metadata = json.loads((ROOT/'backend.json').read_text())
    model = args.model.resolve()
    try:
        matched = [entry for entry in metadata['compatible_models'] if fingerprint(model, entry)]
    except (OSError, ValueError, struct.error):
        p.error('Local checkpoint metadata is missing or malformed')
    if len(matched) != 1:
        p.error('Local model metadata does not match the qualified Gemma4 FP8 checkpoint')
    identity = matched[0]
    for path in args.profile:
        document = json.loads((path/'profile.json').read_text())
        if document.get('model') != {k: identity[k] for k in ['id', 'revision']}:
            p.error('Profile targets a different checkpoint')
        if document.get('transform', 'projection') != 'projection' or document.get('requires', []) != []:
            p.error('This Gemma4 backend variant accepts standard projection profiles only')
        for site in document.get('sites', []):
            pattern = metadata['site_tensor_patterns'].get(site.get('site'))
            if pattern is None or re.fullmatch(pattern, site.get('tensor', '')) is None:
                p.error('Profile includes an unsupported Gemma4 output site')
    if args.draft_model or args.draft_method:
        p.error('External draft models are unqualified for this variant; the requested method will not be disabled')
    spec = json.loads(args.speculative_config) if args.speculative_config else metadata['default_speculative_config']
    if spec.get('method') not in metadata['tested_speculative_methods']:
        p.error('Active speculative method is unqualified; it will not be disabled or replaced')
    command = [sys.executable, '-m', 'flashalign_vllm', '--model', str(model), '--key-file', str(args.key_file.resolve()),
               '--host', args.host, '--port', str(args.port), '--speculative-config', json.dumps(spec, separators=(',', ':'))]
    for path in args.profile: command += ['--profile', str(path.resolve())]
    command += metadata['default_arguments'] + extra
    if args.dry_run:
        print(json.dumps({'backend': 'vllm', 'variant': 'gemma4', 'command': command,
                          'model_match': identity['id'], 'check': 'metadata compatibility; not a payload integrity check'}))
        return
    os.environ['PYTHONPATH'] = str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', '')
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
