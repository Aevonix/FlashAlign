# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launch the pinned GLM Flash backend with native speculative decoding."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from installation import validate_prepared_source
from verify_checkpoint import verify

ROOT = Path(__file__).resolve().parent


def validate_runtime_versions(version=importlib.metadata.version):
    for name, wanted in {'sglang': '0.5.20', 'torch': '2.13.0', 'triton': '3.7.1',
                         'tilelang': '0.1.12', 'safetensors': '0.8.0',
                         'transformers': '5.12.1'}.items():
        if version(name).split('+', 1)[0] != wanted:
            raise ValueError('Runtime dependency version mismatch: ' + name)


def check_extra(extra):
    switches = {'--enable-cache-report', '--enable-metrics'}
    values = {'--served-model-name', '--context-length', '--max-running-requests',
              '--mem-fraction-static', '--chunked-prefill-size', '--log-level'}
    index = 0
    while index < len(extra):
        name, sep, value = extra[index].partition('=')
        if name in switches and not sep:
            index += 1
        elif name in values:
            if not sep:
                index += 1
                if index == len(extra) or extra[index].startswith('--'):
                    raise ValueError('Missing argument value: ' + name)
            index += 1
        else:
            raise ValueError('Unsupported GLM backend option: ' + name)


def build_command(args, extra):
    metadata = json.loads((ROOT / 'backend.json').read_text())
    model = Path(args.model).resolve()
    if args.draft_model is not None and Path(args.draft_model).resolve() != model:
        raise ValueError('Native GLM MTP must use the same stock checkpoint')
    reserved = {value for value in metadata['fixed_launch_arguments'] if value.startswith('--')}
    reserved.update(('--model-path', '--host', '--port', '--speculative-algorithm',
                     '--speculative-draft-model-path', '--disable-speculative-decoding', '--no-drafts'))
    if any(value.split('=', 1)[0] in reserved for value in extra):
        raise ValueError('Extra arguments cannot replace the GLM topology or native speculation')
    check_extra(extra)
    expected = json.loads((ROOT / 'model-metadata.json').read_text())
    actual = {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
              for name in expected['sha256']}
    if actual != expected['sha256']:
        raise ValueError('Model fingerprints do not match the supported GLM checkpoint')
    command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(model),
               '--host', args.host, '--port', str(args.port), *metadata['fixed_launch_arguments'],
               '--speculative-algorithm', args.draft_algorithm,
               '--speculative-draft-model-path', str(model), *extra]
    return command, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--profile', action='append', required=True)
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--stock-receipt', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8030)
    parser.add_argument('--draft-model')
    parser.add_argument('--draft-algorithm', choices=('EAGLE', 'NEXTN'), default='EAGLE')
    parser.add_argument('--print-command', action='store_true')
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    try:
        validate_prepared_source(args.source)
        command, expected = build_command(args, extra)
        validate_runtime_versions()
        verify(args.model, args.stock_receipt)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.update(FLASHALIGN_MODEL_SCOPE=hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                      FLASHALIGN_PROFILES=','.join(str(Path(p).resolve()) for p in args.profile),
                      FLASHALIGN_KEY_FILE=str(Path(args.key_file).resolve()),
                      FLASHALIGN_REPLAY_STORE=str(args.state.resolve() / 'replay.sqlite3'))
    os.environ['PYTHONPATH'] = str(args.source.resolve() / 'python') + os.pathsep + os.environ.get('PYTHONPATH', '')
    if args.print_command:
        print(json.dumps({'command': command, 'native_speculation': args.draft_algorithm}))
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        parser.error('The GLM Flash backend requires exactly four visible GPUs')
    if any(torch.cuda.get_device_capability(index) not in ((12, 0), (12, 1)) for index in range(4)):
        parser.error('The GLM Flash backend requires the supported SM120/SM121 devices')
    from supervisor import memory_headroom, supervise
    if memory_headroom()['effective_available_bytes'] < 64 * 1024**3:
        parser.error('Host/cgroup memory is below the 64 GiB runtime floor')
    raise SystemExit(supervise(command, 64 * 1024**3))


if __name__ == '__main__':
    main()
