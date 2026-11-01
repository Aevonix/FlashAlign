# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Add FlashAlign to an explicitly supplied native DeepSeek command."""
import argparse
import json
import os
from pathlib import Path
import sys

from .core import load_key, load_profiles

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'


def served_names(arguments):
    matches = [i for i, value in enumerate(arguments) if value == '--served-model-name']
    if len(matches) != 1:
        raise ValueError('Declare exactly one native --served-model-name argument')
    names = []
    for value in arguments[matches[0]+1:]:
        if value.startswith('-'):
            break
        names.append(value)
    if not names or len(names) != len(set(names)):
        raise ValueError('Explicit nonempty unique served model aliases required')
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--replay-file', type=Path, required=True)
    parser.add_argument('--canonical-model', choices=[MODEL], required=True)
    parser.add_argument('--allow-research', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--native-receipts', type=Path,
                        help='Precreated private coverage directory for a native matrix profile')
    parser.add_argument('native', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    native = args.native[1:] if args.native[:1] == ['--'] else args.native
    if native[:1] != ['serve']:
        parser.error('Pass the original native serve command after --')
    if '--middleware' in native:
        parser.error('The initial integration owns its one API middleware declaration')
    names = served_names(native)
    load_key(args.key_file)
    profiles = load_profiles([args.profile], allow_research=args.allow_research)
    native_requested = any(getattr(profile, 'is_native_wo', False) for profile in profiles.values())
    if native_requested:
        directory = args.native_receipts
        if (directory is None or not directory.is_absolute() or directory.is_symlink()
                or not directory.is_dir() or directory.stat().st_uid != os.getuid()
                or directory.stat().st_mode & 0o077):
            parser.error('Native profiles require --native-receipts with a private absolute directory')
        os.environ['FLASHALIGN_NATIVE_RECEIPTS'] = str(directory.resolve())
    for profile in profiles.values():
        if profile.meta['model']['id'] != args.canonical_model:
            parser.error('The profile canonical model differs')
        if getattr(profile, 'is_native_wo', False):
            profile.validate_payload()
    prior_plugins = os.environ.get('VLLM_PLUGINS')
    existing = [name for name in (prior_plugins or '').split(',') if name]
    if native_requested:
        if any(name != 'flashalign_native' for name in existing):
            parser.error('The native profile requires only the flashalign_native plugin')
        prior_plugins = 'flashalign_native'
        existing = ['flashalign_native']
    os.environ.update(
        FLASHALIGN_DEEPSEEK_PROFILES=json.dumps([str(args.profile.resolve())]),
        FLASHALIGN_DEEPSEEK_KEY_FILE=str(args.key_file.resolve()),
        FLASHALIGN_DEEPSEEK_MODEL_ID=args.canonical_model,
        FLASHALIGN_NATIVE_SERVED_MODELS=json.dumps(names),
        FLASHALIGN_DEEPSEEK_ALLOW_RESEARCH='1' if args.allow_research else '0',
        FLASHALIGN_NATIVE_SECURE_LAUNCH='1',
        FLASHALIGN_REPLAY_STORE=str(args.replay_file.resolve()),
        VLLM_USE_V2_MODEL_RUNNER='1')
    # An absent filter means all installed general plugins. Preserve it.
    if prior_plugins is not None:
        os.environ['VLLM_PLUGINS'] = ','.join(dict.fromkeys([*existing, 'flashalign_native']))
    from .plugin import register
    register()
    sys.argv = ['vllm', *native, '--middleware', 'flashalign_native.middleware.OwnerMiddleware']
    from vllm.entrypoints.cli.main import main as native_main
    native_main()
