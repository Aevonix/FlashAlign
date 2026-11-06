# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Engine-independent local profile lifecycle client. No generation signing key."""
import argparse
import json
from pathlib import Path


from owner_mode.profiles import read_json
from owner_mode.terminal_ui import TerminalUI


from owner_mode.profile_admin_client import AdminClient, LifecycleRequestError


def main(argv=None):
    parser = argparse.ArgumentParser(prog='flashalign profile')
    parser.add_argument('action', choices=('list', 'stage', 'activate', 'rollback', 'retire',
                                           'recover', 'evict-cache', 'deactivate'))
    parser.add_argument('target', nargs='?')
    parser.add_argument('--generation')
    parser.add_argument('--expected-revision', type=int)
    parser.add_argument('--socket', type=Path)
    parser.add_argument('--staging-directory', type=Path,
                        help='Private shared input directory for a container engine')
    parser.add_argument('--config', type=Path, default=Path.home()/'.config/flashalign/client.json')
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    action, fields = args.action.replace('-', '_'), {}
    if action in ('stage', 'retire', 'activate', 'rollback', 'deactivate') and not args.target:
        parser.error('This operation requires a target')
    if action in ('activate', 'rollback', 'deactivate'):
        if args.expected_revision is None or args.expected_revision < 0:
            parser.error('Supply --expected-revision from profile list')
        fields.update(alias=args.target, expected_revision=args.expected_revision)
        if action != 'deactivate':
            if not args.generation:
                parser.error('Supply --generation from profile stage')
            fields['generation'] = args.generation
    elif action == 'stage':
        fields['path'] = str(Path(args.target).expanduser().absolute())
    elif action == 'retire':
        fields['generation'] = args.target
    elif args.target or args.generation or args.expected_revision is not None:
        parser.error('This operation does not accept a target or revision')
    config = read_json(args.config) if args.config.exists() else {}
    path = args.socket
    if path is None:
        path = config.get('admin_socket')
        if not isinstance(path, str) or not path:
            parser.error('Set admin_socket in the client configuration or supply --socket')
    with AdminClient(path, timeout=args.timeout) as client:
        if action == 'stage':
            remote = config.get('admin_remote') if str(path) == config.get('admin_socket') else None
            result = client.stage(fields['path'], remote=remote, staging_directory=(
                args.staging_directory or (config.get('admin_staging_directory') if remote is None else None)))
        else:
            result = client.request(None if action == 'list' else action, **fields)
    if args.json:
        print(json.dumps(result, allow_nan=False))
    else:
        ui = TerminalUI(animation=False)
        ui.result('Profile operation completed')
        ui.line(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
