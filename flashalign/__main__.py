# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Send a direct model request using a prepared deployment's client settings."""
import argparse
import json
from pathlib import Path
import sys

from . import Client


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['setup']:
        from owner_mode.setup import cli as setup_main
        return setup_main(argv[1:])
    if argv[:1] and argv[0] in ('status', 'doctor', 'dashboard'):
        from .operations import main as operations_main
        return operations_main(argv)
    if argv[:1] == ['why']:
        from owner_mode.audit import main as why_main
        return why_main(argv[1:])
    if argv[:1] == ['benchmark']:
        from .benchmark import main as benchmark_main
        return benchmark_main(argv[1:])
    if argv[:1] == ['profile']:
        if len(argv) > 1 and argv[1] in ('evaluate', 'sweep'):
            from .profile_evaluation import main as evaluation_main
            return evaluation_main(argv[1:])
        if len(argv) > 1 and argv[1] in ('list', 'stage', 'activate', 'rollback', 'retire',
                                       'recover', 'evict-cache', 'deactivate'):
            from .profile_admin import main as admin_main
            return admin_main(argv[1:])
        from owner_mode.profile_inspector import main as profile_main
        return profile_main(argv[1:])
    if argv[:1] and argv[0] in ('chat', 'terminal', 'sessions'):
        from .terminal_cli import run
        return run(argv)
    parser=argparse.ArgumentParser(prog='flashalign-client',
        epilog='Setup: flashalign setup. Diagnostics: status, doctor, why. Interactive commands: chat (direct engine), terminal (native Hermes/OpenClaw), sessions (list/revoke).')
    from owner_mode import __version__
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--config',type=Path,default=Path.home()/'.config/flashalign/client.json')
    parser.add_argument('--owner',action='store_true',help='Explicitly use the configured owner profile')
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--max-tokens',type=int,default=None)
    args=parser.parse_args(argv)
    from owner_mode.profiles import read_json
    config=read_json(args.config)
    with Client.from_config(args.config) as client:
        options = {'max_tokens': args.max_tokens} if args.max_tokens is not None else {}
        result=client.chat([{'role':'user','content':sys.stdin.read()}],
                           profile=config['profile'] if args.owner else None,
                           scale=args.scale,**options)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    sys.exit(main())
