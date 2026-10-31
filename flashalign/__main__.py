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
    if argv[:1] and argv[0] in ('chat', 'terminal', 'sessions'):
        from .terminal_cli import run
        return run(argv)
    parser=argparse.ArgumentParser(prog='flashalign-client',
        epilog='Interactive commands: chat (direct engine), terminal (native Hermes/OpenClaw), sessions (list/revoke).')
    parser.add_argument('--config',type=Path,default=Path.home()/'.config/flashalign/client.json')
    parser.add_argument('--owner',action='store_true',help='Explicitly use the configured owner profile')
    parser.add_argument('--scale',type=float,default=1.)
    parser.add_argument('--max-tokens',type=int,default=256)
    args=parser.parse_args(argv)
    from owner_mode.profiles import read_json
    config=read_json(args.config)
    with Client.from_config(args.config) as client:
        result=client.chat([{'role':'user','content':sys.stdin.read()}],
                           profile=config['profile'] if args.owner else None,
                           scale=args.scale,max_tokens=args.max_tokens)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    sys.exit(main())
