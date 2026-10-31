# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Run Hermes's native interactive CLI with an explicit terminal session."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('owner_mode', root/'__init__.py', submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from owner_mode.terminal import TerminalSessionStore, native_hermes_session, terminal_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-fd', type=int, required=True)
    args = parser.parse_args()
    with os.fdopen(args.receipt_fd) as stream:
        receipt = json.loads(stream.read(8193))
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError('Native terminal sessions require an interactive terminal')
    sessions = TerminalSessionStore(receipt['approvals'])
    if not sessions.active(receipt['id'], harness='hermes'):
        raise ValueError('Terminal session is not active')
    os.environ['HERMES_HOME'] = str(Path(receipt['home'])/'.hermes')
    with terminal_launch(sessions, receipt['id']):
        from cli import HermesCLI
        from run_agent import AIAgent
        from hermes_cli.main import main as hermes_main
        with native_hermes_session(sessions, receipt['id'], HermesCLI, AIAgent):
            sys.argv = ['hermes', 'chat']
            return hermes_main()


if __name__ == '__main__':
    main()
