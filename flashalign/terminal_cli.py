# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Interactive direct chat and explicitly authorized native terminal launchers."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from owner_mode.contacts import ApprovalStore
from owner_mode.terminal import CHANNEL, TerminalSessionStore
from owner_mode.terminal_ui import TerminalUI, clean_text


def _interface(args):
    return TerminalUI(color=not getattr(args, 'no_color', False),
                      animation=not getattr(args, 'no_animation', False))


def _label(value):
    return ' '.join(clean_text(value).splitlines())


def _chat_selection(config, args):
    """Describe the requested mode without claiming engine confirmation."""
    always_on = config.get('mode', 'verified') == 'always_on'
    selected = config.get('profile') if always_on or args.owner else 'off'
    access = 'Always On' if always_on else 'Owner request' if args.owner else 'Standard request'
    return selected, access


def direct_chat(args):
    from . import Client
    from owner_mode import __version__
    from owner_mode.profiles import read_json
    config = read_json(args.config)
    history = []
    ui = _interface(args)
    selected, access = _chat_selection(config, args)
    with Client.from_config(args.config) as client:
        ui.banner(__version__, subtitle='Direct chat')
        ui.context(model=config.get('model', 'Configured model'), profile=selected or 'Not configured',
                   strength=args.scale if selected != 'off' else 'Not applied', access=access)
        ui.line('MODEL  ' + _label(config.get('model', 'Configured model')))
        ui.line('REQUESTED  ' + access + ' / profile: ' + _label(selected or 'Not configured'))
        if selected != 'off':
            ui.line('STRENGTH  ' + str(args.scale))
        ui.line('Engine confirmation is checked on every reply.')
        ui.line('Direct engine chat. /reset clears history; /exit ends the chat.')
        ui.line()
        while True:
            try:
                prompt = input('You: ')
            except (EOFError, KeyboardInterrupt):
                ui.line(); break
            if prompt.strip() in ('/exit', '/quit'):
                break
            if prompt.strip() == '/reset':
                history.clear()
                ui.line('Conversation cleared.')
                continue
            if not prompt.strip():
                continue
            history.append({'role':'user', 'content':prompt})
            options = {'max_tokens':args.max_tokens} if args.max_tokens is not None else {}
            result = client.chat(history, profile=config['profile'] if args.owner else None,
                                 scale=args.scale, **options)
            answer = result['choices'][0]['message']
            history.append(answer)
            # Client.chat returns only after its HTTP header check confirms this
            # effective profile. Keep original message content in history;
            # terminal control sequences are removed only from its display.
            ui.result('Last response profile: ' + _label(selected) + ' (confirmed)')
            ui.line('Assistant: ' + str(answer.get('content') or ''))
            ui.line()
    return 0


def _hermes_python(explicit=None):
    if explicit:
        return explicit
    executable = shutil.which('hermes')
    if executable:
        with Path(executable).open('rb') as stream:
            first = stream.readline(4096).decode('utf-8')
        if first.startswith('#!'):
            parts = shlex.split(first[2:].strip())
            if parts and Path(parts[0]).name == 'env' and len(parts) == 2:
                return shutil.which(parts[1]) or parts[1]
            if len(parts) == 1 and 'python' in Path(parts[0]).name:
                return parts[0]
    raise ValueError('Cannot find the Hermes Python interpreter; supply --harness-python')


def _openclaw_root(explicit=None):
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        executable = shutil.which('openclaw')
        if not executable:
            raise ValueError('OpenClaw is not installed; supply --openclaw-root')
        path = Path(executable).resolve()
        root = next((parent for parent in path.parents if (parent/'package.json').is_file()
                     and json.loads((parent/'package.json').read_text()).get('name') == 'openclaw'), None)
        if root is None:
            raise ValueError('Cannot locate the installed OpenClaw package; supply --openclaw-root')
    if not (root/'dist').is_dir():
        raise ValueError('OpenClaw has no installed native terminal runtime')
    return root


def native_terminal(args):
    from owner_mode import __version__
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError('Native terminal sessions require an interactive terminal')
    if not args.authorize:
        raise ValueError('Use --authorize to grant this expiring local terminal session profile access')
    if not 60 <= args.duration <= 86400:
        raise ValueError('Session duration must be from 60 to 86400 seconds')
    home = args.home.expanduser().resolve()
    approvals = ApprovalStore(home/'.config/owner-mode/approvals.json')
    state = approvals.read(require_existing=True)
    if state.get('mode') != 'verified':
        raise ValueError('Session-scoped authorization requires Verified mode; Always On applies independently of session grants')
    if not state['enabled']:
        raise ValueError('FlashAlign is disabled in this deployment')
    source = Path(__file__).resolve().parents[1]
    import owner_mode
    core = Path(owner_mode.__file__).resolve().parent
    if args.harness == 'hermes':
        command = [_hermes_python(args.harness_python), str(core/'native_terminal.py'), '--receipt-fd']
        extra = {}
    else:
        node = shutil.which('node')
        if node is None:
            raise ValueError('Node is required for the native OpenClaw terminal')
        launcher = source/'openclaw-owner-mode/terminal-runner.mjs'
        if not launcher.is_file():
            launcher = home/'.openclaw/extensions/owner-mode/terminal-runner.mjs'
        if not launcher.is_file():
            raise ValueError('Install the FlashAlign OpenClaw terminal adapter through setup first')
        command = [node, str(launcher)]
        extra = {'root':str(_openclaw_root(args.openclaw_root))}
    approvals.set_channels(sorted(set(approvals.channels()) | {CHANNEL}))
    approvals.set_contact(CHANNEL, args.harness, 'uid:'+str(os.getuid()), True)
    sessions = TerminalSessionStore(approvals)
    row = sessions.create(args.harness, ttl=args.duration)
    read_fd, write_fd = os.pipe()
    try:
        # The session grant is already live. Keep the banner inside the cleanup
        # scope so interrupting the animation still revokes that grant.
        ui = _interface(args)
        ui.banner(__version__, subtitle=args.harness.title() + ' terminal')
        ui.line(f"Authorized {args.harness} terminal session {row['id']} for {args.duration} seconds.")
        ui.line(f"Revoke from another terminal: flashalign-client sessions --home {shlex.quote(str(home))} revoke {row['id']}")
        ui.line('Opening the harness terminal. Its interface and tool permissions apply.')
        sys.stdout.flush()
        child = subprocess.Popen([*command, str(read_fd)], pass_fds=(read_fd,))
        os.close(read_fd); read_fd = None
        with os.fdopen(write_fd, 'w') as stream:
            write_fd = None
            json.dump({'id':row['id'], 'home':str(home), 'approvals':str(approvals.path), **extra}, stream)
        try:
            return child.wait()
        except KeyboardInterrupt:
            return child.wait()
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        sessions.revoke(row['id'])


def run(argv):
    parser = argparse.ArgumentParser(prog='flashalign-client')
    commands = parser.add_subparsers(dest='command', required=True)
    chat = commands.add_parser('chat', help='Interactive direct engine chat')
    chat.add_argument('--config', type=Path, default=Path.home()/'.config/flashalign/client.json')
    chat.add_argument('--owner', action='store_true')
    chat.add_argument('--scale', type=float, default=1.)
    chat.add_argument('--max-tokens', type=int, help='Optional output limit; otherwise use engine defaults')
    chat.add_argument('--no-color', action='store_true', help='Use plain terminal output')
    chat.add_argument('--no-animation', action='store_true', help='Skip the startup animation')
    terminal = commands.add_parser('terminal', help='Launch an authorized native Hermes or OpenClaw interactive terminal')
    terminal.add_argument('--home', type=Path, default=Path.home())
    terminal.add_argument('--harness', choices=('hermes','openclaw'), required=True)
    terminal.add_argument('--authorize', action='store_true')
    terminal.add_argument('--duration', type=int, default=3600, help='Authorization duration in seconds, at most 86400')
    terminal.add_argument('--harness-python', help='Hermes Python interpreter')
    terminal.add_argument('--openclaw-root', type=Path, help='Installed OpenClaw package directory')
    terminal.add_argument('--no-color', action='store_true', help='Use plain terminal output')
    terminal.add_argument('--no-animation', action='store_true', help='Skip the startup animation')
    sessions = commands.add_parser('sessions', help='Inspect or revoke local terminal sessions')
    sessions.add_argument('--home', type=Path, default=Path.home())
    actions = sessions.add_subparsers(dest='action', required=True)
    actions.add_parser('list')
    revoke = actions.add_parser('revoke'); revoke.add_argument('session')
    args = parser.parse_args(argv)
    if args.command == 'chat':
        return direct_chat(args)
    if args.command == 'terminal':
        return native_terminal(args)
    store = TerminalSessionStore(args.home/'.config/owner-mode/approvals.json')
    if args.action == 'revoke':
        store.revoke(args.session)
        print('Terminal session authorization revoked.')
    else:
        rows = store.read()['sessions']
        print(json.dumps([{**row, 'active':store.active(row['id']) is not None} for row in rows], indent=2))
    return 0
