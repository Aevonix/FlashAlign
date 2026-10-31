# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Explicit local terminal sessions and per-submission Hermes receipts.

The session id is a revocation handle, not an authentication token. Only a
trusted in-process terminal adapter may turn a live session into a turn proof.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from .contacts import ApprovalStore
from .content import bind_content, proof_matches


CHANNEL = 'local-terminal'
_CONTEXT_NAME = '_flashalign_authorized_terminal'


class TerminalSessionStore:
    """Private, expiring local approvals; reread for every authorization."""
    def __init__(self, approvals, *, clock=time.time):
        self.approvals = approvals if isinstance(approvals, ApprovalStore) else ApprovalStore(approvals)
        self.path = self.approvals.path.with_name('terminal-sessions.json')
        self.clock = clock

    def read(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return {'version': 1, 'sessions': []}
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                    or info.st_size > 1024 * 1024):
                raise ValueError('Terminal sessions must be an owned private regular file')
            with os.fdopen(fd) as stream:
                fd = None
                value = json.load(stream)
            if (not isinstance(value, dict) or set(value) != {'version', 'sessions'}
                    or value['version'] != 1 or not isinstance(value['sessions'], list)):
                raise ValueError('Invalid terminal session store')
            seen = set()
            for row in value['sessions']:
                if (not isinstance(row, dict) or set(row) != {'id', 'harness', 'account', 'contact', 'grant_id', 'expires', 'enabled'}
                        or not isinstance(row['id'], str) or len(row['id']) != 32 or row['id'] in seen
                        or row['harness'] not in ('hermes', 'openclaw')
                        or any(not isinstance(row[k], str) or not row[k] for k in ('account', 'contact', 'grant_id'))
                        or type(row['expires']) not in (int, float) or not 0 < row['expires'] < 1e12
                        or type(row['enabled']) is not bool):
                    raise ValueError('Invalid terminal session record')
                seen.add(row['id'])
            return value
        finally:
            if fd is not None:
                os.close(fd)

    def _update(self, change):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(str(self.path)+'.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        temporary = None
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                raise ValueError('Invalid terminal session lock')
            fcntl.flock(fd, fcntl.LOCK_EX)
            value = self.read()
            change(value)
            out, temporary = tempfile.mkstemp(prefix='.terminal-sessions-', dir=self.path.parent)
            with os.fdopen(out, 'w') as stream:
                json.dump(value, stream, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path); temporary = None
            return value
        finally:
            if temporary is not None:
                os.unlink(temporary)
            os.close(fd)

    def create(self, harness, *, ttl=3600):
        if harness not in ('hermes', 'openclaw') or type(ttl) is not int or not 60 <= ttl <= 86400:
            raise ValueError('Choose Hermes or OpenClaw and a session duration from 60 to 86400 seconds')
        account, contact = harness, 'uid:'+str(os.getuid())
        grant = self.approvals.approved(CHANNEL, account, contact)
        if not grant:
            raise ValueError('Approve this local terminal account before starting a session')
        row = {'id': secrets.token_hex(16), 'harness': harness, 'account': account,
               'contact': contact, 'grant_id': grant, 'expires': self.clock()+ttl, 'enabled': True}
        def add(value):
            value['sessions'] = [r for r in value['sessions'] if r['enabled'] and r['expires'] > self.clock()][-255:]
            value['sessions'].append(row)
        self._update(add)
        return row

    def active(self, session, *, harness=None):
        try:
            row = next((r for r in self.read()['sessions'] if r['id'] == session), None)
            if (not row or not row['enabled'] or row['expires'] <= self.clock()
                    or harness is not None and row['harness'] != harness
                    or row['contact'] != 'uid:'+str(os.getuid())
                    or self.approvals.approved(CHANNEL, row['account'], row['contact']) != row['grant_id']):
                return None
            return dict(row)
        except (OSError, ValueError, TypeError):
            return None

    def revoke(self, session):
        def change(value):
            found = next((r for r in value['sessions'] if r['id'] == session), None)
            if found is None:
                raise ValueError('Unknown terminal session')
            found['enabled'] = False
        return self._update(change)


class TerminalBinding:
    """Receipts minted at the native terminal input seam, never from labels."""
    def __init__(self, sessions, session):
        self.sessions, self.session = sessions, session
        self.pid = os.getpid()
        self.current = ContextVar('flashalign_terminal_submission', default=None)
        self.proofs = OrderedDict()
        self.lock = threading.RLock()

    @contextmanager
    def submission(self, session_id, content):
        row = self.sessions.active(self.session, harness='hermes')
        bound = bind_content(content) if isinstance(content, str) and content.strip() else None
        receipt = (session_id, bound, secrets.token_hex(16)) if row and bound else None
        token = self.current.set(receipt)
        try:
            yield
        finally:
            self.current.reset(token)

    def verify(self, proof_type, *, session_id, turn_id, user_message, **context):
        current = self.current.get()
        row = self.sessions.active(self.session, harness='hermes')
        if (os.getpid() != self.pid or not row or not current or current[0] != session_id
                or not turn_id or context.get('parent_session_id')
                or not proof_matches(user_message, current[1], allow_text_context=True)):
            return None
        key = (session_id, turn_id)
        with self.lock:
            if any(value == current[2] and known != key for known, value in self.proofs.items()):
                return None
            self.proofs[key] = current[2]
            while len(self.proofs) > 4096:
                self.proofs.popitem(last=False)
        bound = current[1]
        return proof_type(session_id, turn_id, CHANNEL, bound.own_text, grant_id=row['grant_id'],
                          content_sha256=bound.content_sha256, content_parts=bound.content_parts)


class TerminalAuthority:
    def __init__(self, binding, proof_type):
        self.binding, self.proof_type = binding, proof_type
        self.registry = binding.sessions.approvals.registry

    def verify_turn(self, **context):
        return self.binding.verify(self.proof_type, **context)

    def terminal_context(self, context):
        current = self.binding.current.get()
        if (current and current[0] == context.get('session_id')
                and context.get('platform') == 'cli' and not context.get('parent_session_id')):
            return {**context, 'platform': CHANNEL}
        return context

    def is_active(self, proof):
        row = self.binding.sessions.active(self.binding.session, harness='hermes')
        return bool(row and proof.channel == CHANNEL and proof.grant_id == row['grant_id']
                    and (proof.session_id, proof.turn_id) in self.binding.proofs)


def current_binding(approvals=None):
    state = sys.modules.get(_CONTEXT_NAME)
    binding = getattr(state, 'binding', None)
    if binding is None or binding.pid != os.getpid():
        return None
    if approvals is not None and binding.sessions.approvals.path.resolve() != Path(approvals).resolve():
        return None
    return binding


@contextmanager
def terminal_launch(sessions, session):
    """Publish launch authority before the native host discovers its plugins."""
    if current_binding() is not None:
        raise ValueError('A terminal adapter is already active')
    binding = TerminalBinding(sessions, session)
    state = SimpleNamespace(binding=binding)
    sys.modules[_CONTEXT_NAME] = state
    try:
        yield binding
    finally:
        if sys.modules.get(_CONTEXT_NAME) is state:
            del sys.modules[_CONTEXT_NAME]


@contextmanager
def native_hermes_session(sessions, session, cli_class, agent_class):
    """Observe actual interactive CLI submissions and their native agent call."""
    existing = current_binding()
    if existing is not None and (existing.session != session or existing.sessions.path != sessions.path):
        raise ValueError('A different terminal adapter is already active')
    context = terminal_launch(sessions, session) if existing is None else None
    binding = context.__enter__() if context is not None else existing
    old_chat, old_run, old_init = cli_class.chat, agent_class.run_conversation, agent_class.__init__
    pending = {'cli': None, 'message': None}
    def chat(cli, message, *args, **kwargs):
        runtime = getattr(binding, 'runtime', None)
        if runtime is None:
            raise RuntimeError('The configured FlashAlign Hermes plugin did not start the terminal adapter')
        cli.base_url = runtime.proxy.base_url('head')
        agent = getattr(cli, 'agent', None)
        if agent is not None and getattr(agent, 'base_url', '') != cli.base_url:
            raise RuntimeError('Hermes initialized a model outside the terminal adapter')
        if pending['cli'] is not None:
            return old_chat(cli, message, *args, **kwargs)
        pending.update(cli=cli, message=message if not args and not kwargs.get('images') and not kwargs.get('voice_input') else None)
        try:
            return old_chat(cli, message, *args, **kwargs)
        finally:
            pending.update(cli=None, message=None)
    def run(agent, *args, **kwargs):
        cli = pending['cli']
        if cli is None or getattr(cli, 'agent', None) is not agent or pending['message'] is None:
            return old_run(agent, *args, **kwargs)
        with binding.submission(getattr(agent, 'session_id', ''), pending['message']):
            return old_run(agent, *args, **kwargs)
    def initialize(agent, *args, **kwargs):
        runtime = getattr(binding, 'runtime', None)
        if runtime is not None:
            if args:
                args = (runtime.proxy.base_url('head'), *args[1:])
            else:
                kwargs['base_url'] = runtime.proxy.base_url('head')
        return old_init(agent, *args, **kwargs)
    cli_class.chat, agent_class.run_conversation, agent_class.__init__ = chat, run, initialize
    try:
        yield binding
    finally:
        cli_class.chat, agent_class.run_conversation, agent_class.__init__ = old_chat, old_run, old_init
        if context is not None:
            context.__exit__(None, None, None)
