# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Hermes integration via supported middleware and hooks; no core patching."""
from copy import deepcopy
from contextvars import ContextVar
from functools import wraps
import inspect
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from .policy import Policy, TICKET_HEADER, CONTEXT_HEADER, VerifiedTurn, flatten_request
from .proxy import ProviderProxy
from .contacts import ApprovalStore
from .native_authority import NativeAuthority

logger = logging.getLogger(__name__)


class AuditLog:
    """One JSON line per model call. Callers pass fixed fields only: no prompt text, key or header value."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row):
        line = json.dumps(dict({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, **row), sort_keys=True) + "\n"
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)


class Runtime:
    def __init__(self, policy, proxy):
        self.policy, self.proxy = policy, proxy
        self.tool_context = ContextVar("owner_mode_tool_scope", default=())

    def begin(self, **context):
        terminal_context = getattr(self.policy.authority, 'terminal_context', None)
        if callable(terminal_context):
            context = terminal_context(context)
        resolver = getattr(self.policy.authority, 'rendered_user_message', None)
        if callable(resolver) and not context.get('parent_session_id'):
            try:
                context = {**context, 'user_message': resolver(**context)}
            except Exception:
                context = {**context, 'user_message': None}
        return self.policy.begin(**context)

    def execute_tool(self, args, next_call, *, tool_name="", **_):
        """Scope only: forward arguments, results, failures and host approvals unchanged."""
        path = self.tool_context.get() + (tool_name,)
        token = self.tool_context.set(path)
        try:
            result = next_call(args)
        finally:
            self.tool_context.reset(token)
        if inspect.isawaitable(result):
            async def wait():
                current = self.tool_context.set(path)
                try:
                    return await result
                finally:
                    self.tool_context.reset(current)
            return wait()
        return result

    def adopt(self, **context):
        context["tool_path"] = self.tool_context.get()
        return self.policy.adopt(**context)

    def request(self, request, *, session_id="", turn_id="", base_url="", platform="", **_):
        result = deepcopy(request)
        # Strip caller attempts even on unsupported routes. The final proxy is
        # the enforcement boundary when this optional middleware fails or skips.
        headers = result.get("extra_headers") or {}
        result["extra_headers"] = {k: v for k, v in headers.items()
                                   if not k.lower().startswith("x-owner-mode-")}
        result.pop("owner_mode", None)
        result.pop("owner_mode_scale", None)
        extra = dict(result.get("extra_body") or {})
        extra.pop("owner_mode_scale", None)
        # Absent means off, including on providers without owner-mode support.
        extra.pop("owner_mode", None)
        if extra or "extra_body" in result:
            result["extra_body"] = extra
        try:
            for route, config in self.proxy.routes.items():
                if base_url.rstrip("/") != self.proxy.base_url(route):
                    continue
                verified = self.policy.is_owner_work(session_id, turn_id)
                profile = config.get("profile", "off")
                ticket, reason = None, "route_profile_off"
                if profile and profile != "off":
                    if any(not self.policy.inherits(name) for name in self.tool_context.get()):
                        reason = "tool_owner_mode_disabled"
                    else:
                        ticket, reason = self.policy.decide(flatten_request(result), session_id=session_id,
                                                            turn_id=turn_id, profile=profile, route=route)
                if ticket:
                    result["extra_headers"][TICKET_HEADER] = ticket
                info = {"session": str(session_id), "platform": str(platform or ""), "owner_verified": verified,
                        "decision": reason,
                        "identity_reason": self.policy.identity_reason(
                            session_id, turn_id, verified=verified and reason != "owner_unverified"),
                        "identity_ms": self.policy.identity_ms(session_id, turn_id)}
                # The context ticket binds the final body, so it is minted after every edit above.
                result["extra_headers"][CONTEXT_HEADER] = self.policy.context_ticket(
                    flatten_request(result), session_id=session_id, turn_id=turn_id, route=route, info=info)
                break
        except Exception:
            result["extra_headers"].pop(TICKET_HEADER, None)
        return {"request": result, "source": "owner-mode"}

def load_authority(path, settings=None):
    """Load the operator's identity adapter file.

    A file defining ``build_authority(toolkit, proof_type=..., settings=...)`` gets the Hermes identity toolkit
    (``hermes_ingress``) and the ``identity_settings`` mapping. Otherwise ``Authority(proof_type=...)`` is used.
    """
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("owner_mode_identity_adapter", str(Path(path).expanduser()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build_authority", None)
    if callable(build):
        from . import hermes_ingress
        return build(hermes_ingress, proof_type=VerifiedTurn, settings=dict(settings) if isinstance(settings, dict) else {})
    return module.Authority(proof_type=VerifiedTurn)


def serving_process(expected, argv=None):
    """Only the configured long-lived process owns the loopback port and the policy.

    Other processes reading the same config (one-shot CLI runs, standalone cron
    runs) register nothing: their calls reach the serving process's wrapper as
    unticketed stock traffic, and they never bind the port themselves.
    """
    if not expected:
        return True
    argv = sys.argv if argv is None else argv
    return isinstance(expected, list) and list(argv[1:1 + len(expected)]) == [str(x) for x in expected]


_POLICY = None  # Compatibility mirror; authorization reads the process registry below.
_PROCESS_STATE_PREFIX = "_aevonix_flashalign_registration_v1_"


def _process_state(*, create=False):
    # Separate PID keys also avoid acquiring a lock inherited across fork. A
    # profile-local import or a module reload still finds this same process key.
    pid = os.getpid()
    name = _PROCESS_STATE_PREFIX + str(pid)
    state = sys.modules.get(name)
    if state is None and create:
        state = sys.modules.setdefault(name, SimpleNamespace(
            pid=pid, lock=threading.RLock(), owner=None))
    return state


def _registration_active(state, owner):
    if state is None or state.pid != os.getpid():
        return False
    with state.lock:
        return state.owner is owner and owner.active


def owner_turn(session_id="", turn_id="", tool_name=""):
    """Report only this process's active owner's personally verified turn."""
    try:
        state = _process_state()
        if state is None or state.pid != os.getpid():
            return False
        with state.lock:
            owner = state.owner
            if owner is None or not owner.active:
                return False
            return bool(owner.runtime.policy.owner_turn(session_id, turn_id, tool_name))
    except Exception:
        return False


def _guard_callback(state, owner, callback, *, kind="hook"):
    @wraps(callback)
    def guarded(*args, **kwargs):
        if _registration_active(state, owner):
            try:
                return callback(*args, **kwargs)
            finally:
                if not _registration_active(state, owner):
                    _clear_policy(owner)
        if kind == "prepare":
            raise RuntimeError("FlashAlign registration is inactive")
        if kind == "request":
            request = args[0] if args else kwargs["request"]
            # An old middleware must not erase a live owner's signed headers.
            return {"request": request, "source": "owner-mode"}
        if kind == "tool":
            value = args[0] if args else kwargs["args"]
            next_call = args[1] if len(args) > 1 else kwargs["next_call"]
            return next_call(value)
        return None
    return guarded


def _clear_policy(owner):
    if owner.runtime is None:
        return
    policy = owner.runtime.policy
    with policy.lock:
        policy.authority = None
        for table in (policy.turns, policy.tickets, policy.contexts,
                      policy.context_infos, policy.children):
            table.clear()


def _retire_registration(state, owner):
    global _POLICY
    if state.pid != os.getpid():
        return  # Never close resources inherited from another process.
    with state.lock:
        if owner.closed or owner.closing:
            return
        owner.active, owner.closing = False, True
        if owner.runtime is not None and _POLICY is owner.runtime.policy:
            _POLICY = None
        if owner.terminal is not None and getattr(owner.terminal, "runtime", None) is owner.runtime:
            owner.terminal.runtime = None
    _clear_policy(owner)
    failed = False
    for resource in (owner.proxy, owner.authority):
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                failed = True
    with state.lock:
        owner.closing = False
        if not failed:
            owner.closed = True
            if state.owner is owner:
                state.owner = None
        # Keep the inactive claim on cleanup failure. A later unload can retry;
        # a second live proxy must not be started while resource state is unknown.
    if failed:
        try:
            logger.warning("FlashAlign plugin cleanup incomplete")
        except Exception:
            pass


def register(ctx):
    global _POLICY
    state = _process_state(create=True)
    with state.lock:
        if state.owner is not None:
            return None
        from hermes_constants import get_hermes_home
        approvals = ApprovalStore(ctx.get_config("approvals_file", "") or
                                  Path(get_hermes_home()) / "owner-mode" / "approvals.json")
        from .terminal import current_binding, TerminalAuthority
        terminal = current_binding(approvals.path)
        if terminal is None and not serving_process(ctx.get_config("serve_argv", None)):
            return None
        owner = SimpleNamespace(active=False, closed=False, closing=False,
                                runtime=None, proxy=None, authority=None, terminal=terminal)
        state.owner = owner
        try:
            try:
                adapter = ctx.get_config("identity_adapter", "")
                authority = (TerminalAuthority(terminal, VerifiedTurn) if terminal is not None else
                             load_authority(adapter, ctx.get_config("identity_settings", {})) if adapter
                             else NativeAuthority(approvals.path, VerifiedTurn))
            except Exception as error:
                authority = None
                try:
                    logger.warning("FlashAlign identity authority unavailable (%s)", type(error).__name__)
                except Exception:
                    pass
            owner.authority = authority
            registry = getattr(authority, 'registry', approvals.registry)
            approvals = ApprovalStore(approvals.path, registry=registry)
            audit = None
            try:
                audit = AuditLog(ctx.get_config("audit_log", "") or Path(get_hermes_home()) / "logs" / "owner-mode.log")
            except Exception:
                audit = None
            channels = ctx.get_config("allowed_channels", None)
            if not isinstance(channels, list):
                channels = list(registry.ids())
            if terminal is not None and approvals.channel_enabled('local-terminal'):
                channels = [*channels, 'local-terminal']
            def channel_enabled(channel):
                value = approvals.read()
                if (adapter and 'enabled_channels' not in value and channel in channels
                        and registry.supports(channel, 'custom_authority')):
                    return value['enabled']
                return approvals.channel_enabled(channel)
            policy = Policy(trusted_system_hashes=ctx.get_config("trusted_system_sha256", []),
                            trusted_tools_hashes=ctx.get_config("trusted_tools_sha256", []), authority=authority,
                            trust=ctx.get_config("trust", None),
                            tool_policy=approvals.tool_enabled,
                            registry=registry,
                            channel_policy=channel_enabled,
                            **({"allowed_channels": channels} if isinstance(channels, list) else {}))
            owner.proxy = proxy = ProviderProxy(policy, ctx.get_config("routes", {}), ctx.get_config("key_file", ""),
                                  port=0 if terminal is not None else ctx.get_config("port", 0), audit=audit, approvals=approvals)
            owner.runtime = runtime = Runtime(policy, proxy)
            proxy.prepare = _guard_callback(state, owner, proxy.prepare, kind="prepare")
            for name in ("begin", "adopt"):
                setattr(runtime, name, _guard_callback(state, owner, getattr(runtime, name)))
            for name in ("end", "steered"):
                setattr(policy, name, _guard_callback(state, owner, getattr(policy, name)))
            runtime.request = _guard_callback(state, owner, runtime.request, kind="request")
            runtime.execute_tool = _guard_callback(state, owner, runtime.execute_tool, kind="tool")
            ctx.on_unload(lambda: _retire_registration(state, owner))
            ctx.register_hook("pre_llm_call", runtime.begin)
            ctx.register_hook("post_llm_call", policy.end)
            if authority and callable(getattr(authority, "hear", None)):
                ctx.register_hook("pre_gateway_dispatch", _guard_callback(state, owner, authority.hear))
            ctx.register_hook("subagent_start", runtime.adopt)
            ctx.register_hook("pre_tool_call", policy.steered)
            ctx.register_middleware("llm_request", runtime.request)
            ctx.register_middleware("tool_execution", runtime.execute_tool)
            proxy.start()
            if state.owner is not owner or owner.closed:
                raise RuntimeError("FlashAlign registration withdrawn during startup")
            if terminal is not None:
                terminal.runtime = runtime
            owner.active = True
            _POLICY = policy
            return runtime
        except BaseException:
            _retire_registration(state, owner)
            raise
