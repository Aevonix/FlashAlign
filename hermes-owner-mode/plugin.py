# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Hermes integration via supported middleware and hooks; no core patching."""
from copy import deepcopy
from contextvars import ContextVar
import inspect
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

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


_POLICY = None  # the serving process's Policy, published by register() once the proxy is up


def owner_turn(session_id="", turn_id="", tool_name=""):
    """The one interface other plugins read (see the public owner_turn contract): True only while this exact turn is one
    the owner verified himself; False in a process that does not serve owner mode. Never raises."""
    policy = _POLICY
    return bool(policy is not None and policy.owner_turn(session_id, turn_id, tool_name))


def _unpublish():
    global _POLICY
    _POLICY = None


def register(ctx):
    global _POLICY
    from hermes_constants import get_hermes_home
    approvals = ApprovalStore(ctx.get_config("approvals_file", "") or
                              Path(get_hermes_home()) / "owner-mode" / "approvals.json")
    from .terminal import current_binding, TerminalAuthority
    terminal = current_binding(approvals.path)
    if terminal is None and not serving_process(ctx.get_config("serve_argv", None)):
        return None
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
            pass  # Diagnostics must not change the unavailable-authority behavior.
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
    proxy = ProviderProxy(policy, ctx.get_config("routes", {}), ctx.get_config("key_file", ""),
                          port=0 if terminal is not None else ctx.get_config("port", 0), audit=audit, approvals=approvals)
    runtime = Runtime(policy, proxy)
    if callable(getattr(authority, 'close', None)):
        ctx.on_unload(authority.close)
    ctx.register_hook("pre_llm_call", runtime.begin)
    ctx.register_hook("post_llm_call", policy.end)
    if authority and callable(getattr(authority, "hear", None)):
        ctx.register_hook("pre_gateway_dispatch", authority.hear)
    ctx.register_hook("subagent_start", runtime.adopt)  # captures the creating tool's scope
    ctx.register_hook("pre_tool_call", policy.steered)  # delegate_task steer: only owner work steers owner work
    ctx.register_middleware("llm_request", runtime.request)
    ctx.register_middleware("tool_execution", runtime.execute_tool)
    ctx.on_unload(proxy.close)
    proxy.start()
    if terminal is not None:
        terminal.runtime = runtime
    _POLICY = policy
    ctx.on_unload(_unpublish)  # an unloaded plugin answers no owner_turn
    return runtime
