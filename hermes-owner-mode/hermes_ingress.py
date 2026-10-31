# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Hermes binding for ``identity``: the channel adapter's ingress facts and the turn's bound route.

``pre_gateway_dispatch`` gives every inbound ``MessageEvent`` (before Hermes's own access check) and the gateway;
``pre_llm_call`` gives the turn: platform, sender id, the user message and the conversation including the current
user row, which carries the platform message id. The gateway binds the turn's route (platform, user, chat, chat
type, cron flag) in ``gateway.session_context``. Nothing else is needed.

Internal Hermes attributes used by this adapter: ``SessionSource._transport_adapter_ref``,
``GatewayRunner.adapters``, ``GatewayRunner._is_user_authorized_for_source``,
``gateway.session_context.get_session_env``, ``gateway.session_context.session_context_engaged``.
"""
import logging
import time
from types import MappingProxyType

from .identity import (DM_TYPES, OWNER, OWNER_CHANNELS, IngressFacts, IngressLedger, TurnFacts, all_of,
                       canonical_handle, handle_fits, owner_handle_source)
from .channels import DEFAULT_REGISTRY

__all__ = ["HermesAuthority", "ingress_facts", "turn_facts", "all_of", "owner_handle_source", "canonical_handle",
           "handle_fits", "IngressFacts", "TurnFacts", "IngressLedger", "OWNER", "OWNER_CHANNELS", "DM_TYPES"]

logger = logging.getLogger("owner_mode.identity")
# WhatsApp delivers reactions, polls, contacts and locations as bridge-formatted TEXT bodies; only these two
# native types are the sender's own typed words.
WHATSAPP_PLAIN_TYPES = frozenset({"conversation", "extendedTextMessage"})


def _value(value):
    return getattr(value, "value", value)


def _text(value):
    return "" if value is None else str(value).strip()


def ingress_facts(event, gateway):
    """Facts from a Hermes ``MessageEvent`` exactly as the channel adapter built it."""
    source = event.source
    channel = _text(_value(source.platform)).lower()
    metadata = event.metadata if isinstance(event.metadata, dict) else {}
    ref = getattr(source, "_transport_adapter_ref", None)
    adapter = ref() if callable(ref) else None
    adapters = getattr(gateway, "adapters", None)
    registered = adapters.values() if isinstance(adapters, dict) else ()
    native = (adapter is not None and any(adapter is item for item in registered)
              and _text(_value(getattr(adapter, "platform", ""))).lower() == channel)
    internal = (event.internal is not False or getattr(source, "delivered_via_upstream_relay", False) is not False
                or metadata.get("hermes_plugin_injection") is not None)
    kind = _text(_value(event.message_type)).lower()
    plain = (kind == "text" and not event.media_urls
             and (channel != "whatsapp" or metadata.get("whatsapp_native_type") in WHATSAPP_PLAIN_TYPES))
    flag = metadata.get("forwarded")
    try:
        allowlisted = gateway._is_user_authorized_for_source(source) is True
    except Exception:
        allowlisted = False
    return IngressFacts(
        channel=channel, native=native, internal=internal,
        chat_type=_text(source.chat_type).lower(), chat_id=_text(source.chat_id), sender_id=_text(source.user_id),
        message_id=_text(event.message_id) or _text(getattr(source, "message_id", None)),
        text=event.text if isinstance(event.text, str) else "",
        plain_text=plain,
        reply_quoted=bool(event.reply_to_message_id or getattr(event, "reply_to_text", None)),
        forwarded=True if flag is True else False if flag is False else None,
        own_account=metadata.get("whatsapp_from_owner") is True or getattr(source, "is_bot", False) is True,
        allowlisted=allowlisted,
        extra=MappingProxyType({"adapter": adapter, "metadata": metadata,
                                "account_id": _text(getattr(source, "account_id", "")) or "default"}),
        message_kind=kind,
        media=tuple(str(path) for path in (getattr(event, 'media_urls', None) or [])),
        reaction=(kind == 'reaction' or metadata.get('whatsapp_native_type') == 'reactionMessage'
                  or metadata.get('reaction') is True),
    )


def _route():
    """The route the gateway bound for this turn, or None when no gateway session is bound in this process."""
    from gateway.session_context import get_session_env, session_context_engaged
    if session_context_engaged() is not True:
        return None
    return {name: _text(get_session_env("HERMES_" + name, "")) for name in (
        "SESSION_PLATFORM", "SESSION_USER_ID", "SESSION_CHAT_ID", "SESSION_CHAT_TYPE", "CRON_SESSION",
        "SESSION_ACCOUNT_ID", "SESSION_MESSAGE_ID")}


def turn_facts(*, platform="", sender_id="", user_message=None, conversation_history=None,
               native_message_id="", **_):
    """Facts for the turn from ``pre_llm_call`` keyword arguments plus the gateway's bound route."""
    channel = _text(_value(platform)).lower()
    sender = _text(sender_id)
    route = _route() or {}
    history = conversation_history if isinstance(conversation_history, list) else []
    row = history[-1] if history and isinstance(history[-1], dict) else {}
    # Hermes may project the hook's original_user_message to text while the
    # current native user row retains the host-rendered image/audio payload.
    # The authority still compares this full value to its exact render digest.
    rendered = row.get('content')
    projected_media = isinstance(user_message, str) and isinstance(rendered, list)
    row_id = _text(row.get('platform_message_id'))
    route_id = _text(native_message_id)
    current_media = (projected_media and row.get('role') == 'user' and bool(row_id) and row_id == route_id
                     and sum(isinstance(item, dict) and item.get('role') == 'user'
                             and _text(item.get('platform_message_id')) == row_id for item in history) == 1)
    content = rendered if projected_media and current_media else user_message
    route_ok = (bool(route) and route["SESSION_PLATFORM"].lower() == channel and route["SESSION_USER_ID"] == sender
                and not route["CRON_SESSION"])
    return TurnFacts(
        channel=channel, sender_id=sender,
        chat_id=route.get("SESSION_CHAT_ID", ""), chat_type=route.get("SESSION_CHAT_TYPE", "").lower(),
        message_id=_text(row.get("platform_message_id")) or route.get('SESSION_MESSAGE_ID', ''),
        text=content, route_ok=route_ok,
        synthetic=(row.get("role") != "user" or bool(row.get("display_kind"))
                   or projected_media and not current_media),
        account_id=route.get("SESSION_ACCOUNT_ID") or "default",
    )


def _reject(label, error=None):
    """One warning line with a fixed label (and an error class name). Never message text, ids or handles."""
    try:
        if error is None:
            logger.warning("owner-mode identity: owner verify rejected: %s", label)
        else:
            logger.warning("owner-mode identity: owner verify rejected: %s (%s)", label, type(error).__name__)
    except Exception:
        pass
    return None


class HermesAuthority:
    """The object ``plugin.load_authority`` installs: ``hear`` on ``pre_gateway_dispatch``, ``verify_turn`` from
    ``Policy.begin`` on ``pre_llm_call``."""

    def __init__(self, source, *, proof_type, channels=OWNER_CHANNELS, ttl=300.0, recheck=None,
                 clock=time.monotonic, registry=DEFAULT_REGISTRY):
        self.proof_type = proof_type
        self.registry = registry
        self.ledger = IngressLedger(source, channels=channels, ttl=ttl,
                                    recheck=recheck if recheck is not None else getattr(source, "recheck", None),
                                    clock=clock, registry=registry)

    def hear(self, *, event=None, gateway=None, **_):
        try:
            self.ledger.hear(ingress_facts(event, gateway))
        except Exception as error:
            try:
                logger.warning("owner-mode identity: hear() failed (%s); this message cannot be owner",
                               type(error).__name__)
            except Exception:
                pass
        return None  # observe only: never skip, rewrite or allow a dispatch

    def verify_turn(self, *, session_id, turn_id, user_message, **context):
        try:
            facts = turn_facts(user_message=user_message, **context)
            owner, label = self.ledger.verify(facts)
            if owner is not True:
                return _reject(label)
            return self.proof_type(session_id, turn_id, facts.channel, user_message)
        except Exception as error:
            return _reject("exception", error)
