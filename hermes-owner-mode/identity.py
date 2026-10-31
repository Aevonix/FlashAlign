# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Transport identity primitives and an optional strict handle-source utility.

An identity source is any callable ``source(facts) -> (bool, label)``. This module's strict source accepts a direct,
plain-text message that a native channel adapter delivered on an owner channel, from a sender id the adapter
recorded that equals a configured owner handle. ``IngressLedger`` binds one verified message to at most one
later turn: same channel, sender, chat and message id, the exact text (SHA-256), a short freshness window,
one-time use.

Fail closed throughout: anything missing, unreadable or mismatched is "not owner". Labels are fixed strings and
never carry message text, sender ids, chat ids or handles. The deployment supplies the owner handles through a
loader; no handle is ever written in this package. The default installed adapter
uses NativeAuthority with persistent approvals and native rendered-message bindings.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional

from .channels import DEFAULT_ENABLED, DEFAULT_REGISTRY

# Compatibility export. Eligibility is resolved through the installed registry.
OWNER_CHANNELS = frozenset(DEFAULT_ENABLED)
DM_TYPES = frozenset({"dm", "direct", "private"})
OWNER = "owner"
_LABEL = re.compile(r"[a-z][a-z0-9_]{0,47}")
_EMPTY = MappingProxyType({})


@dataclass(frozen=True)
class IngressFacts:
    """What a channel adapter recorded for one inbound message, as the harness observed it at ingress.

    ``forwarded`` is None when the channel does not report forwarding. ``extra`` holds adapter-specific
    attestations that only deployment sources read.
    """
    channel: str = ""
    native: bool = False          # delivered by the harness's own registered adapter for ``channel``
    internal: bool = False        # synthetic, injected, relayed or replayed by the harness
    chat_type: str = ""
    chat_id: str = ""
    sender_id: str = ""           # the sender id the adapter recorded (never a caller-supplied value)
    message_id: str = ""
    text: str = ""
    plain_text: bool = False      # plain text: no media, reaction, poll, contact or location
    reply_quoted: bool = False
    forwarded: Optional[bool] = None
    own_account: bool = False     # an echo of the assistant's own account
    allowlisted: bool = False     # the harness's own access list admits this sender
    extra: Mapping[str, Any] = field(default=_EMPTY, compare=False, repr=False)
    message_kind: str = "text"
    media: tuple[str, ...] = ()
    reaction: bool = False


@dataclass(frozen=True)
class TurnFacts:
    """The turn about to call the model, as the harness bound it.

    ``route_ok``: the harness's bound route for this turn names the same channel and sender and is not a
    scheduled job. ``synthetic``: the current user row was produced by the harness, not delivered by a channel.
    """
    channel: str = ""
    sender_id: str = ""
    chat_id: str = ""
    chat_type: str = ""
    message_id: str = ""
    text: Any = None
    route_ok: bool = False
    synthetic: bool = True
    account_id: str = "default"


def _clean(value):
    return value.strip() if isinstance(value, str) else ""


def canonical_handle(channel, value, *, registry=DEFAULT_REGISTRY):
    """Canonical identity from an installed adapter, or untrusted text for rejection."""
    try:
        return registry.canonical_contact(channel, value)
    except ValueError:
        return _clean(value)


def handle_fits(channel, canonical, *, registry=DEFAULT_REGISTRY):
    """Whether the installed adapter accepts this exact canonical identity."""
    try:
        return registry.canonical_contact(channel, canonical) == canonical
    except (TypeError, ValueError):
        return False


def _decision(result):
    """Normalise a source's answer. Anything but (True, ...) is a refusal; a malformed label is replaced."""
    if not isinstance(result, tuple) or len(result) != 2:
        return False, "source_refused"
    owner, label = result
    if owner is True:
        return True, OWNER
    return False, label if isinstance(label, str) and _LABEL.fullmatch(label) and label != OWNER else "source_refused"


class OwnerHandleSource:
    """The default identity source. ``load_handles(channel)`` returns the configured handles; it is called on
    every check so a removed handle stops counting at once, and any error means nobody is the owner."""

    def __init__(self, load_handles, *, channels=OWNER_CHANNELS, require_forward_flag=False,
                 registry=DEFAULT_REGISTRY):
        self.load_handles = load_handles
        self.registry = registry
        self.channels = frozenset(c for c in channels if registry.supports(c, "native_message"))
        self.require_forward_flag = require_forward_flag is True

    def recheck(self, channel, sender_id):
        """(True, "owner") while ``sender_id`` is still a configured handle for ``channel``."""
        try:
            handles = self.load_handles(channel)
            if isinstance(handles, (str, bytes)) or not hasattr(handles, "__iter__"):
                return False, "owner_handles_unavailable"
            configured = {canonical_handle(channel, item, registry=self.registry) for item in handles if isinstance(item, str)}
            configured = {item for item in configured if handle_fits(channel, item, registry=self.registry)}
        except Exception:
            return False, "owner_handles_unavailable"
        if not configured:
            return False, "owner_handles_unavailable"
        sender = canonical_handle(channel, sender_id, registry=self.registry)
        if not handle_fits(channel, sender, registry=self.registry) or sender not in configured:
            return False, "sender_mismatch"
        return True, OWNER

    def __call__(self, facts):
        channel = _clean(facts.channel).lower()
        if not channel:
            return False, "no_channel"
        if channel not in self.channels:
            return False, "non_owner_channel"
        if facts.internal is not False:
            return False, "internal_event"
        if facts.native is not True:
            return False, "not_native_adapter"
        if _clean(facts.chat_type).lower() not in DM_TYPES:
            return False, "group_chat"
        if facts.own_account is not False:
            return False, "own_account_echo"
        if facts.forwarded is True or (self.require_forward_flag and facts.forwarded is not False):
            return False, "forwarded"
        if facts.reply_quoted is not False:
            return False, "reply_quoted"
        if facts.plain_text is not True:
            return False, "not_plain_text"
        if not (_clean(facts.sender_id) and _clean(facts.chat_id) and _clean(facts.message_id)
                and isinstance(facts.text, str) and facts.text):
            return False, "incomplete_event"
        matched = self.recheck(channel, facts.sender_id)
        if matched[0] is not True:
            return matched
        if facts.allowlisted is not True:
            return False, "not_authorized"
        return True, OWNER


def owner_handle_source(load_handles, *, channels=OWNER_CHANNELS, require_forward_flag=False,
                        registry=DEFAULT_REGISTRY):
    return OwnerHandleSource(load_handles, channels=channels, require_forward_flag=require_forward_flag,
                             registry=registry)


def all_of(*sources):
    """AND: every source must say owner. The first refusal's label wins; a raising source refuses."""
    def combined(facts):
        for source in sources:
            try:
                owner, label = _decision(source(facts))
            except Exception:
                return False, "exception"
            if not owner:
                return False, label
        return (True, OWNER) if sources else (False, "no_source")
    combined.recheck = getattr(sources[0], "recheck", None) if sources else None
    return combined


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IngressLedger:
    """Verified messages waiting for their turn, plus refusals and consumed keys (all in memory, bounded).

    Rows hold a monotonic time and the SHA-256 of the text, never the text. A key already heard or consumed is
    never overwritten, so a duplicate delivery cannot replace the original content.
    """

    def __init__(self, source, *, channels=OWNER_CHANNELS, ttl=300.0, recheck=None, clock=time.monotonic,
                 max_rows=4096, registry=DEFAULT_REGISTRY):
        self.source = source
        self.registry = registry
        self.channels = frozenset(c for c in channels if registry.supports(c, "native_message"))
        self.ttl = float(ttl)
        if not 0 < self.ttl <= 3600:
            raise ValueError("ttl must be in (0, 3600] seconds")
        self.recheck = recheck
        self.clock = clock
        self.max_rows = int(max_rows)
        self.lock = threading.Lock()
        self.rows = OrderedDict()      # key -> (heard_at, text_sha256)
        self.refused = OrderedDict()   # key -> label
        self.consumed = OrderedDict()  # key -> consumed_at

    def _key(self, channel, sender_id, chat_id, message_id):
        parts = (_clean(channel).lower(), _clean(sender_id), _clean(chat_id), _clean(message_id))
        return parts if all(parts) else None

    def _trim(self, table, limit):
        while len(table) > limit:
            table.popitem(last=False)

    def hear(self, facts):
        """Record one inbound message. Returns the label (audit/test only; never logged here)."""
        try:
            owner, label = _decision(self.source(facts))
        except Exception:
            owner, label = False, "exception"
        key = self._key(facts.channel, facts.sender_id, facts.chat_id, facts.message_id)
        if key is None or key[0] not in self.channels:
            return label
        with self.lock:
            if key in self.rows or key in self.consumed:
                return "duplicate"
            if owner:
                self.refused.pop(key, None)
                self.rows[key] = (self.clock(), _sha256(facts.text))
                self._trim(self.rows, self.max_rows)
            elif key not in self.refused:
                self.refused[key] = label
                self._trim(self.refused, self.max_rows)
        return label

    def verify(self, turn):
        """(True, "owner") only for the one turn bound to a verified message; else (False, label)."""
        channel = _clean(turn.channel).lower()
        if not channel:
            return False, "no_channel"
        if channel not in self.channels:
            return False, "non_owner_channel"
        if not _clean(turn.sender_id):
            return False, "no_sender"
        if turn.synthetic is not False or not _clean(turn.message_id):
            return False, "turn_unbound"
        key = self._key(channel, turn.sender_id, turn.chat_id, turn.message_id)
        if key is None:
            return False, "turn_route_mismatch"
        with self.lock:
            if key in self.consumed:
                return False, "replay"
            if key in self.refused:
                return False, self.refused[key]
            if turn.route_ok is not True or _clean(turn.chat_type).lower() not in DM_TYPES:
                return False, "turn_route_mismatch"
            row = self.rows.pop(key, None)
            if row is None:
                return False, "no_ingress_record"
            # Burn the key before comparing anything, so a failed attempt cannot be retried.
            self.consumed[key] = self.clock()
            self._trim(self.consumed, 2 * self.max_rows)
        if self.clock() - row[0] > self.ttl:
            return False, "stale"
        if not isinstance(turn.text, str) or _sha256(turn.text) != row[1]:
            return False, "text_mismatch"
        if self.recheck is not None:
            owner, label = _decision(self.recheck(channel, turn.sender_id))
            if not owner:
                return False, label
        return True, OWNER
