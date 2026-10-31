# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Native message receipts, rendered-content binding and persistent approvals."""
from collections import OrderedDict
from contextvars import ContextVar
import threading

from .contacts import ApprovalStore
from .channels import DEFAULT_REGISTRY
from .content import bind_content, proof_matches
from .hermes_observer import HermesObserver, event_copy
from .identity import DM_TYPES, canonical_handle
from . import hermes_ingress


class NativeAuthority:
    def __init__(self, path, proof_type, *, registry=DEFAULT_REGISTRY):
        self.registry = registry
        self.store = ApprovalStore(path, registry=registry)
        self.proof_type = proof_type
        self.lock = threading.RLock()
        self.pending = OrderedDict()
        self.consumed = OrderedDict()
        self.groups = OrderedDict()
        self.rendered = OrderedDict()
        self.render_scope = ContextVar('flashalign_native_render_scope', default=(False, None))
        self.observer = HermesObserver(self)

    def _key(self, facts):
        account = getattr(facts, 'account_id', None) or getattr(facts, 'extra', {}).get('account_id') or 'default'
        return (facts.channel, account, canonical_handle(facts.channel, facts.sender_id, registry=self.registry),
                facts.chat_id, facts.message_id)

    @staticmethod
    def _trim(table, limit=4096):
        while len(table) > limit:
            table.popitem(last=False)

    def hear(self, *, event=None, gateway=None, **_):
        root_key = None
        try:
            self.observer.attach(gateway)
            self.observer.remember_ingress(gateway, event)
            current = hermes_ingress.ingress_facts(event, gateway)
            root_key = self._key(current)
            parts = self.observer.parts(event) if event is not None else (None,)
            facts = [hermes_ingress.ingress_facts(part, gateway) if part is not None else current for part in parts]
            keys = [self._key(item) for item in facts]
            if (not keys or len(keys) != len(set(keys)) or root_key not in keys
                    or any(key[:4] != root_key[:4] for key in keys)):
                raise ValueError('Invalid native composition')
            rows = []
            for item, key in zip(facts, keys):
                if (not self.registry.supports(item.channel, 'native_message')
                        or not item.native or item.internal or not item.allowlisted or item.own_account
                        or item.reaction or item.chat_type not in DM_TYPES
                        or not all(key) or not (item.text or item.media)):
                    raise ValueError('Unverified native message')
                grant = self.store.approved(key[0], key[1], key[2])
                if not grant:
                    raise ValueError('Contact is not approved')
                rows.append((item, grant))
            with self.lock:
                if any(key in self.consumed for key in keys):
                    return
                for key, row in zip(keys, rows):
                    prior = self.pending.get(key)
                    if prior and (prior[0].text, prior[0].media, prior[1]) != (row[0].text, row[0].media, row[1]):
                        raise ValueError('Native message identity was reused')
                for key, row in zip(keys, rows):
                    self.pending.setdefault(key, row)
                self.groups[root_key] = tuple(keys)
                self._trim(self.pending); self._trim(self.groups)
        except Exception:
            if root_key:
                with self.lock:
                    self.rendered[root_key] = None
                    self._trim(self.rendered)

    def record_render(self, *, event, gateway, content, members=None):
        """Called by a trusted host rendering seam; never by request-body metadata."""
        if members is not None:
            self.observer.remember(event, tuple(event_copy(member) for member in members))
        current = hermes_ingress.ingress_facts(event, gateway)
        key = self._key(current)
        with self.lock:
            if (not self.registry.supports(current.channel, 'native_message')
                    or not current.native or current.internal or not current.allowlisted or current.own_account
                    or current.reaction or current.chat_type not in DM_TYPES):
                self.rendered[key] = None
                self._trim(self.rendered)
                return
            grouped = self.observer.groups.get(id(event))
            if key not in self.groups or grouped and grouped[0]() is event:
                self.hear(event=event, gateway=gateway)
            members = self.groups.get(key, ())
            if not members or any(member not in self.pending for member in members):
                self.rendered[key] = None
            elif key not in self.rendered or self.rendered[key] is not None:
                self.rendered[key] = bind_content(content)
            self._trim(self.rendered)

    def reject_render(self, event):
        try:
            key = self._key(hermes_ingress.ingress_facts(event, None))
            with self.lock:
                self.rendered[key] = None
                self._trim(self.rendered)
        except Exception:
            pass

    def verify_turn(self, *, session_id, turn_id, **kwargs):
        try:
            facts = hermes_ingress.turn_facts(**kwargs)
            if (not self.registry.supports(facts.channel, 'native_message')
                    or not facts.route_ok or facts.synthetic or facts.chat_type not in DM_TYPES):
                return None
            key = self._key(facts)
            with self.lock:
                members = self.groups.pop(key, (key,))
                rows = [self.pending.get(member) for member in members]
                bound = self.rendered.pop(key, False)
                if not rows or any(row is None for row in rows):
                    return None
                for member in members:
                    self.pending.pop(member, None)
                    self.consumed[member] = True
                self._trim(self.consumed, 8192)
            if bound is None:
                return None
            if bound is False:
                # Unmodified text works on hosts without a rendering observer.
                if len(rows) != 1 or rows[0][0].media:
                    return None
                bound = bind_content(rows[0][0].text)
            if not proof_matches(facts.text, bound, allow_text_context=True):
                return None
            grant = rows[0][1]
            for member, row in zip(members, rows):
                if row[1] != grant or self.store.approved(member[0], member[1], member[2]) != row[1]:
                    return None
            return self.proof_type(session_id, turn_id, facts.channel, bound.own_text, grant_id=grant,
                                   content_sha256=bound.content_sha256, content_parts=bound.content_parts)
        except Exception:
            return None

    def rendered_user_message(self, **context):
        """Normalize a native text-only hook to its current, route-bound user row."""
        history = context.get('conversation_history')
        row = history[-1] if isinstance(history, list) and history and isinstance(history[-1], dict) else {}
        if not isinstance(context.get('user_message'), str) or not isinstance(row.get('content'), list):
            return context.get('user_message')
        active, marker = self.render_scope.get()
        self.render_scope.set((active, None))
        if not active or marker is None:
            return None
        key, binding = marker
        facts = hermes_ingress.turn_facts(native_message_id=key[-1], **context)
        if (not facts.route_ok or facts.synthetic or self._key(facts) != key
                or not proof_matches(facts.text, binding, allow_text_context=True)):
            return None
        return facts.text

    def record_native_render(self, *, event, gateway, content):
        self.record_render(event=event, gateway=gateway, content=content)
        active, _ = self.render_scope.get()
        if active:
            key = self._key(hermes_ingress.ingress_facts(event, gateway))
            with self.lock:
                binding = self.rendered.get(key)
            self.render_scope.set((True, (key, binding) if binding else None))

    def is_active(self, proof):
        try:
            return self.store.active(proof.grant_id)
        except Exception:
            return False

    def close(self):
        self.observer.close()
