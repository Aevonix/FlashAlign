# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Authenticated loopback control API for a native non-Python harness adapter.

The adapter attests native ingress in code. Its private control key never goes
to the model. Opaque run capabilities stay in transport headers and expire when
the host ends the run; durable contact approval is rechecked for every request.
"""
from collections import OrderedDict
import hmac
import json
import os
import secrets
import threading

from .auth import read_key
from .contacts import pair as contact_pair
from .content import content_digest, snapshot
from .policy import Policy, VerifiedTurn, strict_json


class BridgeAuthority:
    def __init__(self, store):
        self.store, self.proofs = store, {}
        self.terminal_proofs = OrderedDict()

    def verify_turn(self, *, session_id, turn_id, **_):
        return self.proofs.get((session_id, turn_id))

    def is_active(self, proof):
        try:
            if not self.store.active(proof.grant_id):
                return False
            terminal = self.terminal_proofs.get((proof.session_id, proof.turn_id))
            if proof.channel == 'local-terminal':
                if terminal is None:
                    return False
                from .terminal import TerminalSessionStore
                return TerminalSessionStore(self.store).active(terminal, harness='openclaw') is not None
            return True
        except Exception:
            return False


class Bridge:
    def __init__(self, store, control_key_file, routes, *, instance_id=None):
        self.store, self.routes = store, routes
        self.instance_id = instance_id
        self.key = read_key(control_key_file).hex()
        self.authority = BridgeAuthority(store)
        self.policy = Policy(authority=self.authority, tool_policy=store.tool_enabled,
                             registry=store.registry, allowed_channels=store.registry.ids(),
                             channel_policy=store.channel_enabled,
                             trust={"system_prompt": "host", "tools": "host",
                                    "turn_context": "host", "history": "verified_owner"})
        self.runs, self.lock = OrderedDict(), threading.RLock()
        self.tool_scopes = {}
        self.claims = {}
        self.claimed_messages = OrderedDict()

    def _put(self, session, run):
        for token, pair in self.runs.items():
            if pair == (session, run):
                return token
        token = secrets.token_urlsafe(32)
        self.runs[token] = (session, run)
        while len(self.runs) > 4096:
            old, (sid, tid) = self.runs.popitem(last=False)
            self.tool_scopes.pop(old, None)
            self.claims.pop(old, None)
            self.policy.end(session_id=sid, turn_id=tid)
            self.authority.proofs.pop((sid, tid), None)
        return token

    def command(self, command, body):
        with self.lock:
            if command == 'ready':
                return {'pid': os.getpid(), 'instance': self.instance_id}
            if command == 'terminal_claim':
                from .terminal import TerminalSessionStore
                sid, tid = body.get('session'), body.get('run')
                text = body.get('text')
                if (not all(isinstance(v, str) and v and v == v.strip() for v in (sid, tid))
                        or not isinstance(text, str) or not text.strip()):
                    raise ValueError('Missing terminal submission')
                terminal = body.get('terminal_session')
                rendered = snapshot(body.get('rendered'))
                rendered_digest = content_digest(rendered)
                row = TerminalSessionStore(self.store).active(terminal, harness='openclaw')
                if row is None:
                    return {'mode': 'off'}
                prior_cap = next((cap for cap, location in self.runs.items() if location == (sid, tid)), None)
                prior = self.claims.get(prior_cap)
                if prior_cap and (not prior or prior.get('kind') != 'local_terminal'
                                  or prior.get('terminal_session') != terminal or prior.get('text') != text
                                  or prior.get('origin_rendered_digest') != rendered_digest):
                    raise ValueError('Run already has a different terminal binding')
                cap = self._put(sid, tid)
                self.claims.setdefault(cap, {'kind': 'local_terminal', 'terminal_session': terminal,
                    'text': text, 'origin_rendered_digest': rendered_digest, 'rendered_digest': None,
                    'grant_id': row['grant_id'], 'channel': 'local-terminal', 'bound': None})
                self.authority.terminal_proofs[(sid, tid)] = terminal
                while len(self.authority.terminal_proofs) > 8192:
                    self.authority.terminal_proofs.popitem(last=False)
                return {'mode': 'pending', 'capability': cap}
            if command == 'terminal_render':
                from .terminal import TerminalSessionStore
                cap = body.get('capability')
                claim = self.claims.get(cap)
                if (not claim or claim.get('kind') != 'local_terminal'
                        or self.runs.get(cap) != (body.get('session'), body.get('run'))
                        or TerminalSessionStore(self.store).active(claim['terminal_session'], harness='openclaw') is None):
                    return {'mode':'off'}
                rendered_digest = content_digest(snapshot(body.get('content')))
                if claim['rendered_digest'] is not None and claim['rendered_digest'] != rendered_digest:
                    raise ValueError('Terminal rendered content changed')
                claim['rendered_digest'] = rendered_digest
                return {'mode':'pending'}
            if command == 'claim_session':
                sid, tid = body.get('session'), body.get('run')
                if not all(isinstance(v, str) and v and v == v.strip() for v in (sid, tid)):
                    raise ValueError('Missing native session claim')
                if not self.store.registry.supports(body.get('channel'), 'native_session'):
                    raise ValueError('Channel has no installed native-session adapter')
                scope = contact_pair(body.get('channel'), body.get('account'), body.get('contact'), registry=self.store.registry)
                prior_cap = next((cap for cap, pair in self.runs.items() if pair == (sid, tid)), None)
                prior = self.claims.get(prior_cap)
                if prior_cap and (not prior or prior.get('kind') != 'native_session'
                                  or prior.get('source') != scope):
                    raise ValueError('Run already has a different native binding')
                grant = self.store.approved(*scope)
                if not grant:
                    return {'mode': 'off'}
                if prior and prior['grant_id'] != grant:
                    raise ValueError('Native session grant changed')
                cap = self._put(sid, tid)
                self.claims.setdefault(cap, {'kind': 'native_session', 'source': scope,
                                            'grant_id': grant, 'channel': scope[0], 'bound': None})
                return {'mode': 'pending', 'capability': cap}
            if command == 'claim':
                sid, tid = body.get('session'), body.get('run')
                messages = body.get('messages')
                if (not all(isinstance(v, str) and v for v in (sid, tid))
                        or not isinstance(messages, list) or not 0 < len(messages) <= 64):
                    raise ValueError('Missing native message claims')
                members, grants = [], []
                for row in messages:
                    if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
                        raise ValueError('Missing native message identity')
                    if not self.store.registry.supports(row.get('channel'), 'native_message'):
                        raise ValueError('Channel has no installed native-message adapter')
                    scope = contact_pair(row.get('channel'), row.get('account'), row.get('contact'), registry=self.store.registry)
                    grant = self.store.approved(*scope)
                    if not grant:
                        return {'mode': 'off'}
                    members.append((row['id'], *scope)); grants.append(grant)
                if len(set(members)) != len(members) or len(set(m[1:] for m in members)) != 1:
                    raise ValueError('Inconsistent native message group')
                if any(member in self.claimed_messages and self.runs.get(self.claimed_messages[member]) != (sid, tid)
                       for member in members):
                    raise ValueError('Native message was already claimed')
                cap = self._put(sid, tid)
                if (sid, tid) in self.authority.proofs and cap not in self.claims:
                    raise ValueError('Run already has a different native binding')
                claim = {'kind': 'native_message', 'members': tuple(members), 'grant_id': grants[0],
                         'channel': members[0][1], 'bound': None}
                prior = self.claims.get(cap)
                if prior and (prior.get('kind', 'native_message') != 'native_message'
                              or prior['members'] != claim['members'] or prior['grant_id'] != claim['grant_id']):
                    raise ValueError('Native claim changed')
                self.claims.setdefault(cap, claim)
                for member in members:
                    self.claimed_messages[member] = cap
                while len(self.claimed_messages) > 8192:
                    self.claimed_messages.popitem(last=False)
                return {'mode': 'pending', 'capability': cap}
            if command == 'bind':
                cap = body.get('capability')
                claim, location = self.claims.get(cap), self.runs.get(cap)
                if not claim or not location or not self.store.active(claim['grant_id']):
                    return {'mode': 'off'}
                content = snapshot(body.get('content'))
                if claim.get('kind') == 'local_terminal':
                    if content_digest(content) != claim['rendered_digest']:
                        raise ValueError('Terminal submission changed before dispatch')
                digest = content_digest(content)
                if claim['bound'] and claim['bound'] != digest:
                    raise ValueError('Rendered message changed')
                if not claim['bound']:
                    sid, tid = location
                    proof = VerifiedTurn(sid, tid, claim['channel'], content if isinstance(content, str) else '',
                        grant_id=claim['grant_id'], content_sha256=digest if isinstance(content, list) else '',
                        content_parts=len(content) if isinstance(content, list) else 0)
                    self.authority.proofs[location] = proof
                    self.policy.begin(session_id=sid, turn_id=tid, user_message=content, platform=claim['channel'])
                    claim['bound'] = digest
                return {'mode': 'owner' if self.policy.is_owner_work(*location) else 'off'}
            if command == "begin":
                fields = [body.get(k) for k in ("session", "run", "channel", "account", "contact", "text")]
                if any(not isinstance(v, str) or not v for v in fields):
                    raise ValueError("missing native ingress fields")
                sid, tid, channel, account, contact, text = fields
                if not self.store.registry.supports(channel, 'native_message'):
                    raise ValueError('Channel has no installed native-message adapter')
                if any(pair == (sid, tid) and cap in self.claims for cap, pair in self.runs.items()):
                    raise ValueError('Run already has a different native binding')
                grant = self.store.approved(channel, account, contact)
                if not grant:
                    self.policy.end(session_id=sid, turn_id=tid)
                    return {"mode": "off"}
                proof = VerifiedTurn(sid, tid, channel, text, grant_id=grant)
                prior = self.authority.proofs.get((sid, tid))
                if prior is not None and prior != proof:
                    raise ValueError("run identity changed")
                self.authority.proofs[(sid, tid)] = proof
                self.policy.begin(session_id=sid, turn_id=tid, user_message=text, platform=channel)
                return {"mode": "owner", "capability": self._put(sid, tid)}
            if command == "adopt":
                parent = self.runs.get(body.get("parent"))
                sid, tid, tool = body.get("session"), body.get("run"), body.get("tool")
                if not parent or not all(isinstance(v, str) and v for v in (sid, tid, tool)):
                    return {"mode": "off"}
                if any(pair == (sid, tid) and cap in self.claims for cap, pair in self.runs.items()):
                    raise ValueError('Run already has a different native binding')
                path = tuple(self.tool_scopes.get(body.get("parent"), {}).values()) + (tool,)
                self.policy.adopt(parent_session_id=parent[0], parent_turn_id=parent[1],
                                  child_session_id=sid, child_subagent_id=tid, tool_name=tool, tool_path=path)
                self.policy.begin(session_id=sid, turn_id=tid, user_message="", parent_session_id=parent[0])
                if not self.policy.is_owner_work(sid, tid):
                    return {"mode": "off"}
                return {"mode": "owner", "capability": self._put(sid, tid)}
            if command == "end":
                pair = self.runs.pop(body.get("capability"), None)
                self.tool_scopes.pop(body.get("capability"), None)
                self.claims.pop(body.get('capability'), None)
                if pair:
                    self.policy.end(session_id=pair[0], turn_id=pair[1])
                    self.authority.proofs.pop(pair, None)
                return {"ok": True}
            if command in {"tool_begin", "tool_end"}:
                cap, call = body.get("capability"), body.get("call")
                if cap not in self.runs or not isinstance(call, str) or not call:
                    raise ValueError("unknown tool scope")
                scopes = self.tool_scopes.setdefault(cap, {})
                if command == "tool_end":
                    scopes.pop(call, None)
                else:
                    tool = body.get("tool")
                    if not isinstance(tool, str) or not tool or tool != tool.strip() or len(scopes) >= 4096:
                        raise ValueError("invalid tool scope")
                    if call in scopes and scopes[call] != tool:
                        raise ValueError("tool scope changed")
                    scopes[call] = tool
                return {"ok": True}
            if command == "tool":
                pair = self.runs.get(body.get("capability"))
                return {"enabled": bool(pair and self.policy.is_owner_work(*pair)
                                         and self._tools_active(body.get("capability"))
                                         and self.policy.inherits(body.get("tool")))}
            raise ValueError("unknown command")

    def ticket(self, capability, body, route):
        with self.lock:
            pair = self.runs.get(capability)
            profile = self.routes.get(route, {}).get("profile", "off")
            if not pair or profile == "off" or not self._tools_active(capability):
                return ""
            return self.policy.issue(body, session_id=pair[0], turn_id=pair[1], profile=profile, route=route) or ""

    def _tools_active(self, capability):
        return all(self.policy.inherits(tool) for tool in self.tool_scopes.get(capability, {}).values())

    def http(self, handler):
        status, result = 403, {"error": "control authentication required"}
        supplied = handler.headers.get("Authorization", "")
        if hmac.compare_digest(supplied, "Bearer " + self.key):
            try:
                length = int(handler.headers.get("Content-Length", "0"))
                limit = 64 * 1024 * 1024 if handler.path == '/_owner/bind' else 1024 * 1024
                if not 0 < length <= limit or handler.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid control body")
                body = strict_json(handler.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("invalid control object")
                result = self.command(handler.path.removeprefix("/_owner/"), body)
                status = 200
            except Exception:
                status, result = 400, {"error": "control request rejected"}
        raw = json.dumps(result, separators=(",", ":")).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
