# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Verified owner work and single-use request capabilities.

An authenticated turn retains its owner identity through its tool loop and
eligible delegated work. Context is validated and forwarded without rewriting.
Identity failures affect the current turn; no session content state is stored.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import secrets
import threading
import time

from .content import proof_matches
from .channels import DEFAULT_ENABLED, DEFAULT_REGISTRY

CHANNELS = frozenset(DEFAULT_ENABLED) | DEFAULT_REGISTRY.ids("custom_authority")
TICKET_HEADER = "X-Owner-Mode-Ticket"
# Hermes' row-first compaction carrier (Hermes context_compressor _MERGED_PRIOR_CONTEXT_HEADER): one header line, then the row.
MERGED_PRIOR = "[PRIOR CONTEXT — for reference only; not a new message]"
CONTEXT_HEADER = "X-Owner-Mode-Context"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError("duplicate JSON key")
            obj[k] = v
        return obj
    def constant(_):
        raise ValueError("non-finite JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


@dataclass(frozen=True)
class VerifiedTurn:
    """Returned only by an operator-installed adapter to existing ingress auth.

    Never construct this from a request body, model text, sender ID, or headers.
    The adapter binds the rendered text/media to authenticated native ingress.
    Unverifiable transport or composition paths cannot produce a proof.
    """
    session_id: str
    turn_id: str
    channel: str
    own_text: str
    grant_id: str = ""
    content_sha256: str = ""
    content_parts: int = 0


@dataclass
class Turn:
    proof: VerifiedTurn | None
    started: float
    identity_ms: int | None = None  # time spent in the identity adapter (audit only)
    # Set once a call of this turn passed the owner's-words check, or at once for a child adopted from an owner
    # turn: later calls of the turn carry its tool loop, not his words, and pass on shape alone.
    anchored: bool = False
    steered: bool = False  # a child turn steered from outside owner work (see steered); off for good
    tool_path: tuple[str, ...] = ()


TRUST_DEFAULTS = {"system_prompt": "pinned", "tools": "pinned", "turn_context": "none", "history": "none"}
TRUST_CHOICES = {"system_prompt": {"pinned", "host"}, "tools": {"pinned", "host"},
                 "turn_context": {"none", "host"}, "history": {"none", "verified_owner"}}


def parse_trust(raw):
    """Unknown or malformed trust values fall back to the strictest choice."""
    trust = dict(TRUST_DEFAULTS)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in TRUST_CHOICES and value in TRUST_CHOICES[key]:
                trust[key] = value
    return trust


class Policy:
    def __init__(self, *, trusted_system_hashes=(), trusted_tools_hashes=(),
                 authority=None, clock=time.monotonic, ticket_ttl=5,
                 trust=None, allowed_channels=None, tool_policy=None,
                 registry=DEFAULT_REGISTRY, channel_policy=None):
        self.lock = threading.RLock()
        self.authority = authority
        self.tool_policy = tool_policy
        self.clock, self.ticket_ttl = clock, ticket_ttl
        self.systems = frozenset(trusted_system_hashes)
        self.tools = frozenset(trusted_tools_hashes)
        self.trust = parse_trust(trust)
        self.registry = registry
        self.channels = frozenset(c for c in (CHANNELS if allowed_channels is None else allowed_channels)
                                  if registry.get(c) is not None)
        self.channel_policy = channel_policy
        self.turns = OrderedDict()
        self.tickets = OrderedDict()
        self.contexts = OrderedDict()
        self.context_infos = {}
        self.children = OrderedDict()  # child session -> (parent session, the owner turn's proof, child subagent id)

    def begin(self, *, session_id="", turn_id="", user_message="", parent_session_id="", **context):
        key = (session_id, turn_id)
        # Invalidate old decisions BEFORE calling any fallible identity code.
        with self.lock:
            self.turns.pop(key, None)
            pending = Turn(None, self.clock())
            self.turns[key] = pending
            while len(self.turns) > 4096:
                self.turns.popitem(last=False)
        try:
            if not session_id:
                return
            if not turn_id or not isinstance(user_message, (str, list)):
                return
            if parent_session_id:
                # A delegated child is owner work only when adopt() saw it created during an owner turn.
                with self.lock:
                    adopted = self.children.get(session_id)
                    if adopted and adopted[0] == parent_session_id and self.turns.get(key) is pending:
                        pending.proof, pending.anchored, pending.tool_path = adopted[1], True, adopted[3]
                        return
                return
            platform = context.get("platform")
            if (not isinstance(platform, str) or platform.lower() not in self.channels
                    or not self._channel_enabled(platform.lower())):
                # Only installed, configured adapters may provide an authority
                # proof. A request's platform label cannot register an adapter.
                return
            asked = time.monotonic()
            try:
                proof = self.authority.verify_turn(session_id=session_id, turn_id=turn_id,
                                                   user_message=user_message, **context) if self.authority else None
            finally:
                with self.lock:
                    if self.turns.get(key) is pending:
                        pending.identity_ms = int((time.monotonic() - asked) * 1000)
            if (not isinstance(proof, VerifiedTurn) or proof.session_id != session_id or proof.turn_id != turn_id
                    or proof.channel not in self.channels or not self._proof_matches(user_message, proof)
                    or proof.channel != platform.lower()):
                return
            with self.lock:
                if self.turns.get(key) is pending:
                    pending.proof = proof
        except Exception:
            # The pending turn has no proof. A later turn must authenticate afresh.
            return

    def inherits(self, tool_name):
        if not isinstance(tool_name, str) or not tool_name or tool_name != tool_name.strip():
            return False
        try:
            return self.tool_policy(tool_name) is True if self.tool_policy else True
        except Exception:
            return False

    def _active(self, turn):
        if not turn or not turn.proof or not all(self.inherits(n) for n in turn.tool_path):
            return False
        if turn.proof.channel not in self.channels or not self._channel_enabled(turn.proof.channel):
            return False
        try:
            check = getattr(self.authority, "is_active", None)
            return check(turn.proof) is True if callable(check) else True
        except Exception:
            return False

    def _channel_enabled(self, channel):
        try:
            return self.channel_policy(channel) is True if self.channel_policy else True
        except Exception:
            return False

    def adopt(self, *, parent_session_id="", parent_turn_id="", child_session_id="", child_subagent_id="",
              tool_name="delegate_task", tool_path=(), **_):
        """subagent_start: a child created during an owner turn is that owner's work (its own turns, see begin)."""
        with self.lock:
            turn = self._turn(parent_session_id, parent_turn_id)
            path = tuple(tool_path) or (tool_name,)
            if (child_session_id and self._active(turn) and turn.anchored
                    and all(self.inherits(name) for name in path)):
                self.children[child_session_id] = (parent_session_id, turn.proof,
                                                  str(child_subagent_id or ""), turn.tool_path + path)
                while len(self.children) > 4096:
                    self.children.popitem(last=False)

    def steered(self, *, tool_name="", args=None, session_id="", turn_id="", **_):
        """pre_tool_call: delegate_task's steer action writes into a running child. A steer from a turn that is not
        owner work (the current turn carries the owner words) takes owner mode from that child for good:
        its current turn now, and no adoption for a later one. Never blocks or changes the call."""
        if (tool_name != "delegate_task" or not isinstance(args, dict)
                or str(args.get("action") or "").strip().lower() != "steer"):
            return None
        with self.lock:
            turn = self._turn(session_id, turn_id)
            if self._active(turn) and turn.anchored and self.inherits(tool_name):
                return None
            # Hermes resolves the target by this exact id (delegate_tool_registry _handle_control_action).
            target = str(args.get("subagent_id") or "").strip()
            for child in [c for c, row in self.children.items() if row[2] == target]:
                del self.children[child]
                for (sid, _), t in self.turns.items():
                    if sid == child:
                        t.proof, t.steered = None, True
        return None

    def _turn(self, session_id, turn_id):
        """A compaction mid-turn moves the agent to a new session id and keeps the turn id, which Hermes mints
        fresh for every turn of every agent: a call under a new session id finds its turn by that id alone."""
        turn = self.turns.get((session_id, turn_id))
        if turn is None and turn_id:
            found = [t for (_, tid), t in self.turns.items() if tid == turn_id]
            turn = found[0] if len(found) == 1 else None
        return turn

    def end(self, *, session_id="", turn_id="", **_):
        with self.lock:
            for key in [k for k in self.turns if k == (session_id, turn_id) or (turn_id and k[1] == turn_id)]:
                del self.turns[key]
            for token, row in list(self.tickets.items()):
                if row[1:3] == (session_id, turn_id):
                    del self.tickets[token]
            for token, row in list(self.contexts.items()):
                if row[1:3] == (session_id, turn_id):
                    del self.contexts[token]
                    self.context_infos.pop(token, None)

    def context_ticket(self, body, *, session_id, turn_id, route, info=None):
        """Bind audit metadata to a request without granting owner mode."""
        with self.lock:
            token = secrets.token_urlsafe(32)
            bound = {k: v for k, v in body.items() if k not in {"stream", "stream_options"}}
            self.contexts[token] = (self.clock(), session_id, turn_id, route, digest(bound))
            self.context_infos[token] = dict(info or {})
            while len(self.contexts) > 4096:
                old, _ = self.contexts.popitem(last=False)
                self.context_infos.pop(old, None)
            return token

    def consume_context(self, token, body, route):
        """Return bound audit metadata once; absent or altered capabilities give no metadata."""
        with self.lock:
            row = self.contexts.pop(token, None)
            info = self.context_infos.pop(token, {})
            if not row:
                return {}
            issued, sid, tid, intended_route, bound = row
            actual = {k: v for k, v in body.items() if k not in {"stream", "stream_options"}}
            if self.clock() - issued > self.ticket_ttl or intended_route != route or bound != digest(actual):
                return {}
            return dict(info)

    ALLOWED_KEYS = frozenset({"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "temperature",
                              "top_p", "max_tokens", "max_completion_tokens", "max_output_tokens", "stream",
                              "stream_options", "stop",
                              "frequency_penalty", "presence_penalty", "seed", "n", "response_format",
                              "owner_mode", "owner_mode_scale", "reasoning_effort", "top_k", "min_p",
                              "chat_template_kwargs"})

    def _owner_words(self, content, own_text):
        """Exactly the owner's words, or (trusted host context) the words with host text joined at blank lines."""
        if not isinstance(content, str) or not own_text:
            return False
        if content == own_text:
            return True
        if content.startswith(MERGED_PRIOR + "\n"):
            # Compaction folded a summary into his row (Hermes _merge_summary_into_tail_row): its header line counts
            # as a blank line, as the summary-first carrier's join already does.
            content = MERGED_PRIOR + "\n\n" + content[len(MERGED_PRIOR) + 1:]
        # Host text may sit before the owner's words (a redirect note, a reply pointer, an attachment note), after
        # them (injected context), or both. The plaintext form of host_joined's rule: linear, on every owner turn.
        return self.trust["turn_context"] == "host" and ("\n\n" + own_text + "\n\n") in ("\n\n" + content + "\n\n")

    def _proof_matches(self, content, proof):
        if isinstance(content, str) and not proof.content_sha256:
            return self._owner_words(content, proof.own_text)
        return proof_matches(content, proof, allow_text_context=self.trust['turn_context'] == 'host')

    def check(self, body, proof, session_id="", anchored=False):
        """Return "" when the final body may carry owner mode, else a fixed reason code (never content).

        A turn's first call must end with his words. Once anchored (a call of the turn passed that, or a child
        adopted from an owner turn) the tool loop's rows may follow them, so every row is checked for shape alone."""
        if not isinstance(body, dict) or set(body) - self.ALLOWED_KEYS:
            return "unknown_request_field"
        if body.get("n", 1) != 1:
            return "n_not_1"
        if "stream" in body and type(body["stream"]) is not bool:
            return "bad_stream"
        options = body.get("stream_options", {})
        if (not isinstance(options, dict) or set(options) - {"include_usage"}
                or ("include_usage" in options and type(options["include_usage"]) is not bool)):
            return "bad_stream_options"
        template = body.get("chat_template_kwargs", {})
        if (not isinstance(template, dict) or set(template) - {"enable_thinking", "reasoning_effort"}
                or ("enable_thinking" in template and type(template["enable_thinking"]) is not bool)
                or template.get("reasoning_effort", "low") not in {"none", "low", "medium", "high", "xhigh", "max"}):
            return "bad_chat_template_kwargs"
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return "no_messages"
        last, history, reason = messages[-1], messages[:-1], ""
        if not isinstance(last, dict) or last.get("role") != "user" or set(last) != {"role", "content"}:
            reason = "last_not_owner_user"
        elif not self._proof_matches(last["content"], proof):
            reason = "last_not_owner_words" if self.trust["turn_context"] == "host" else "turn_context_untrusted"
        if reason:
            if not anchored:
                return reason
            history = messages
        for message in history:
            reason = self._row(message)
            if reason:
                return reason
        if body.get("tools") and self.trust["tools"] != "host" and digest(body["tools"]) not in self.tools:
            return "tools_unpinned"
        # These can carry free text. Accept only plain selectors and no schema injection.
        if body.get("response_format") not in (None, {"type": "text"}):
            return "response_format"
        if body.get("tool_choice") not in (None, "auto", "none", "required"):
            return "tool_choice"
        return ""

    def _row(self, message):
        """Shape only. Native tool results may retain their name on the wire;
        this metadata does not authenticate a sender or grant tool access."""
        if not isinstance(message, dict):
            return "bad_message"
        role = message.get("role")
        if role == "system":
            if set(message) != {"role", "content"}:
                return "bad_system_message"
            if self.trust["system_prompt"] != "host" and digest(message) not in self.systems:
                return "system_prompt_unpinned"
            return ""
        if role not in {"user", "assistant", "tool"}:
            return "unknown_role"
        if self.trust["history"] != "verified_owner":
            return "history_present"
        if role == "user":
            if set(message) != {"role", "content"} or not isinstance(message["content"], (str, list)):
                return "bad_user_history"
        elif role == "tool":
            required = {"role", "content", "tool_call_id"}
            if (not required <= set(message) or set(message) - required - {"name"}
                    or not isinstance(message["tool_call_id"], str)
                    or ("name" in message and (not isinstance(message["name"], str) or not message["name"].strip()))
                    or not isinstance(message["content"], (str, list))):
                return "bad_tool_history"
        else:
            calls = message.get("tool_calls", [])
            if (set(message) - {"role", "content", "reasoning_content", "reasoning", "tool_calls"}
                    or not isinstance(message.get("content"), (str, type(None)))
                    or any(not isinstance(message.get(k), (str, type(None))) for k in ("reasoning_content", "reasoning"))
                    or not isinstance(calls, list)
                    or not all(isinstance(call, dict) and isinstance(call.get("function"), dict) for call in calls)):
                return "bad_assistant_history"
        return ""

    def _clean(self, body, proof, session_id=""):
        return not self.check(body, proof, session_id)

    def identity_ms(self, session_id, turn_id):
        """Milliseconds the identity adapter took for this turn (None if it was not asked). Audit only."""
        with self.lock:
            turn = self._turn(session_id, turn_id)
            return turn.identity_ms if turn else None

    def is_owner_work(self, session_id, turn_id):
        """Whether this turn has an owner proof; used only for audit metadata."""
        try:
            with self.lock:
                turn = self._turn(session_id, turn_id)
                return self._active(turn)
        except Exception:
            return False

    def owner_turn(self, session_id="", turn_id="", tool_name=""):
        """True while this exact turn is one he verified himself; read by other plugins (the public owner_turn contract). An adopted child holds its parent's proof, whose turn id is not its own; a steered or synthetic turn
        (a completion, an injection, a resume) holds none. Never raises."""
        try:
            with self.lock:
                turn = self._turn(session_id, turn_id)
                proof = turn.proof if turn else None
                return bool(self._active(turn) and proof.turn_id == turn_id
                            and (not tool_name or self.inherits(tool_name)))
        except Exception:
            return False

    def decide(self, body, *, session_id, turn_id, profile, route):
        """Issue a single-use ticket or return (None, reason)."""
        with self.lock:
            turn = self._turn(session_id, turn_id)
            if turn and turn.steered:
                return None, "non_owner_steer"
            if not self._active(turn):
                return None, "owner_unverified"
            # Work retains its identity until the host ends the turn or revokes it.
            anchored = turn.anchored
            reason = self.check(body, turn.proof, session_id, anchored)
            if reason:
                return None, reason
            turn.anchored = True
            token = secrets.token_urlsafe(32)
            # Streaming flags are added by the stock transport after middleware;
            # every other field, including model and all prompt fields, is bound.
            bound = {k: v for k, v in body.items() if k not in {"stream", "stream_options"}}
            self.tickets[token] = (self.clock(), session_id, turn_id, profile, route, digest(bound), anchored)
            while len(self.tickets) > 4096:
                self.tickets.popitem(last=False)
            return token, "owner_clean"

    def issue(self, body, *, session_id, turn_id, profile, route):
        return self.decide(body, session_id=session_id, turn_id=turn_id, profile=profile, route=route)[0]

    def consume(self, token, body, route):
        with self.lock:
            row = self.tickets.pop(token, None)
            if not row:
                return "off"
            issued, sid, tid, profile, intended_route, bound, anchored = row
            turn = self._turn(sid, tid)
            if (self.clock() - issued > self.ticket_ttl or route != intended_route
                    or not self._active(turn)):
                return "off"
            actual = {k: v for k, v in body.items() if k not in {"stream", "stream_options"}}
            if digest(actual) != bound or self.check(body, turn.proof, sid, anchored):
                return "off"
            return profile


def flatten_request(request):
    """Mirror SDK extra_body merging without letting it bypass provenance checks."""
    body = {k: v for k, v in request.items() if k not in {"extra_body", "extra_headers", "timeout", "extra_query"}}
    extra = request.get("extra_body", {})
    if not isinstance(extra, dict):
        raise ValueError("invalid extra body")
    body.update(extra)
    return body
