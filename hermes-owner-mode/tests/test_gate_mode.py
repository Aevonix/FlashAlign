# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Gate mode (stock passthrough), per-route gain, trust settings, audit lines and process ownership."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

import httpx

import test_owner_mode  # noqa: F401  (loads the plugin package as owner_mode)
from owner_mode.policy import Policy, VerifiedTurn, digest, flatten_request
from owner_mode.proxy import ProviderProxy, UNAVAILABLE_MESSAGE, owner_400_case
from owner_mode.plugin import Runtime, serving_process
from support import Engine

OWN = "Please answer my question."
INJECTED = OWN + "\n\n<memory-context>host recall block</memory-context>"


class Authority:
    def __init__(self, channel="whatsapp", owner=True):
        self.channel, self.owner = channel, owner

    def verify_turn(self, session_id, turn_id, user_message, **_):
        return VerifiedTurn(session_id, turn_id, self.channel, user_message) if self.owner else None


class GateBase(unittest.TestCase):
    trust = {"system_prompt": "host", "tools": "host", "turn_context": "host", "history": "verified_owner"}
    route = {"profile": "owner-test", "owner_mode_scale": 0.6}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.key = os.urandom(32)
        self.keyfile = root / "key"
        self.keyfile.write_bytes(self.key)
        self.keyfile.chmod(0o600)
        self.rows = []
        self.audit_recorded = threading.Event()
        def audit(row):
            self.rows.append(row)
            self.audit_recorded.set()
        self.policy = Policy(authority=Authority(), trust=self.trust,
                             allowed_channels=("whatsapp", "sms", "rcs"))
        self.engine = Engine(self.key)
        self.proxy = ProviderProxy(self.policy, {"head": dict(self.route, upstream=self.engine.base_url)},
                                   self.keyfile, audit=audit)
        self.proxy.start()
        self.runtime = Runtime(self.policy, self.proxy)
        self.tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]

    def tearDown(self):
        self.proxy.close()
        self.engine.close()
        self.tmp.cleanup()

    def body(self, *history, last=INJECTED, **extra):
        messages = [{"role": "system", "content": "Dynamic host system prompt."}, *history,
                    {"role": "user", "content": last}]
        return dict({"model": "bench", "messages": messages, "tools": self.tools, "temperature": 1.0}, **extra)

    def send(self, body, *, sid="s", tid="t", platform="whatsapp", middleware=True):
        self.audit_recorded.clear()
        request = self.runtime.request(body, session_id=sid, turn_id=tid, platform=platform,
                                       base_url=self.proxy.base_url("head"))["request"] if middleware else body
        with httpx.Client(timeout=5) as client:
            response = client.post(self.proxy.base_url("head") + "/chat/completions", json=flatten_request(request),
                                   headers=request.get("extra_headers", {}))
        # A buffered error response can reach the client before the handler's
        # finally block records its audit row. Wait for that callback explicitly.
        self.assertTrue(self.audit_recorded.wait(5), "Proxy audit callback did not complete")
        return response, self.engine.records[-1]


class TestScale(GateBase):
    def test_owner_call_carries_configured_scale_only(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        response, row = self.send(dict(self.body(), extra_body={"owner_mode_scale": 0.01, "owner_mode": "evil"}))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(row["effective"], "owner-test")
        self.assertEqual(row["body"]["owner_mode"], "owner-test")
        self.assertEqual(row["body"]["owner_mode_scale"], 0.6)
        self.assertEqual(response.headers["X-Owner-Mode-Effective"], "owner-test")

    def test_client_scale_stripped_on_stock_call(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        _, row = self.send(dict(self.body(), owner_mode="owner-test", owner_mode_scale=5.0), middleware=False)
        self.assertEqual(row["effective"], "off")
        self.assertNotIn("owner_mode", row["body"])
        self.assertNotIn("owner_mode_scale", row["body"])

    def test_client_scale_stripped_on_non_owner_call(self):
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id="n", turn_id="t", user_message=OWN, platform="whatsapp")
        _, row = self.send(dict(self.body(), extra_body={"owner_mode_scale": 1.0}), sid="n")
        self.assertEqual(row["effective"], "off")
        self.assertNotIn("owner_mode_scale", row["body"])
        self.assertNotIn("owner_mode", row["body"])

    def test_invalid_scale_disables_route_but_serves_stock(self):
        for bad in (0, -1, 1.5, float("nan"), "0.6", True):
            with self.subTest(bad=bad):
                proxy = ProviderProxy(self.policy, {"head": {"upstream": self.engine.base_url, "profile": "owner-test",
                                                             "owner_mode_scale": bad}}, self.keyfile)
                self.assertIn("head", proxy.disabled)
                proxy.server.server_close()

    def test_no_scale_configured_sends_none(self):
        proxy = ProviderProxy(self.policy, {"head": {"upstream": self.engine.base_url, "profile": "owner-test"}},
                              self.keyfile)
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        body = flatten_request(self.body())
        token, reason = self.policy.decide(body, session_id="s", turn_id="t", profile="owner-test", route="head")
        raw, headers, mode = proxy.prepare("head", body, token)
        proxy.server.server_close()
        self.assertEqual(mode, "owner-test")
        self.assertNotIn("owner_mode_scale", json.loads(raw))
        self.assertIn("X-Owner-Mode-Auth", headers)


class TestGateDecisions(GateBase):
    def test_owner_turn_signed_and_audited_without_content(self):
        session_id = "private-channel-session-58129"
        self.policy.begin(session_id=session_id, turn_id="t", user_message=OWN, platform="whatsapp")
        response, row = self.send(self.body(), sid=session_id)
        self.assertEqual(row["effective"], "owner-test")
        audit = self.rows[-1]
        self.assertNotIn("session", audit)
        self.assertEqual((audit["platform"], audit["owner_verified"],
                          audit["decision"], audit["owner_mode_sent"], audit["owner_mode_scale_sent"],
                          audit["effective"]), ("whatsapp", True, "owner_clean", "owner-test", 0.6,
                                                "owner-test"))
        text = json.dumps(self.rows)
        for secret in (OWN, "host recall", "Dynamic host", self.key.hex(), session_id):
            self.assertNotIn(secret, text)

    def test_audit_filters_identity_metadata_in_all_activation_modes(self):
        private_values = {
            "session": "private-session-48211", "turn": "private-turn-93418",
            "sender_id": "private-sender-38214", "chat_id": "private-chat-29014",
            "handle": "private-handle-84519", "prompt": "private-prompt-64281",
            "headers": {"X-Owner-Mode-Auth": "private-auth-header-45219"},
        }
        for activation_mode in ("verified", "always_on", "off"):
            with self.subTest(activation_mode=activation_mode):
                self.proxy.record("head", "owner-test", "owner-test", info={
                    **private_values, "activation_mode": activation_mode,
                    "owner_verified": activation_mode == "verified",
                    "decision": "always_on" if activation_mode == "always_on" else "owner_clean",
                }, status=200)
                audit = self.rows[-1]
                self.assertEqual(audit["activation_mode"], activation_mode)
                self.assertTrue(set(private_values).isdisjoint(audit))
                for value in private_values.values():
                    self.assertNotIn(str(value), json.dumps(audit))

    def test_same_text_from_non_owner_is_off(self):
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id="other", turn_id="t", user_message=OWN, platform="whatsapp")
        _, row = self.send(self.body(), sid="other")
        self.assertEqual(row["effective"], "off")
        self.assertEqual(self.rows[-1]["decision"], "owner_unverified")
        self.assertFalse(self.rows[-1]["owner_verified"])

    def test_unanchored_tool_rows_do_not_prove_ownership(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        call = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_extract", "arguments": "{}"}}]}
        tool = {"role": "tool", "tool_call_id": "c1", "content": "PAGE TEXT"}
        body = self.body()
        body["messages"] += [call, tool]
        _, row = self.send(body)
        self.assertEqual(row["effective"], "off")
        self.assertEqual(self.rows[-1]["decision"], "last_not_owner_user")
        self.assertEqual(row["body"]["messages"], body["messages"])  # passthrough: unchanged content
        self.policy.begin(session_id="s", turn_id="t2", user_message=OWN, platform="whatsapp")
        _, row = self.send(self.body(), tid="t2")
        self.assertEqual(row["effective"], "owner-test")
        self.assertEqual(self.rows[-1]["decision"], "owner_clean")


    def test_cron_platform_never_owner_even_with_owner_authority(self):
        for platform in ("cron", "cli", "api_server", "email", "voice", "companion", "kanban", ""):
            with self.subTest(platform=platform):
                sid = "cron-" + platform
                self.policy.begin(session_id=sid, turn_id="t", user_message=OWN, platform=platform)
                response, row = self.send(self.body(last=OWN), sid=sid, platform=platform)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(row["effective"], "off")
                self.assertEqual(row["body"]["messages"][0]["content"], "Dynamic host system prompt.")
                self.assertEqual(row["body"]["tools"], self.tools)

    def test_adapter_channel_must_match_platform(self):
        self.policy.authority = Authority(channel="sms")
        self.policy.begin(session_id="x", turn_id="t", user_message=OWN, platform="whatsapp")
        _, row = self.send(self.body(), sid="x")
        self.assertEqual(row["effective"], "off")

    def test_history_of_verified_owner_turns_allowed(self):
        self.policy.begin(session_id="s", turn_id="t1", user_message="First question", platform="whatsapp")
        self.policy.end(session_id="s", turn_id="t1")
        self.policy.begin(session_id="s", turn_id="t2", user_message=OWN, platform="whatsapp")
        history = [{"role": "user", "content": "First question\n\nold recall"}, {"role": "assistant", "content": "Answer"}]
        _, row = self.send(self.body(*history), tid="t2")
        self.assertEqual(row["effective"], "owner-test")

    def test_unverified_history_is_admitted(self):
        # History passes as Hermes sent it; only its shape is checked.
        self.policy.begin(session_id="s", turn_id="t2", user_message=OWN, platform="whatsapp")
        history = [{"role": "user", "content": "Text the plugin never verified"}, {"role": "assistant", "content": "A"}]
        body = self.body(*history)
        _, row = self.send(body, tid="t2")
        self.assertEqual(row["effective"], "owner-test")
        self.assertEqual(self.rows[-1]["decision"], "owner_clean")
        self.assertEqual(row["body"]["messages"], body["messages"])

    def test_identity_exception_does_not_authorize_the_turn(self):
        class Boom:
            def verify_turn(self, **_):
                raise RuntimeError("down")
        self.policy.authority = Boom()
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.assertFalse(self.policy.is_owner_work("s", "t"))
        self.policy.authority = Authority()
        self.policy.begin(session_id="s", turn_id="next", user_message=OWN, platform="whatsapp")
        _, row = self.send(self.body(), tid="next")
        self.assertEqual(row["effective"], "owner-test")

    def test_stock_passthrough_preserves_body(self):
        self.policy.authority = Authority(owner=False)
        body = self.body(top_k=20, min_p=0.05, prompt_cache_key="x")
        _, row = self.send(body, sid="z")
        self.assertEqual(row["body"], body)

    def test_models_get_passthrough(self):
        with httpx.Client(timeout=5) as client:
            response = client.get(self.proxy.base_url("head") + "/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "bench")

    def test_upstream_down_fails_fast_with_502(self):
        self.engine.close()
        import time
        started = time.monotonic()
        with httpx.Client(timeout=30) as client:
            response = client.post(self.proxy.base_url("head") + "/chat/completions", json=self.body())
        self.assertEqual(response.status_code, 502)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(self.rows[-1]["status"], 502)
        self.engine = Engine(self.key)  # for tearDown


class TestOwner400(GateBase):
    def owner_turn(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        return self.send(self.body())

    def test_valid_owner_call_is_on_with_no_retry(self):
        response, _ = self.owner_turn()
        self.assertEqual(response.headers["X-Owner-Mode-Effective"], "owner-test")
        self.assertEqual(len(self.engine.records), 1)

    def test_key_absent_400_retried_once_as_stock(self):
        self.engine.refuse_owner_mode = True
        with self.assertLogs("owner_mode.proxy", "WARNING") as logs:
            response, _ = self.owner_turn()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Owner-Mode-Effective"], "off")
        self.assertEqual(len(self.engine.records), 2)
        first, second = self.engine.records
        self.assertEqual((first["body"]["owner_mode"], first["body"]["owner_mode_scale"]), ("owner-test", 0.6))
        self.assertNotIn("owner_mode", second["body"])
        self.assertNotIn("owner_mode_scale", second["body"])
        self.assertEqual(second["auth_reason"], "off")  # unsigned
        text = "".join(logs.output)
        self.assertIn("OWNER_MODE_UNAVAILABLE", text)
        self.assertNotIn("v1:", text)
        self.assertNotIn(self.key.hex(), text)
        self.assertEqual(len(self.rows), 1)
        audit = self.rows[-1]
        self.assertEqual((audit["decision"], audit["owner_mode_sent"], audit["owner_mode_scale_sent"],
                          audit["effective"], audit["status"]), ("owner_mode_unavailable", "off", None, "off", 200))

    def test_bad_scale_400_surfaces_and_is_never_demoted(self):
        self.proxy.scales["head"] = 2.5  # e.g. an engine whose accepted range differs from the wrapper's
        with self.assertLogs("owner_mode.proxy", "ERROR") as logs:
            response, _ = self.owner_turn()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["message"], "owner_mode_scale must be a number in (0, 2]")
        self.assertEqual(len(self.engine.records), 1)
        self.assertIn("OWNER_MODE_REJECTED", "".join(logs.output))
        self.assertIn("case=bad_scale", "".join(logs.output))
        self.assertEqual((self.rows[-1]["decision"], self.rows[-1]["status"]), ("owner_mode_rejected_bad_scale", 400))

    def test_unknown_profile_400_surfaces_without_retry(self):
        self.engine.profiles = set()
        response, _ = self.owner_turn()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.engine.records), 1)
        self.assertEqual(self.rows[-1]["decision"], "owner_mode_rejected_unknown_profile")

    def test_non_400_owner_failure_is_not_resent(self):
        self.engine.fail_models["bench"] = 500
        response, _ = self.owner_turn()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(self.engine.records), 1)
        self.assertEqual(self.rows[-1]["decision"], "owner_clean")

    def test_400_on_stock_call_passes_through_unretried(self):
        self.engine.fail_models["bench"] = 400
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id="n", turn_id="t", user_message=OWN, platform="whatsapp")
        response, _ = self.send(self.body(), sid="n")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.engine.records), 1)

    def test_classifier_matches_message_exactly(self):
        def body(message, code="invalid_owner_mode"):
            return json.dumps({"error": {"message": message, "type": "invalid_request_error", "param": "owner_mode",
                                         "code": code}}).encode()
        # Pinned engine build: the exact full string of the 'no owner-mode profiles' 400.
        exact = 'this server was started without owner-mode profiles: owner_mode must be "off"'
        self.assertEqual(UNAVAILABLE_MESSAGE, exact)
        self.assertEqual(owner_400_case(body(exact)), "no_profiles")
        self.assertEqual(owner_400_case(body(exact[:-1])), "other_owner_mode")
        self.assertEqual(owner_400_case(body("this server was started without owner-mode profiles")), "other_owner_mode")
        self.assertEqual(owner_400_case(body(UNAVAILABLE_MESSAGE + " ")), "other_owner_mode")
        self.assertEqual(owner_400_case(body(UNAVAILABLE_MESSAGE.upper())), "other_owner_mode")
        self.assertEqual(owner_400_case(body(UNAVAILABLE_MESSAGE, code="other")), "other")
        self.assertEqual(owner_400_case(b"not json"), "other")


class TestStrictDefaults(GateBase):
    trust = None  # strictest choices

    def test_strict_trust_refuses_live_shaped_request(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        _, row = self.send(self.body())
        self.assertEqual(row["effective"], "off")
        self.assertEqual(self.rows[-1]["decision"], "turn_context_untrusted")

    def test_unknown_trust_values_fall_back_to_strict(self):
        policy = Policy(trust={"system_prompt": "anything", "history": 1})
        self.assertEqual(policy.trust, {"system_prompt": "pinned", "tools": "pinned", "turn_context": "none",
                                        "history": "none"})


class FixedAuthority:
    """Verifies every turn with one fixed own_text, so the proof can differ from the turn content."""
    def __init__(self, own_text):
        self.own_text = own_text

    def verify_turn(self, session_id, turn_id, **_):
        return VerifiedTurn(session_id, turn_id, "whatsapp", self.own_text)


class TrustRules(unittest.TestCase):
    """Host-context suffix, queued composite owner text and prior-row suffix acceptance in Policy."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def make_policy(self, turn_context, authority=None):
        return Policy(authority=authority, trust={"turn_context": turn_context})

    def test_host_suffix_only_with_host_turn_context(self):
        content = "host recall\n\n" + OWN
        self.assertTrue(self.make_policy("host")._owner_words(content, OWN))
        self.assertFalse(self.make_policy("none")._owner_words(content, OWN))
        self.assertFalse(self.make_policy("host")._owner_words("x" + OWN, OWN))  # no "\n\n" boundary

    def test_host_text_both_sides_only_with_host_turn_context(self):
        content = "[Replying to: \"quoted\"]\n\n" + OWN + "\n\nhost recall"
        self.assertTrue(self.make_policy("host")._owner_words(content, OWN))
        self.assertFalse(self.make_policy("none")._owner_words(content, OWN))
        self.assertFalse(self.make_policy("host")._owner_words("pointer\n\n" + OWN + "x\n\nrecall", OWN))
        self.assertFalse(self.make_policy("host")._owner_words("pointer\n\nx" + OWN + "\n\nrecall", OWN))

    def test_queued_composite_owner_text(self):
        composite = "first\nsecond"  # two queued owner messages joined by a plain newline
        body = {"model": "bench", "messages": [{"role": "user", "content": composite}]}
        whole = self.make_policy("host", FixedAuthority(composite))
        whole.begin(session_id="c", turn_id="t", user_message=composite, platform="whatsapp")
        _, reason = whole.decide(body, session_id="c", turn_id="t", profile="owner-test", route="head")
        self.assertEqual(reason, "owner_clean")
        member = self.make_policy("host", FixedAuthority("second"))
        member.begin(session_id="m", turn_id="t", user_message=composite, platform="whatsapp")
        self.assertEqual(member.decide(body, session_id="m", turn_id="t", profile="owner-test", route="head"),
                         (None, "owner_unverified"))  # refused by begin: no "\n\n" boundary before "second"

    def test_owner_words_rule(self):
        words = ["Please answer my question.", "first line\n\nsecond paragraph", "\nleading newline", "x"]
        for own in words:
            shapes = [own, "note\n\n" + own, own + "\n\nrecall", "note\n\n" + own + "\n\nrecall",
                      "note\n\n\n" + own + "\n\n\nrecall", "\n\n" + own + "\n\n", "note\n" + own, own + "\nrecall",
                      "note " + own + "\n\nrecall", "a\n\n" + own + own + "\n\nb", "\n" + own, own + "\n"]
            for content in shapes:
                expected = ("\n\n" + own + "\n\n") in ("\n\n" + content + "\n\n")
                for trust in ("host", "none"):
                    with self.subTest(own=own, content=content, trust=trust):
                        policy = self.make_policy(trust)
                        words_ok = policy._owner_words(content, own)
                        self.assertEqual(words_ok, content == own or (trust == "host" and expected))

    def test_owner_words_is_linear_on_a_long_pasted_document(self):
        # Every owner turn runs _owner_words (begin and check); the digest walk there took ~3 s on this shape.
        words = "\n\n".join(("paragraph %d " % i).ljust(28, "x") for i in range(5000))
        content = "[The user sent a document: notes.txt]\n\n" + words + "\n\n" + "\n\n".join(["recall"] * 25)
        policy = self.make_policy("host")
        started = time.perf_counter()
        self.assertTrue(policy._owner_words(content, words))
        self.assertFalse(policy._owner_words(content, words + "x"))
        self.assertLess(time.perf_counter() - started, 0.25)

    def test_turn_two_after_a_quote_reply_stays_owner(self):
        # Turn 1 is a quote-reply with recall appended; Hermes replays that exact request text as history.
        first = "[Replying to: \"Shall I book the table?\"]\n\nYes, book it for eight"
        sent = first + "\n\n<memory-context>host recall</memory-context>"
        trust = {"turn_context": "host", "history": "verified_owner"}
        one = Policy(authority=FixedAuthority("Yes, book it for eight"), trust=trust)
        one.begin(session_id="q", turn_id="t1", user_message=first, platform="whatsapp")
        body = {"model": "bench", "messages": [{"role": "user", "content": sent}]}
        self.assertEqual(one.decide(body, session_id="q", turn_id="t1", profile="owner-test", route="head")[1],
                         "owner_clean")
        two = Policy(authority=FixedAuthority("And a cab at half seven"), trust=trust)
        two.begin(session_id="q", turn_id="t2", user_message="And a cab at half seven", platform="whatsapp")
        body = {"model": "bench", "messages": [{"role": "user", "content": sent},
                                               {"role": "assistant", "content": "Booked for eight."},
                                               {"role": "user", "content": "And a cab at half seven\n\nrecall"}]}
        self.assertEqual(two.decide(body, session_id="q", turn_id="t2", profile="owner-test", route="head")[1],
                         "owner_clean")


class TestProcessOwnership(unittest.TestCase):
    def test_serving_process(self):
        self.assertTrue(serving_process(None, ["main.py", "cron", "run"]))
        self.assertTrue(serving_process(["gateway", "run"], ["/x/hermes_cli/main.py", "gateway", "run", "--replace"]))
        self.assertFalse(serving_process(["gateway", "run"], ["/x/hermes", "cron", "run", "b85f75f0ed48"]))
        self.assertFalse(serving_process(["gateway", "run"], ["/x/hermes", "-z", "hi"]))
        self.assertFalse(serving_process("gateway", ["x", "gateway", "run"]))


if __name__ == "__main__":
    unittest.main()
