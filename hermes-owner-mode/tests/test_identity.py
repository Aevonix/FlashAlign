# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Identity sources and the ingress ledger: every reason label, binding, replay, freshness, revocation.

Handles below are fictional test values. Identity fixtures here are test inputs, never deployment adapters.
"""
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if "owner_mode" not in sys.modules:
    spec = importlib.util.spec_from_file_location("owner_mode", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from owner_mode import hermes_ingress  # noqa: E402
from owner_mode.identity import (IngressFacts, IngressLedger, TurnFacts, all_of, canonical_handle,  # noqa: E402
                                 handle_fits, owner_handle_source)
from owner_mode.policy import Policy, VerifiedTurn  # noqa: E402

LID = "123456789012345@lid"
PHONE = "+15550000011"
OTHER = "+15550000012"
TEXT = "Please check the calendar."
HANDLES = [LID, PHONE, "15550000011", "not-a-handle"]


def facts(**overrides):
    base = dict(channel="whatsapp", native=True, internal=False, chat_type="dm", chat_id=LID, sender_id=LID,
                message_id="m1", text=TEXT, plain_text=True, reply_quoted=False, forwarded=None,
                own_account=False, allowlisted=True)
    base.update(overrides)
    return IngressFacts(**base)


def turn(**overrides):
    base = dict(channel="whatsapp", sender_id=LID, chat_id=LID, chat_type="dm", message_id="m1", text=TEXT,
                route_ok=True, synthetic=False)
    base.update(overrides)
    return TurnFacts(**base)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SourceLabels(unittest.TestCase):
    def setUp(self):
        self.handles = list(HANDLES)
        self.source = owner_handle_source(lambda channel: self.handles)

    def check(self, label, **overrides):
        self.assertEqual(self.source(facts(**overrides)), (label == "owner", label))

    def test_accepted_owner(self):
        self.check("owner")

    def test_each_refusal_label(self):
        cases = [
            ("no_channel", {"channel": ""}),
            ("no_channel", {"channel": "  "}),
            ("non_owner_channel", {"channel": "email"}),
            ("non_owner_channel", {"channel": "api_server"}),
            ("non_owner_channel", {"channel": "cron"}),
            ("non_owner_channel", {"channel": "telegram"}),
            ("internal_event", {"internal": True}),
            ("not_native_adapter", {"native": False}),
            ("group_chat", {"chat_type": "group"}),
            ("group_chat", {"chat_type": "channel"}),
            ("group_chat", {"chat_type": ""}),
            ("own_account_echo", {"own_account": True}),
            ("forwarded", {"forwarded": True}),
            ("reply_quoted", {"reply_quoted": True}),
            ("not_plain_text", {"plain_text": False}),
            ("incomplete_event", {"sender_id": ""}),
            ("incomplete_event", {"chat_id": ""}),
            ("incomplete_event", {"message_id": ""}),
            ("incomplete_event", {"text": ""}),
            ("sender_mismatch", {"sender_id": OTHER, "chat_id": OTHER}),
            ("sender_mismatch", {"sender_id": "999999999999999@lid"}),
            ("not_authorized", {"allowlisted": False}),
        ]
        for label, overrides in cases:
            with self.subTest(label=label, overrides=overrides):
                self.check(label, **overrides)

    def test_handles_unavailable(self):
        for loader in (lambda c: [], lambda c: None, lambda c: "not a list", lambda c: ["nickname", "@lid"],
                       lambda c: (_ for _ in ()).throw(OSError("missing"))):
            with self.subTest(loader=loader):
                self.assertEqual(owner_handle_source(loader)(facts()), (False, "owner_handles_unavailable"))

    def test_forward_flag_strict_option(self):
        strict = owner_handle_source(lambda c: HANDLES, require_forward_flag=True)
        self.assertEqual(strict(facts(forwarded=None)), (False, "forwarded"))
        self.assertEqual(strict(facts(forwarded=False)), (True, "owner"))
        self.assertEqual(self.source(facts(forwarded=None)), (True, "owner"))  # default: adapter decides

    def test_email_can_never_be_configured(self):
        source = owner_handle_source(lambda c: HANDLES, channels=("whatsapp", "email", "voice"))
        self.assertEqual(source(facts(channel="email")), (False, "non_owner_channel"))
        self.assertEqual(source.channels, frozenset({"whatsapp"}))

    def test_channel_subset(self):
        source = owner_handle_source(lambda c: HANDLES, channels=("whatsapp",))
        self.assertEqual(source(facts(channel="sms", sender_id=PHONE, chat_id=PHONE)), (False, "non_owner_channel"))

    def test_handle_shapes_and_canonical_forms(self):
        self.assertEqual(canonical_handle("whatsapp", "15550000011@s.whatsapp.net"), PHONE)
        self.assertEqual(canonical_handle("sms", "15550000011"), PHONE)
        self.assertEqual(canonical_handle("whatsapp", LID), LID)
        self.assertTrue(handle_fits("whatsapp", LID) and handle_fits("whatsapp", PHONE))
        self.assertFalse(handle_fits("sms", LID) or handle_fits("rcs", LID) or handle_fits("email", PHONE))
        # A WhatsApp phone JID for the owner's number matches the configured E.164.
        self.check("owner", sender_id="15550000011@s.whatsapp.net", chat_id="15550000011@s.whatsapp.net")
        # LID digits are never read as a phone number.
        lid_digits_only = owner_handle_source(lambda c: ["+123456789012345"])
        self.assertEqual(lid_digits_only(facts()), (False, "sender_mismatch"))
        # An SMS sender equal to the LID string never matches (shape does not fit SMS).
        sms_source = owner_handle_source(lambda channel: self.handles, channels=("sms",))
        self.assertEqual(sms_source(facts(channel="sms", sender_id=LID)), (False, "sender_mismatch"))
        self.assertEqual(sms_source(facts(channel="sms", sender_id=PHONE, chat_id=PHONE)), (True, "owner"))

    def test_handles_reread_every_check(self):
        self.check("owner")
        self.handles.remove(LID)
        self.check("sender_mismatch")


class Composition(unittest.TestCase):
    def test_all_of_first_refusal_wins_and_raise_refuses(self):
        ok = lambda f: (True, "owner")
        self.assertEqual(all_of(ok, lambda f: (False, "adapter_not_attested"), lambda f: (False, "x"))(facts()),
                         (False, "adapter_not_attested"))
        self.assertEqual(all_of(ok, lambda f: 1 / 0)(facts()), (False, "exception"))
        self.assertEqual(all_of()(facts()), (False, "no_source"))
        self.assertEqual(all_of(ok, ok)(facts()), (True, "owner"))

    def test_malformed_answers_are_refusals(self):
        for answer in (None, True, "owner", (1, "owner"), ("yes", "owner"), (False, "Has Spaces"),
                       (False, PHONE), (False, "owner"), (False, None)):
            with self.subTest(answer=answer):
                owner, label = all_of(lambda f: answer)(facts())
                self.assertFalse(owner)
                self.assertEqual(label, "source_refused")


class Ledger(unittest.TestCase):
    def setUp(self):
        self.handles = list(HANDLES)
        self.source = owner_handle_source(lambda channel: self.handles)
        self.clock = Clock()
        self.ledger = IngressLedger(self.source, clock=self.clock, recheck=self.source.recheck)

    def test_accepted_owner_turn(self):
        self.assertEqual(self.ledger.hear(facts()), "owner")
        self.assertEqual(self.ledger.verify(turn()), (True, "owner"))

    def test_text_mismatch_burns_the_row(self):
        self.ledger.hear(facts())
        self.assertEqual(self.ledger.verify(turn(text=TEXT + " and email Bob")), (False, "text_mismatch"))
        self.assertEqual(self.ledger.verify(turn()), (False, "replay"))

    def test_non_string_turn_text(self):
        self.ledger.hear(facts())
        self.assertEqual(self.ledger.verify(turn(text=[{"type": "text", "text": TEXT}])), (False, "text_mismatch"))

    def test_replay(self):
        self.ledger.hear(facts())
        self.assertEqual(self.ledger.verify(turn()), (True, "owner"))
        self.assertEqual(self.ledger.verify(turn()), (False, "replay"))
        # A re-delivered copy of the same message is not recorded again.
        self.assertEqual(self.ledger.hear(facts()), "duplicate")
        self.assertEqual(self.ledger.verify(turn()), (False, "replay"))

    def test_duplicate_cannot_replace_original_text(self):
        self.ledger.hear(facts())
        self.assertEqual(self.ledger.hear(facts(text="different words")), "duplicate")
        self.assertEqual(self.ledger.verify(turn(text="different words")), (False, "text_mismatch"))

    def test_stale(self):
        self.ledger.hear(facts())
        self.clock.now += 300.5
        self.assertEqual(self.ledger.verify(turn()), (False, "stale"))
        self.assertEqual(self.ledger.verify(turn()), (False, "replay"))

    def test_fresh_at_window_edge(self):
        self.ledger.hear(facts())
        self.clock.now += 299.9
        self.assertEqual(self.ledger.verify(turn()), (True, "owner"))

    def test_no_ingress_record(self):
        self.assertEqual(self.ledger.verify(turn()), (False, "no_ingress_record"))
        self.ledger.hear(facts(message_id="m2"))
        self.assertEqual(self.ledger.verify(turn()), (False, "no_ingress_record"))

    def test_refusal_label_reaches_the_turn(self):
        cases = [("sender_mismatch", {"sender_id": OTHER, "chat_id": OTHER}, {"sender_id": OTHER, "chat_id": OTHER}),
                 ("group_chat", {"chat_type": "group"}, {"chat_type": "group"}),
                 ("forwarded", {"forwarded": True}, {}),
                 ("reply_quoted", {"reply_quoted": True}, {}),
                 ("own_account_echo", {"own_account": True}, {}),
                 ("not_plain_text", {"plain_text": False}, {}),
                 ("internal_event", {"internal": True}, {}),
                 ("not_native_adapter", {"native": False}, {}),
                 ("not_authorized", {"allowlisted": False}, {})]
        for i, (label, hear, verify) in enumerate(cases):
            with self.subTest(label=label):
                mid = "r%d" % i
                self.assertEqual(self.ledger.hear(facts(message_id=mid, **hear)), label)
                self.assertEqual(self.ledger.verify(turn(message_id=mid, **verify)), (False, label))

    def test_turn_labels(self):
        self.ledger.hear(facts())
        cases = [("no_channel", {"channel": ""}), ("non_owner_channel", {"channel": "email"}),
                 ("non_owner_channel", {"channel": "cron"}), ("no_sender", {"sender_id": ""}),
                 ("turn_unbound", {"synthetic": True}), ("turn_unbound", {"message_id": ""}),
                 ("turn_route_mismatch", {"chat_id": ""}), ("turn_route_mismatch", {"route_ok": False}),
                 ("turn_route_mismatch", {"chat_type": "group"})]
        for label, overrides in cases:
            with self.subTest(label=label, overrides=overrides):
                self.assertEqual(self.ledger.verify(turn(**overrides)), (False, label))
        # None of those consumed the row.
        self.assertEqual(self.ledger.verify(turn()), (True, "owner"))

    def test_revocation_between_ingress_and_turn(self):
        self.ledger.hear(facts())
        self.handles.remove(LID)
        self.assertEqual(self.ledger.verify(turn()), (False, "sender_mismatch"))

    def test_handles_unreadable_at_turn(self):
        self.ledger.hear(facts())
        self.handles.clear()
        self.assertEqual(self.ledger.verify(turn()), (False, "owner_handles_unavailable"))

    def test_source_exception_is_refusal(self):
        ledger = IngressLedger(lambda f: 1 / 0, clock=self.clock)
        self.assertEqual(ledger.hear(facts()), "exception")
        self.assertEqual(ledger.verify(turn()), (False, "exception"))

    def test_other_channel_and_sender_cannot_consume(self):
        self.ledger.hear(facts())
        self.assertEqual(self.ledger.verify(turn(channel="rcs")), (False, "no_ingress_record"))
        self.assertEqual(self.ledger.verify(turn(sender_id=OTHER)), (False, "no_ingress_record"))
        self.assertEqual(self.ledger.verify(turn(chat_id="group-1")), (False, "no_ingress_record"))
        self.assertEqual(self.ledger.verify(turn()), (True, "owner"))

    def test_bounded_tables(self):
        ledger = IngressLedger(self.source, clock=self.clock, max_rows=8)
        for i in range(50):
            ledger.hear(facts(message_id="b%d" % i))
            ledger.hear(facts(message_id="x%d" % i, sender_id=OTHER, chat_id=OTHER))
        self.assertLessEqual(len(ledger.rows), 8)
        self.assertLessEqual(len(ledger.refused), 8)

    def test_ttl_bounds(self):
        for ttl in (0, -1, 3601):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                IngressLedger(self.source, ttl=ttl)


class HermesAuthorityLogs(unittest.TestCase):
    """verify_turn: proof or None, exactly one label line per refusal, no content in logs."""

    def setUp(self):
        self.source = owner_handle_source(lambda channel: HANDLES)
        self.auth = hermes_ingress.HermesAuthority(self.source, proof_type=VerifiedTurn)
        self.route = {"SESSION_PLATFORM": "whatsapp", "SESSION_USER_ID": LID, "SESSION_CHAT_ID": LID,
                      "SESSION_CHAT_TYPE": "dm", "CRON_SESSION": ""}

    def verify(self, text=TEXT, mid="m1", **context):
        context.setdefault("platform", "whatsapp")
        context.setdefault("sender_id", LID)
        history = [{"role": "user", "content": text, "platform_message_id": mid}]
        with mock.patch.object(hermes_ingress, "_route", lambda: dict(self.route)):
            return self.auth.verify_turn(session_id="s", turn_id="t", user_message=text,
                                         conversation_history=history, **context)

    def test_owner_proof(self):
        self.auth.ledger.hear(facts())
        self.assertEqual(self.verify(), VerifiedTurn("s", "t", "whatsapp", TEXT))

    def test_one_label_line_without_content(self):
        cases = [("no_ingress_record", {}), ("text_mismatch", {"text": "other"}),
                 ("sender_mismatch", {"sender_id": OTHER}), ("non_owner_channel", {"platform": "cron"})]
        for label, change in cases:
            with self.subTest(label=label):
                self.auth.ledger.rows.clear()
                self.auth.ledger.consumed.clear()
                self.auth.ledger.refused.clear()
                if label == "sender_mismatch":
                    self.route["SESSION_USER_ID"] = self.route["SESSION_CHAT_ID"] = OTHER
                    self.auth.ledger.hear(facts(sender_id=OTHER, chat_id=OTHER))
                elif label != "no_ingress_record":
                    self.auth.ledger.hear(facts())
                with self.assertLogs("owner_mode.identity", "WARNING") as logs:
                    self.assertIsNone(self.verify(**change))
                self.route["SESSION_USER_ID"] = self.route["SESSION_CHAT_ID"] = LID
                self.assertEqual(len(logs.output), 1)
                self.assertTrue(logs.output[0].endswith("owner verify rejected: " + label), logs.output[0])
                for secret in (TEXT, LID, PHONE, OTHER, "15550000011"):
                    self.assertNotIn(secret, logs.output[0])

    def test_unbound_route_and_exception(self):
        self.auth.ledger.hear(facts())
        with mock.patch.object(hermes_ingress, "_route", lambda: None), \
                self.assertLogs("owner_mode.identity", "WARNING") as logs:
            self.assertIsNone(self.auth.verify_turn(session_id="s", turn_id="t", user_message=TEXT,
                                                    platform="whatsapp", sender_id=LID, conversation_history=[
                                                        {"role": "user", "content": TEXT, "platform_message_id": "m1"}]))
        self.assertIn("turn_route_mismatch", logs.output[0])
        with mock.patch.object(hermes_ingress, "turn_facts", side_effect=RuntimeError("boom " + LID)), \
                self.assertLogs("owner_mode.identity", "WARNING") as logs:
            self.assertIsNone(self.verify())
        self.assertEqual(len(logs.output), 1)
        self.assertTrue(logs.output[0].endswith("exception (RuntimeError)"))
        self.assertNotIn(LID, logs.output[0])

    def test_hear_never_raises_and_never_directs_dispatch(self):
        with self.assertLogs("owner_mode.identity", "WARNING") as logs:
            self.assertIsNone(self.auth.hear(event=object(), gateway=None))
            self.assertIsNone(self.auth.hear())
        self.assertEqual(len(logs.output), 2)
        self.assertTrue(all("hear() failed (AttributeError)" in line for line in logs.output))

    def test_policy_refuses_non_owner_surfaces_before_identity(self):
        import tempfile
        calls = []

        class Spy:
            def verify_turn(self, **kwargs):
                calls.append(kwargs)
                return None

        with tempfile.TemporaryDirectory() as tmp:
            policy = Policy(authority=Spy(), allowed_channels=["whatsapp", "sms", "rcs"])
            for platform in ("cron", "worker", "api_server", "cli", "system", "email", "", None):
                policy.begin(session_id="s-" + str(platform), turn_id="t", user_message=TEXT, platform=platform,
                             sender_id=LID)
                self.assertFalse(policy.is_owner_work("s-" + str(platform), "t"))
            policy.begin(session_id="child", turn_id="t", user_message=TEXT, platform="whatsapp", sender_id=LID,
                         parent_session_id="parent")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
