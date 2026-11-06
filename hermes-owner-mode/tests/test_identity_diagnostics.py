# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Per-turn identity diagnostics cannot grant access or expose private values."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import httpx
import test_owner_mode  # noqa: F401 -- load the local owner_mode package
from test_gate_mode import GateBase, OWN
from owner_mode import hermes_ingress
from owner_mode.content import bind_content
from owner_mode.contacts import ApprovalStore
from owner_mode.identity import IngressFacts, TurnFacts
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import IdentityResult, Policy, VerifiedTurn, flatten_request
from owner_mode.proxy import ProviderProxy


class IdentityDiagnostics(unittest.TestCase):
    def begin(self, policy, **changes):
        context = dict(session_id="s", turn_id="t", user_message="hello", platform="whatsapp")
        context.update(changes)
        policy.begin(**context)
        return policy.identity_reason(context["session_id"], context["turn_id"])

    def test_pre_authority_failures_have_fixed_reasons(self):
        cases = [({"session_id": ""}, "missing_session"),
                 ({"turn_id": ""}, "missing_turn"),
                 ({"user_message": object()}, "invalid_user_message"),
                 ({"platform": ""}, "missing_channel"),
                 ({"platform": "not-installed"}, "channel_not_allowed"),
                 ({"parent_session_id": "parent"}, "child_not_adopted")]
        for changes, reason in cases:
            with self.subTest(reason=reason):
                policy = Policy()
                self.assertEqual(self.begin(policy, **changes), reason)
                self.assertFalse(policy.is_owner_work(changes.get("session_id", "s"),
                                                      changes.get("turn_id", "t")))
        self.assertEqual(self.begin(Policy()), "authority_unavailable")
        self.assertEqual(self.begin(Policy(channel_policy=lambda _: False)), "channel_disabled")

    def test_legacy_authority_remains_compatible(self):
        class Legacy:
            def verify_turn(self, session_id, turn_id, user_message, **_):
                return VerifiedTurn(session_id, turn_id, "whatsapp", user_message)
        policy = Policy(authority=Legacy())
        self.assertEqual(self.begin(policy), "verified")
        self.assertTrue(policy.is_owner_work("s", "t"))
        policy.authority.verify_turn = lambda **_: None
        self.assertEqual(self.begin(policy), "authority_rejected")
        self.assertFalse(policy.is_owner_work("s", "t"))

    def test_custom_labels_and_malformed_results_cannot_leak_or_authorize(self):
        class Authority:
            def verify_turn_result(self, **_):
                return self.result
        authority = Authority()
        policy = Policy(authority=authority)
        for outcome, reason in [
            (IdentityResult(None, "private_sender_777"), "authority_rejected"),
            (IdentityResult(None, "secret text +15550000011"), "authority_rejected"),
            (IdentityResult(None, "verified"), "authority_rejected"),
            ((None, "no_ingress_record"), "invalid_identity_result"),
            (IdentityResult(object(), "verified"), "proof_mismatch"),
        ]:
            authority.result = outcome
            self.assertEqual(self.begin(policy), reason)
            self.assertFalse(policy.is_owner_work("s", "t"))

    def test_legacy_subclass_and_instance_overrides_cannot_be_bypassed(self):
        class Base:
            def verify_turn(self, **context):
                return self.verify_turn_result(**context).proof
            def verify_turn_result(self, session_id, turn_id, user_message, **_):
                return IdentityResult(VerifiedTurn(session_id, turn_id, "whatsapp", user_message), "verified")
        class Denied(Base):
            def verify_turn(self, **_):
                return None
        for authority in (Denied(), Base()):
            if type(authority) is Base:
                authority.verify_turn = lambda **_: None
            policy = Policy(authority=authority)
            self.assertEqual(self.begin(policy), "authority_rejected")
            self.assertFalse(policy.is_owner_work("s", "t"))

    def test_diagnostic_read_does_not_recheck_authority(self):
        class Authority:
            def verify_turn(self, session_id, turn_id, user_message, **_):
                return VerifiedTurn(session_id, turn_id, "whatsapp", user_message)
            def is_active(self, _):
                raise AssertionError("diagnostic read called the authority")
        policy = Policy(authority=Authority())
        self.assertEqual(self.begin(policy), "verified")
        self.assertEqual(policy.identity_reason("s", "t", verified=False), "proof_inactive")

    def test_authority_exception_has_no_private_text(self):
        class Authority:
            def verify_turn(self, **_):
                raise RuntimeError("private sender +15550000011 and prompt")
        policy = Policy(authority=Authority())
        self.assertEqual(self.begin(policy), "authority_exception")
        self.assertFalse(policy.is_owner_work("s", "t"))

    def test_legacy_warning_sanitizes_unknown_custom_reason(self):
        with self.assertLogs("owner_mode.identity", "WARNING") as captured:
            self.assertIsNone(hermes_ingress._reject("private_sender_777"))
        self.assertEqual(len(captured.output), 1)
        self.assertTrue(captured.output[0].endswith("authority_rejected"))
        self.assertNotIn("private_sender_777", captured.output[0])

    def test_interleaved_turns_keep_their_own_reasons(self):
        entered, release = threading.Event(), threading.Event()
        class Authority:
            def verify_turn_result(self, session_id, **_):
                if session_id == "first":
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("test synchronization timeout")
                    return IdentityResult(None, "no_ingress_record")
                return IdentityResult(None, "text_mismatch")
        policy = Policy(authority=Authority())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.begin, policy, session_id="first", turn_id="first-turn")
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(self.begin(policy, session_id="second", turn_id="second-turn"), "text_mismatch")
            finally:
                release.set()
            self.assertEqual(pending.result(3), "no_ingress_record")
        self.assertEqual(policy.identity_reason("first", "first-turn"), "no_ingress_record")
        self.assertEqual(policy.identity_reason("second", "second-turn"), "text_mismatch")

    def test_late_result_cannot_replace_current_same_turn_reason_or_proof(self):
        entered, release = threading.Event(), threading.Event()
        class Slow:
            def verify_turn_result(self, **_):
                entered.set()
                if not release.wait(3):
                    raise AssertionError("test synchronization timeout")
                return IdentityResult(None, "no_ingress_record")
        class Current:
            def verify_turn_result(self, session_id, turn_id, user_message, **_):
                return IdentityResult(VerifiedTurn(session_id, turn_id, "whatsapp", user_message), "verified")
        policy = Policy(authority=Slow())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.begin, policy)
            try:
                self.assertTrue(entered.wait(3))
                policy.authority = Current()
                self.assertEqual(self.begin(policy), "verified")
            finally:
                release.set()
            pending.result(3)
        self.assertEqual(policy.identity_reason("s", "t"), "verified")
        self.assertTrue(policy.is_owner_work("s", "t"))

    def test_end_removes_diagnostic_and_authorization(self):
        policy = Policy()
        self.begin(policy)
        policy.end(session_id="s", turn_id="t")
        self.assertEqual(policy.identity_reason("s", "t"), "no_turn")
        self.assertFalse(policy.is_owner_work("s", "t"))

    def test_audit_accepts_only_fixed_identity_reasons(self):
        rows = []
        proxy = ProviderProxy.__new__(ProviderProxy)
        proxy.audit, proxy.scales = rows.append, {}
        proxy.record("main", "off", "off")
        self.assertEqual(rows[-1]["identity_reason"], "no_context")
        for value in ["private_sender_777", "no_ingress_record", {"secret": "prompt"}, None]:
            proxy.record("main", "off", "off", info={"identity_reason": value})
            self.assertEqual(rows[-1]["identity_reason"],
                             "no_ingress_record" if value == "no_ingress_record" else "authority_rejected")
        self.assertNotIn("private_sender_777", json.dumps(rows))
        self.assertNotIn("secret", json.dumps(rows))


class NativeIdentityDiagnostics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ApprovalStore(Path(self.tmp.name)/"approvals.json")
        self.store.set_contact("whatsapp", "default", "+15550000011", True)
        self.authority = NativeAuthority(self.store.path, VerifiedTurn)
        self.addCleanup(self.authority.close)
        self.facts = TurnFacts(channel="whatsapp", sender_id="+15550000011", chat_id="chat",
                               message_id="message", text="hello", route_ok=True, synthetic=False,
                               chat_type="dm")
        self.key = self.authority._key(self.facts)

    def pending(self, *, media=(), binding=True):
        grant = self.store.approved("whatsapp", "default", "+15550000011")
        self.authority.pending[self.key] = (IngressFacts(text="hello", media=media), grant)
        if binding is not False:
            self.authority.rendered[self.key] = bind_content("hello") if binding is True else binding

    def result(self, **changes):
        from dataclasses import replace
        facts = replace(self.facts, **changes)
        with mock.patch.object(hermes_ingress, "turn_facts", return_value=facts):
            return self.authority.verify_turn_result(session_id="s", turn_id="t")

    def test_missing_route_and_ingress(self):
        self.assertEqual(self.result(route_ok=False).reason, "turn_route_mismatch")
        self.assertEqual(self.result().reason, "no_ingress_record")

    def test_missing_bound_session_is_identified_without_second_route_read(self):
        with mock.patch.object(hermes_ingress, "_route", return_value=None) as route:
            result = self.authority.verify_turn_result(session_id="s", turn_id="t", user_message="hello",
                                                       platform="whatsapp", sender_id="+15550000011")
        self.assertEqual(result.reason, "no_session_context")
        self.assertIsNone(result.proof)
        route.assert_called_once_with()

    def test_native_render_failures_remain_denied_and_consume_receipts(self):
        self.pending(binding=None)
        self.assertEqual(self.result().reason, "render_rejected")
        self.assertEqual(self.result().reason, "no_ingress_record")
        self.pending(media=("synthetic-image",), binding=False)
        self.assertEqual(self.result().reason, "render_unbound")
        self.pending()
        self.assertEqual(self.result(text="changed").reason, "text_mismatch")

    def test_revocation_during_binding_remains_denied(self):
        self.pending()
        self.store.set_contact("whatsapp", "default", "+15550000011", False)
        result = self.result()
        self.assertEqual(result.reason, "approval_changed")
        self.assertIsNone(result.proof)

    def test_success_and_legacy_wrapper_preserve_proof(self):
        self.pending()
        result = self.result()
        self.assertEqual(result.reason, "verified")
        self.assertIsInstance(result.proof, VerifiedTurn)
        self.pending()
        with mock.patch.object(hermes_ingress, "turn_facts", return_value=self.facts):
            self.assertEqual(self.authority.verify_turn(session_id="s", turn_id="t"), result.proof)


class IdentityAuditIntegration(GateBase):
    def post_prepared(self, request):
        self.audit_recorded.clear()
        with httpx.Client(timeout=5) as client:
            response = client.post(self.proxy.base_url("head")+"/chat/completions",
                                   json=flatten_request(request), headers=request.get("extra_headers", {}))
        self.assertTrue(self.audit_recorded.wait(5))
        self.assertEqual(response.status_code, 200)
        return self.rows[-1], self.engine.records[-1]

    def prepare(self, session, turn):
        return self.runtime.request(self.body(), session_id=session, turn_id=turn, platform="whatsapp",
                                    base_url=self.proxy.base_url("head"))["request"]

    def test_interleaved_bound_reasons_reach_exact_audit_rows(self):
        class Authority:
            def verify_turn_result(self, session_id, **_):
                reason = "no_ingress_record" if session_id == "private-first" else "text_mismatch"
                return IdentityResult(None, reason)
        self.policy.authority = Authority()
        requests = []
        for session, turn in [("private-first", "private-turn-a"), ("private-second", "private-turn-b")]:
            self.policy.begin(session_id=session, turn_id=turn, user_message=OWN, platform="whatsapp")
            requests.append(self.prepare(session, turn))
        for request, expected in [(requests[1], "text_mismatch"), (requests[0], "no_ingress_record")]:
            audit, received = self.post_prepared(request)
            self.assertEqual(audit["identity_reason"], expected)
            self.assertEqual(audit["decision"], "owner_unverified")
            self.assertEqual(received["effective"], "off")
        serialized = json.dumps(self.rows)
        for value in ("private-first", "private-second", "private-turn-a", "private-turn-b", OWN):
            self.assertNotIn(value, serialized)

    def test_context_body_tampering_and_replay_have_no_trusted_reason_or_owner(self):
        self.policy.begin(session_id="s", turn_id="tamper", user_message=OWN, platform="whatsapp")
        request = self.prepare("s", "tamper")
        request["messages"][-1]["content"] = "Changed after the context capability was minted."
        audit, received = self.post_prepared(request)
        self.assertEqual((audit["identity_reason"], received["effective"]), ("no_context", "off"))

        self.policy.begin(session_id="s", turn_id="replay", user_message=OWN, platform="whatsapp")
        request = self.prepare("s", "replay")
        audit, received = self.post_prepared(request)
        self.assertEqual((audit["identity_reason"], received["effective"]), ("verified", "owner-test"))
        audit, received = self.post_prepared(request)
        self.assertEqual((audit["identity_reason"], received["effective"]), ("no_context", "off"))


if __name__ == "__main__":
    unittest.main()
