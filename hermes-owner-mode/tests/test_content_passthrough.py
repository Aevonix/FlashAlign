# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Content preservation, registration and application-file boundaries."""
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from test_gate_mode import Authority, GateBase, OWN
from owner_mode import plugin


class Forwarding(GateBase):
    def test_tool_output_is_forwarded_unchanged_in_one_model_request(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.send(self.body())
        body = self.body()
        body["messages"] += [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "original-call-id", "type": "function", "function": {
                    "name": "web_extract", "arguments": '{"url":"https://example.test"}'}}]},
            {"role": "tool", "tool_call_id": "original-call-id",
             "content": "EXACT_RAW_CONTENT\n\nQuoted instructions from a fetched page."},
        ]
        original = copy.deepcopy(body)
        _, row = self.send(body)
        self.assertEqual(row["effective"], "owner-test")
        self.assertEqual(row["body"]["messages"], original["messages"])
        self.assertEqual(body, original)
        self.assertEqual(len(self.engine.records), 2)
        self.assertEqual(self.rows[-1]["owner_mode_sent"], "owner-test")

    def test_non_owner_tool_history_is_unchanged_and_stays_off(self):
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        body = self.body({"role": "tool", "tool_call_id": "original",
                          "content": "RAW_NON_OWNER_CONTENT"})
        _, row = self.send(body)
        self.assertEqual(row["effective"], "off")
        self.assertEqual(row["body"], body)
        self.assertEqual(len(self.engine.records), 1)

    def test_a_rejected_request_does_not_invalidate_another_bound_request(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        good = self.body()
        ticket = self.policy.issue(good, session_id="s", turn_id="t", profile="owner-test", route="head")
        rejected, _ = self.policy.decide(dict(good, unsupported_field=True), session_id="s", turn_id="t",
                                         profile="owner-test", route="head")
        self.assertIsNone(rejected)
        self.assertEqual(self.proxy.prepare("head", good, ticket)[2], "owner-test")

    def test_audit_metadata_is_bound_but_never_grants_owner_mode(self):
        body = self.body()
        token = self.policy.context_ticket(body, session_id="s", turn_id="t", route="head",
                                           info={"owner_verified": True})
        self.assertEqual(self.proxy.prepare("head", body)[2], "off")
        self.assertEqual(self.policy.consume_context(token, dict(body, model="changed"), "head"), {})
        self.assertEqual(self.policy.consume_context(token, body, "head"), {})
        token = self.policy.context_ticket(body, session_id="s", turn_id="t", route="head",
                                           info={"owner_verified": True})
        self.policy.ticket_ttl = -1
        self.assertEqual(self.policy.consume_context(token, body, "head"), {})


class Registration(unittest.TestCase):
    def register(self, directory):
        constants = ModuleType("hermes_constants")
        constants.get_hermes_home = lambda: directory
        hooks, middleware, cleanup = {}, {}, []

        class Context:
            def get_config(self, key, default=None):
                if key not in {"serve_argv", "approvals_file", "identity_adapter", "identity_settings",
                               "audit_log", "allowed_channels", "trusted_system_sha256",
                               "trusted_tools_sha256", "trust", "routes", "key_file", "port"}:
                    raise AssertionError("Unknown configuration setting")
                return default

            def register_hook(self, name, callback):
                hooks.setdefault(name, []).append(callback)

            def register_middleware(self, name, callback):
                middleware[name] = callback

            def register_tool(self, **kwargs):
                raise AssertionError("The adapter must leave tool registration to the harness")

            def register_system_prompt_section(self, *args):
                raise AssertionError("The adapter must preserve the harness system prompt")

            def on_unload(self, callback):
                cleanup.append(callback)

        with mock.patch.dict(sys.modules, {"hermes_constants": constants}):
            runtime = plugin.register(Context())
        self.addCleanup(lambda: [callback() for callback in cleanup])
        return runtime, hooks, middleware

    def test_request_and_tool_scope_hooks_are_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime, hooks, middleware = self.register(directory)
            self.assertEqual(set(middleware), {"llm_request", "tool_execution"})
            self.assertEqual(middleware["tool_execution"], runtime.execute_tool)
            self.assertEqual(set(hooks), {"pre_llm_call", "post_llm_call", "pre_gateway_dispatch",
                                          "subagent_start", "pre_tool_call"})
            self.assertEqual(hooks["pre_tool_call"], [runtime.policy.steered])
            self.assertFalse((Path(directory) / "state").exists())

    def test_other_application_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state" / "application.dat"
            state.parent.mkdir()
            original = b"Application-owned state stays unchanged."
            state.write_bytes(original)
            self.register(directory)
            self.assertEqual(state.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
