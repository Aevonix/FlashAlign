# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

import httpx

ROOT = Path(__file__).resolve().parents[1]
if "owner_mode" not in sys.modules:
    spec = importlib.util.spec_from_file_location("owner_mode", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from owner_mode.auth import sign, read_key
from owner_mode.policy import Policy, VerifiedTurn, canonical, digest, flatten_request, strict_json
from owner_mode.proxy import ProviderProxy
from owner_mode.plugin import Runtime
from owner_mode import plugin as plugin_module
from support import Engine


class TestAuthority:
    """A trusted-host fixture, not an identity implementation shipped by the plugin."""
    def __init__(self, channel="whatsapp", owner=True):
        self.channel, self.owner = channel, owner

    def verify_turn(self, session_id, turn_id, user_message, **_):
        return VerifiedTurn(session_id, turn_id, self.channel, user_message) if self.owner else None


class Matrix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.key = os.urandom(32)
        self.keyfile = self.root / "key"
        self.keyfile.write_bytes(self.key)
        self.keyfile.chmod(0o600)
        self.system = {"role": "system", "content": "Audited static instruction."}
        self.body = {"model": "bench", "messages": [self.system, {"role": "user", "content": "Hello"}], "owner_mode": "off"}
        self.policy = Policy(trusted_system_hashes=[digest(self.system)], authority=TestAuthority())
        self.engine = Engine(self.key)
        self.proxy = ProviderProxy(self.policy, {"main": {"upstream": self.engine.base_url, "profile": "owner-test"}}, self.keyfile)
        self.proxy.start()
        self.runtime = Runtime(self.policy, self.proxy)
        self.policy.begin(session_id="s", turn_id="t", user_message="Hello", platform=self.policy.authority.channel)


    def tearDown(self):
        self.proxy.close()
        self.engine.close()
        self.tmp.cleanup()


    def send(self, body=None, *, sid="s", tid="t", middleware=True, headers=None):
        body = body or self.body
        request = self.runtime.request(body, session_id=sid, turn_id=tid, base_url=self.proxy.base_url("main"))["request"] \
            if middleware else body
        with httpx.Client(timeout=3) as client:
            response = client.post(self.proxy.base_url("main") + "/chat/completions", json=flatten_request(request),
                                   headers=headers or request.get("extra_headers", {}))
            response.raise_for_status()
        return self.engine.records[-1]


    def test_missing_hook_plugin_and_forged_body_fail_closed(self):
        self.assertEqual(self.send(dict(self.body, owner_mode="owner-test"), middleware=False)["effective"], "off")
        self.assertEqual(self.send(sid="not-observed", tid="unobserved-turn")["effective"], "off")
        self.assertEqual(self.send(headers={"X-Owner-Mode-Ticket": "forged", "X-Owner-Mode-Auth": "v1:forged"})["effective"], "off")


    def test_highest_template_efforts_through_runtime_and_http_proxy(self):
        for effort in ("xhigh", "max"):
            for approved in (True, False):
                with self.subTest(effort=effort, approved=approved):
                    sid = f"template-{effort}-{approved}"
                    self.policy.authority = TestAuthority(owner=approved)
                    self.runtime.begin(session_id=sid, turn_id="t", user_message="Hello", platform="whatsapp")
                    body = dict(self.body, reasoning_effort=effort,
                                extra_body={"chat_template_kwargs": {"reasoning_effort": effort}})
                    row = self.send(body, sid=sid)
                    self.assertEqual(row["effective"], "owner-test" if approved else "off")
                    self.assertEqual(row["auth_reason"], "valid" if approved else "off")
                    self.assertEqual(row["body"]["chat_template_kwargs"]["reasoning_effort"], effort)


    def test_ticket_mutation_replay_and_session_isolation(self):
        issue = lambda: self.policy.issue(self.body, session_id="s", turn_id="t", profile="owner-test", route="main")
        ticket = issue()
        self.assertEqual(self.proxy.prepare("main", self.body, ticket)[2], "owner-test")
        self.assertEqual(self.proxy.prepare("main", self.body, ticket)[2], "off")
        ticket = issue()
        self.assertEqual(self.proxy.prepare("main", dict(self.body, model="other"), ticket)[2], "off")
        self.assertEqual(self.send(sid="other-session", tid="other-turn")["effective"], "off")


    def test_missing_bad_and_insecure_keys(self):
        for raw, mode in [(b"short", 0o600), (self.key, 0o644)]:
            with self.subTest(mode=mode, length=len(raw)):
                self.keyfile.write_bytes(raw)
                self.keyfile.chmod(mode)
                self.assertEqual(self.send()["effective"], "off")
        self.keyfile.unlink()
        self.assertEqual(self.send()["effective"], "off")


    def test_exception_timeout_unknown_profile(self):
        self.policy.authority.verify_turn = lambda **_: (_ for _ in ()).throw(TimeoutError())
        self.policy.begin(session_id="s", turn_id="t", user_message="Hello", platform=self.policy.authority.channel)
        self.assertEqual(self.send()["effective"], "off")
        self.policy.authority = TestAuthority()
        self.policy.begin(session_id="clean", turn_id="t", user_message="Hello", platform=self.policy.authority.channel)
        token = self.policy.issue(self.body, session_id="clean", turn_id="t", profile="unconfigured", route="main")
        self.assertEqual(self.proxy.prepare("main", self.body, token)[2], "off")


    def test_ticket_ttl_and_concurrent_one_use(self):
        token = self.policy.issue(self.body, session_id="s", turn_id="t", profile="owner-test", route="main")
        self.policy.ticket_ttl = -1
        self.assertEqual(self.proxy.prepare("main", self.body, token)[2], "off")
        self.policy.ticket_ttl = 5
        token = self.policy.issue(self.body, session_id="s", turn_id="t", profile="owner-test", route="main")
        with ThreadPoolExecutor(max_workers=8) as pool:
            modes = list(pool.map(lambda _: self.proxy.prepare("main", self.body, token)[2], range(16)))
        self.assertEqual(modes.count("owner-test"), 1)


    def test_unsigned_invalid_expired_replayed_engine_requests(self):
        body = canonical(dict(self.body, owner_mode="owner-test"))
        good = sign(self.key, body)
        for header, expected in [("", "off"), (sign(os.urandom(32), body), "off"),
                                 # Leave enough margin for integer timestamps and earlier HTTP cases.
                                 (sign(self.key, body, now=time.time()-120), "off"),
                                 (sign(self.key, body, now=time.time()+120), "off"),
                                 (good, "owner-test"), (good, "off")]:
            with httpx.Client() as client:
                response = client.post(self.engine.base_url + "/chat/completions", content=body,
                                       headers={"X-Owner-Mode-Auth": header})
            self.assertEqual(response.headers["X-Owner-Mode-Effective"], expected)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "Bench response. Effective=" + expected)


    def test_identical_prefix_both_request_orders(self):
        # Protocol isolation only: this mock has no model/KV cache.
        cold = self.send(middleware=False)
        for modes in ((True, False), (False, True)):
            for owner in modes:
                sid = "order-" + str(modes) + str(owner)
                self.policy.authority = TestAuthority(owner=owner)
                self.policy.begin(session_id=sid, turn_id="t", user_message="Hello", platform=self.policy.authority.channel)
                row = self.send(sid=sid)
                self.assertEqual(row["effective"], "owner-test" if owner else cold["effective"])


    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"owner_mode":"off","owner_mode":"owner-test"}')


class StartupDiagnostics(unittest.TestCase):
    def test_authority_load_failure_reports_only_fixed_label_and_exception_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = root / "key"
            key.write_bytes(b"k" * 32)
            key.chmod(0o600)
            constants = ModuleType("hermes_constants")
            constants.get_hermes_home = lambda: root
            values = {"identity_adapter": "/private/synthetic-adapter.py", "key_file": str(key),
                      "audit_log": str(root / "audit.jsonl"), "port": 0}
            unload = []

            class Context:
                def get_config(self, name, default=None):
                    return values.get(name, default)

                def register_hook(self, *_):
                    pass

                def register_middleware(self, *_):
                    pass

                def on_unload(self, callback):
                    unload.append(callback)

            secret = "synthetic-secret-value /private/credential-path"
            try:
                with mock.patch.dict(sys.modules, {"hermes_constants": constants}), \
                        mock.patch.object(plugin_module, "load_authority", side_effect=RuntimeError(secret)), \
                        self.assertLogs("owner_mode.plugin", level="WARNING") as captured:
                    runtime = plugin_module.register(Context())
                self.assertEqual([record.getMessage() for record in captured.records],
                                 ["FlashAlign identity authority unavailable (RuntimeError)"])
                self.assertNotIn(secret, "\n".join(captured.output))
                self.assertTrue(all(record.exc_info is None for record in captured.records))
                self.assertIsNone(runtime.policy.authority)
                runtime.begin(session_id="synthetic-session", turn_id="synthetic-turn",
                              user_message="Hello", platform="whatsapp")
                self.assertFalse(runtime.policy.is_owner_work("synthetic-session", "synthetic-turn"))
            finally:
                while unload:
                    unload.pop()()
