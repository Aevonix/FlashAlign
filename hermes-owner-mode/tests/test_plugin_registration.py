# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Process ownership survives duplicate module loads without sharing new grants."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
import unittest
from unittest import mock

import test_owner_mode
from owner_mode import plugin
from owner_mode.policy import VerifiedTurn


class Authority:
    def __init__(self):
        self.closed = 0

    def verify_turn(self, *, session_id, turn_id, user_message, **_):
        return VerifiedTurn(session_id, turn_id, "whatsapp", user_message)

    def close(self):
        self.closed += 1


class Proxy:
    instances = []
    fail_start = False
    fail_close = False

    def __init__(self, policy, routes, key_file, **kwargs):
        self.policy, self.routes = policy, routes
        self.started = False
        self.closed = 0
        self.instances.append(self)

    def start(self):
        if self.fail_start:
            raise RuntimeError("fixture startup failure")
        self.started = True

    def prepare(self, *args, **kwargs):
        return b"fixture", {}, "owner-test"

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("fixture cleanup failure")
        self.started = False


class Context:
    def __init__(self, root, **config):
        self.config = {"identity_adapter": "synthetic-authority.py",
                       "key_file": str(root / "key"), **config}
        self.config_reads = 0
        self.hooks, self.middleware, self.unload = {}, {}, []

    def get_config(self, name, default=None):
        self.config_reads += 1
        return self.config.get(name, default)

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def register_middleware(self, name, callback):
        self.middleware[name] = callback

    def on_unload(self, callback):
        self.unload.append(callback)

    def close(self):
        for callback in reversed(self.unload):
            callback()


class ProcessRegistrationTests(unittest.TestCase):
    def setUp(self):
        state = plugin._process_state()
        self.assertTrue(state is None or state.owner is None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        constants = ModuleType("hermes_constants")
        constants.get_hermes_home = lambda: str(self.root)
        self.modules = mock.patch.dict(sys.modules, {"hermes_constants": constants})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        Proxy.instances = []
        Proxy.fail_start = Proxy.fail_close = False
        self.proxies = mock.patch.object(plugin, "ProviderProxy", Proxy)
        self.proxies.start()
        self.addCleanup(self.proxies.stop)
        self.authorities = []
        def authority(*args, **kwargs):
            result = Authority()
            self.authorities.append(result)
            return result
        self.authority_factory = mock.patch.object(plugin, "load_authority", side_effect=authority)
        self.authority_factory.start()
        self.addCleanup(self.authority_factory.stop)

    def context(self, **config):
        result = Context(self.root, **config)
        self.addCleanup(result.close)
        return result

    def verify(self, runtime):
        runtime.begin(session_id="session", turn_id="turn", user_message="Hello", platform="whatsapp")
        self.assertTrue(plugin.owner_turn("session", "turn"))

    def module_copy(self):
        name = "owner_mode._registration_test_copy"
        spec = importlib.util.spec_from_file_location(name, Path(plugin.__file__))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_duplicate_context_is_not_read_and_registers_nothing(self):
        first = self.context(port=8123)
        runtime = plugin.register(first)
        self.verify(runtime)
        duplicate = self.context(port=9456, routes={"different": {}})
        self.assertIs(plugin.register(duplicate), runtime)
        self.assertEqual(duplicate.config_reads, 0)
        self.assertEqual((duplicate.hooks, duplicate.middleware, len(duplicate.unload)),
                         (first.hooks, first.middleware, 1))
        self.assertEqual(len(Proxy.instances), 1)
        self.assertEqual(len(self.authorities), 1)
        duplicate.close()
        self.assertTrue(plugin.owner_turn("session", "turn"))
        self.assertTrue(runtime.proxy.started)

    def test_distinct_import_uses_first_owner_and_never_creates_a_proxy(self):
        first = self.context()
        runtime = plugin.register(first)
        self.verify(runtime)
        other = self.module_copy()
        with mock.patch.object(other, "ProviderProxy", side_effect=AssertionError("duplicate proxy")):
            duplicate = self.context()
            self.assertIs(other.register(duplicate), runtime)
        self.assertEqual(duplicate.config_reads, 0)
        self.assertTrue(other.owner_turn("session", "turn"))
        first.close()
        self.assertTrue(other.owner_turn("session", "turn"))

    def test_module_reload_keeps_owner_and_saved_accessor(self):
        first = self.context()
        runtime = plugin.register(first)
        self.verify(runtime)
        saved_accessor = plugin.owner_turn
        reloaded = importlib.reload(plugin)
        duplicate = self.context()
        self.assertIs(reloaded.register(duplicate), runtime)
        self.assertTrue(saved_accessor("session", "turn"))
        self.assertTrue(reloaded.owner_turn("session", "turn"))
        first.close()
        self.assertTrue(saved_accessor("session", "turn"))

    def test_start_failure_closes_partial_resources_and_stale_hooks_are_inert(self):
        first = self.context()
        Proxy.fail_start = True
        with self.assertRaisesRegex(RuntimeError, "fixture startup"):
            plugin.register(first)
        Proxy.fail_start = False
        self.assertEqual(Proxy.instances[0].closed, 1)
        self.assertEqual(self.authorities[0].closed, 1)
        self.assertFalse(plugin.owner_turn("session", "turn"))
        replacement = self.context()
        runtime = plugin.register(replacement)
        self.verify(runtime)
        request = {"extra_headers": {"X-Owner-Mode-Context": "already-signed"}}
        answer = first.middleware["llm_request"](request)
        self.assertIs(answer["request"], request)
        called = []
        self.assertEqual(first.middleware["tool_execution"]({"x": 1}, lambda value: called.append(value) or 7), 7)
        self.assertEqual(called, [{"x": 1}])
        first.hooks["pre_llm_call"](session_id="session", turn_id="turn", user_message="Other", platform="whatsapp")
        first.hooks["post_llm_call"](session_id="session", turn_id="turn")
        first.close()
        self.assertTrue(plugin.owner_turn("session", "turn"))
        self.assertTrue(runtime.proxy.started)

    def test_unloaded_hooks_leave_request_and_new_authority_untouched(self):
        first = self.context()
        old_runtime = plugin.register(first)
        self.verify(old_runtime)
        first.close()
        replacement = self.context()
        runtime = plugin.register(replacement)
        self.assertFalse(plugin.owner_turn("session", "turn"))
        self.verify(runtime)
        first.hooks["post_llm_call"](session_id="session", turn_id="turn")
        first.hooks["pre_tool_call"](tool_name="delegate_task", args={})
        request = {"extra_headers": {"X-Owner-Mode-Context": "live-owner"}, "owner_mode": "existing"}
        self.assertIs(first.middleware["llm_request"](request)["request"], request)
        first.close()
        self.assertTrue(plugin.owner_turn("session", "turn"))
        self.assertEqual(old_runtime.proxy.closed, 1)
        self.assertEqual(runtime.proxy.closed, 0)

    def test_forked_accessor_and_unload_do_not_touch_parent_state(self):
        first = self.context()
        runtime = plugin.register(first)
        self.verify(runtime)
        original_pid = os.getpid()
        with mock.patch.object(plugin.os, "getpid", return_value=original_pid + 100000):
            self.assertFalse(plugin.owner_turn("session", "turn"))
            first.close()
        self.assertTrue(plugin.owner_turn("session", "turn"))
        self.assertEqual(runtime.proxy.closed, 0)

    def test_concurrent_registrations_construct_one_runtime(self):
        barrier = threading.Barrier(4)
        contexts = [self.context() for _ in range(4)]
        def register(ctx):
            barrier.wait(timeout=5)
            return plugin.register(ctx)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(register, contexts))
        self.assertEqual(sum(result is not None for result in results), 4)
        self.assertEqual(len(Proxy.instances), 1)
        self.assertEqual(len(self.authorities), 1)
        self.assertEqual(sum(len(ctx.unload) for ctx in contexts), 4)

    def test_failed_cleanup_blocks_replacement_until_retry_finishes(self):
        first = self.context()
        runtime = plugin.register(first)
        self.verify(runtime)
        Proxy.fail_close = True
        with self.assertLogs(plugin.logger, level="WARNING") as captured:
            first.close()
        self.assertEqual([row.getMessage() for row in captured.records], ["FlashAlign plugin cleanup incomplete"])
        self.assertFalse(plugin.owner_turn("session", "turn"))
        self.assertIsNone(plugin.register(self.context()))
        Proxy.fail_close = False
        first.close()
        self.assertIsNotNone(plugin.register(self.context()))

    def test_retired_proxy_rejects_preparation_even_if_cleanup_failed(self):
        first = self.context()
        runtime = plugin.register(first)
        self.verify(runtime)
        # The request body and approval mode cannot bypass the owner guard.
        runtime.proxy.approvals = type("AlwaysOn", (), {"read": lambda *a, **k: {"enabled": True, "mode": "always_on"}})()
        runtime.policy.tickets["fixture"] = object()
        runtime.policy.contexts["fixture"] = object()
        runtime.policy.context_infos["fixture"] = object()
        runtime.policy.children["fixture"] = object()
        Proxy.fail_close = True
        try:
            with self.assertLogs(plugin.logger, level="WARNING"):
                first.close()
            with self.assertRaisesRegex(RuntimeError, "registration is inactive"):
                runtime.proxy.prepare("head", {"owner_mode": "owner-test"})
            self.assertIsNone(runtime.policy.authority)
            for table in (runtime.policy.turns, runtime.policy.tickets, runtime.policy.contexts,
                          runtime.policy.context_infos, runtime.policy.children):
                self.assertFalse(table)
        finally:
            Proxy.fail_close = False
            first.close()

    def test_inflight_verification_cannot_retain_a_proof_after_unload(self):
        entered, released = threading.Event(), threading.Event()
        first = self.context()
        runtime = plugin.register(first)
        class BlockingAuthority(Authority):
            def verify_turn(self, **context):
                entered.set()
                if not released.wait(5):
                    raise RuntimeError("fixture verification timed out")
                return super().verify_turn(**context)
        runtime.policy.authority = BlockingAuthority()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime.begin, session_id="session", turn_id="turn",
                                 user_message="Hello", platform="whatsapp")
            try:
                self.assertTrue(entered.wait(5))
                first.close()
            finally:
                released.set()
            future.result(timeout=5)
        self.assertFalse(runtime.policy.turns)
        self.assertFalse(plugin.owner_turn("session", "turn"))

    def test_hook_registration_failure_releases_claim(self):
        context = self.context()
        context.register_hook = mock.Mock(side_effect=RuntimeError("fixture hook failure"))
        with self.assertRaisesRegex(RuntimeError, "fixture hook"):
            plugin.register(context)
        self.assertEqual(Proxy.instances[0].closed, 1)
        self.assertEqual(self.authorities[0].closed, 1)
        self.assertIsNotNone(plugin.register(self.context()))


if __name__ == "__main__":
    unittest.main()
