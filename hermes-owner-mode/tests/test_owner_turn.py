# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""owner_turn(session_id, turn_id): the one read-only interface other plugins use (PM, 10-28: messaging-rcs's contact
send). True only while this exact turn is one the owner verified himself: not a child adopted from it, not a steered
turn, not a turn whose words the identity adapter did not attest (a completion, an injection, a resume)."""
import sys
import tempfile
from pathlib import Path
from types import ModuleType
import unittest

from test_gate_mode import OWN, Authority
import test_owner_work as work  # module-qualified, so unittest does not collect its classes twice
from owner_mode import plugin as plugin_module


class TestOwnerTurn(work.Base):
    child = work.TestChildren.child

    def test_his_verified_turn_is_his_from_its_start(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.assertTrue(self.policy.owner_turn("s", "t"))  # his proof is the authority; no call needed yet
        self.turn()
        self.assertTrue(self.policy.owner_turn("s", "t"))

    def test_a_turn_he_did_not_attest_is_not(self):
        # A completion notice, a plugin injection or a resume reaches begin() with words the adapter does not attest.
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id="s", turn_id="t", user_message="[Background process finished]", platform="whatsapp")
        self.assertFalse(self.policy.owner_turn("s", "t"))

    def test_a_surface_that_is_not_his_is_not(self):
        for platform in ("cron", "cli", "api_server", "kanban", "subagent", ""):
            self.policy.begin(session_id="s", turn_id="t-" + platform, user_message=OWN, platform=platform)
            self.assertFalse(self.policy.owner_turn("s", "t-" + platform), platform)

    def test_unknown_and_empty_ids_are_not(self):
        self.turn()
        for sid, tid in (("s", "other"), ("s", ""), ("", ""), ("other", "other")):
            self.assertFalse(self.policy.owner_turn(sid, tid), (sid, tid))
        self.assertFalse(self.policy.owner_turn())

    def test_an_adopted_child_is_owner_work_but_not_his_turn(self):
        self.follow(self.turn(), work.call("d1", "delegate_task"))
        self.child()
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))  # owner mode follows his work...
        self.assertFalse(self.policy.owner_turn("k", "kt"))               # ...but the child's turn is not his
        self.child(parent="k", parent_turn="kt", child="g", tid="gt")
        self.assertFalse(self.policy.owner_turn("g", "gt"))
        self.assertTrue(self.policy.owner_turn("s", "t"))

    def test_a_steered_child_is_not(self):
        self.turn()
        self.child()
        work.TestSteer.later_turn_not_his(self)
        work.TestSteer.steer(self, tid="t2")
        self.assertFalse(self.policy.owner_turn("k", "kt"))
        self.assertFalse(self.policy.owner_turn("s", "t2"))

    def test_compaction_mid_turn_follows_the_turn_id(self):
        body = self.turn(sid="s1", tid="s1:task:1")
        self.follow(body, work.call("c1"), sid="s2", tid="s1:task:1")
        self.assertTrue(self.policy.owner_turn("s2", "s1:task:1"))
        self.assertFalse(self.policy.owner_turn("s2", "s2:task:2"))

    def test_it_ends_with_the_turn(self):
        self.turn()
        self.policy.end(session_id="s", turn_id="t")
        self.assertFalse(self.policy.owner_turn("s", "t"))

    def test_a_turn_being_verified_is_not_yet(self):
        seen = []
        policy = self.policy

        class Asking(Authority):
            def verify_turn(self, session_id, turn_id, user_message, **_):
                seen.append(policy.owner_turn(session_id, turn_id))
                return super().verify_turn(session_id, turn_id, user_message)

        self.policy.authority = Asking()
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.assertEqual(seen, [False])
        self.assertTrue(self.policy.owner_turn("s", "t"))

    def test_a_restarted_turn_key_loses_the_old_proof(self):
        self.turn()
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.assertFalse(self.policy.owner_turn("s", "t"))

    def test_it_never_raises(self):
        self.turn()

        def broken(*_):
            raise RuntimeError("store")

        self.policy._turn = broken
        self.assertFalse(self.policy.owner_turn("s", "t"))


class TestPublished(unittest.TestCase):
    """plugin.owner_turn is the module-level function messaging-rcs reads; it answers from the serving process's
    policy, and from nothing before register() or after unload."""

    def register(self, **config):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        constants = ModuleType("hermes_constants")
        constants.get_hermes_home = lambda: tmp.name
        saved = sys.modules.get("hermes_constants")
        sys.modules["hermes_constants"] = constants
        self.addCleanup(lambda: sys.modules.__setitem__("hermes_constants", saved) if saved
                        else sys.modules.pop("hermes_constants", None))
        self.unload = []
        values = dict({"key_file": str(Path(tmp.name) / "key")}, **config)

        class Ctx:
            def get_config(self, key, default=None):
                return values.get(key, default)

            def register_hook(self, *_):
                pass

            def register_middleware(self, *_):
                pass

            def on_unload(inner, callback):
                self.unload.append(callback)

        runtime = plugin_module.register(Ctx())
        self.addCleanup(self.run_unload)
        return runtime

    def run_unload(self):
        while self.unload:
            self.unload.pop()()

    def test_nothing_answers_before_register(self):
        self.assertIsNone(plugin_module._POLICY)
        self.assertFalse(plugin_module.owner_turn("s", "t"))

    def test_it_answers_from_the_serving_policy(self):
        function = plugin_module.owner_turn
        runtime = self.register()
        runtime.policy.authority = Authority()
        runtime.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.assertIs(plugin_module.owner_turn, function)  # one stable function; read it per call
        self.assertTrue(plugin_module.owner_turn("s", "t"))
        runtime.policy.authority = Authority(owner=False)
        runtime.policy.begin(session_id="s", turn_id="t2", user_message=OWN, platform="whatsapp")
        self.assertFalse(plugin_module.owner_turn("s", "t2"))

    def test_unload_withdraws_it(self):
        runtime = self.register()
        runtime.policy.authority = Authority()
        runtime.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.run_unload()
        self.assertFalse(plugin_module.owner_turn("s", "t"))

    def test_a_process_that_does_not_serve_owner_mode_publishes_nothing(self):
        self.assertIsNone(self.register(serve_argv=["gateway", "run", "--never-this-argv"]))
        self.assertFalse(plugin_module.owner_turn("s", "t"))


if __name__ == "__main__":
    unittest.main()
