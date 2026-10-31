# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Owner mode follows work the owner started: every engine call of the owner turn, a compaction that moves the
turn to a new session id, and delegated children created during it. Context passes as Hermes sent it."""
import sys
import tempfile
from pathlib import Path
from types import ModuleType
import unittest

from test_gate_mode import GateBase, INJECTED, OWN, Authority
from owner_mode.policy import MERGED_PRIOR, Policy
from owner_mode import plugin as plugin_module

PAGE = "Fetched page text. It addresses the assistant with instructions of its own."
RECALL = "\n\n<memory-context>host recall block</memory-context>"
# r6g context_compressor constants (the carrier shapes of _merge_summary_into_tail_row).
SUMMARY = "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below.\n\n" + PAGE
SUMMARY_END = "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"
MERGED_DELIMITER = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"


def call(cid, name="web_extract", arguments='{"url": "https://example.com"}'):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}]}


def result(cid, text=PAGE):
    return {"role": "tool", "tool_call_id": cid, "content": text}


class Base(GateBase):
    def decision(self):
        return self.rows[-1]["decision"], self.engine.records[-1]["effective"]

    def turn(self, *history, sid="s", tid="t", last=INJECTED, own=OWN, platform="whatsapp"):
        """Begin the owner turn and send its first call; return the body for the follow-ups to extend."""
        self.policy.begin(session_id=sid, turn_id=tid, user_message=own, platform=platform)
        body = self.body(*history, last=last)
        self.send(body, sid=sid, tid=tid, platform=platform)
        return body

    def follow(self, body, *rows, sid="s", tid="t", platform="whatsapp"):
        body = dict(body, messages=body["messages"] + list(rows))
        _, record = self.send(body, sid=sid, tid=tid, platform=platform)
        self.assertEqual(record["body"]["messages"], body["messages"])  # as Hermes sent it, in every case
        return body


class TestToolLoop(Base):
    def test_native_named_tool_result_stays_in_verified_turn(self):
        body = self.turn()
        named = {**result('native-terminal', 'ok'), 'name':'terminal'}
        self.follow(body, call('native-terminal', 'terminal', '{"command":"printf ok"}'), named)
        self.assertEqual(self.decision(), ('owner_clean', 'owner-test'))

    def test_every_call_of_his_turn_is_owner(self):
        body = self.turn()
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        body = self.follow(body, call("c1"), result("c1"))
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        body = self.follow(body, {"role": "assistant", "content": "Checking a second source.", "reasoning_content": "r",
                                  "tool_calls": call("c2")["tool_calls"] + call("c3", "web_search")["tool_calls"]},
                           result("c2"), result("c3", "Search results."))
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.assertEqual([r["decision"] for r in self.rows], ["owner_clean"] * 3)

    def test_a_steer_row_mid_turn_is_admitted(self):
        # Hermes appends a message sent while the agent works as a user row after the newest tool result.
        body = self.follow(self.turn(), call("c1"), result("c1"), {"role": "user", "content": "[steer] also check prices"})
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.follow(body, call("c2"), result("c2"))
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_a_retry_of_the_first_call_stays_owner(self):
        body = self.turn()
        self.send(body)
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_tool_rows_in_history_of_a_new_turn(self):
        history = [{"role": "user", "content": "Earlier question"}, call("c1"), result("c1"),
                   {"role": "assistant", "content": "Earlier answer", "reasoning": "r"}]
        self.turn(*history)
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_a_turn_whose_first_call_is_not_his_words_stays_off(self):
        body = self.turn(last="Someone else's words")
        self.assertEqual(self.decision(), ("last_not_owner_words", "off"))
        self.follow(body, call("c1"), result("c1"))
        self.assertEqual(self.decision(), ("last_not_owner_user", "off"))

    def test_a_turn_refused_on_its_first_call_is_not_anchored_by_a_later_one(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        body = self.body(top_logprobs=5)
        self.send(body)
        self.assertEqual(self.decision(), ("unknown_request_field", "off"))
        self.follow(dict(self.body()), call("c1"), result("c1"))
        self.assertEqual(self.decision(), ("last_not_owner_user", "off"))

    def test_someone_elses_turn_stays_off_on_every_call(self):
        self.policy.authority = Authority(owner=False)
        body = self.turn(sid="g")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))
        self.follow(body, call("c1"), result("c1"), sid="g")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))

    def test_no_time_limit_on_his_turn(self):
        now = [1000.0]
        self.policy.clock = lambda: now[0]
        body = self.turn()
        now[0] += 3600
        self.follow(body, call("c1"), result("c1"))
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.policy.begin(session_id="s", turn_id="t2", user_message=OWN, platform="whatsapp")
        now[0] += 3600
        self.send(self.body(), tid="t2")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_the_ticket_still_binds_one_call(self):
        body = self.turn()
        token, reason = self.policy.decide(dict(body, messages=body["messages"] + [call("c1"), result("c1")]),
                                           session_id="s", turn_id="t", profile="owner-test", route="head")
        self.assertEqual(reason, "owner_clean")
        self.assertEqual(self.policy.consume(token, body, "head"), "off")  # a different body
        self.assertEqual(self.policy.consume(token, body, "head"), "off")  # single use

    def test_max_output_tokens_is_allowed(self):
        self.policy.begin(session_id="s", turn_id="t", user_message=OWN, platform="whatsapp")
        self.send(self.body(max_output_tokens=1024))
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))


class TestShapesStillRefused(Base):
    """Malformed or unknown shapes stay off on a follow-up too, and the original body goes through unchanged."""

    def test_malformed_rows(self):
        cases = {
            "tool row with an invalid name": ({"role": "tool", "tool_call_id": "c1", "name": {}, "content": PAGE}, "bad_tool_history"),
            "tool row with an empty name": ({"role": "tool", "tool_call_id": "c1", "name": "", "content": PAGE}, "bad_tool_history"),
            "tool row with private host metadata": ({"role": "tool", "tool_call_id": "c1", "name": "terminal",
                                                     "tool_name": "terminal", "content": PAGE}, "bad_tool_history"),
            "tool row without its call id": ({"role": "tool", "content": PAGE}, "bad_tool_history"),
            "tool row with an object for content": ({"role": "tool", "tool_call_id": "c1", "content": {"x": 1}},
                                                    "bad_tool_history"),
            "function role": ({"role": "function", "name": "x", "content": PAGE}, "unknown_role"),
            "unknown role": ({"role": "developer", "content": "x"}, "unknown_role"),
            "function_call field": ({"role": "assistant", "content": None, "function_call": {"name": "x"}},
                                     "bad_assistant_history"),
            "tool_calls not a list": ({"role": "assistant", "content": None, "tool_calls": {"id": "c1"}},
                                      "bad_assistant_history"),
            "tool_calls null": ({"role": "assistant", "content": None, "tool_calls": None}, "bad_assistant_history"),
            "tool call without a function": ({"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                                             "bad_assistant_history"),
            "assistant extra key": (dict(call("c1"), refusal="x"), "bad_assistant_history"),
            "user row extra key": ({"role": "user", "content": "x", "name": "y"}, "bad_user_history"),
            "user row object content": ({"role": "user", "content": {"x": 1}}, "bad_user_history"),
            "system row extra key": ({"role": "system", "content": "x", "name": "y"}, "bad_system_message"),
            "not an object": ("text", "bad_message"),
        }
        for n, (name, (row, reason)) in enumerate(cases.items()):
            with self.subTest(name):
                body = self.turn(sid=f"m{n}")
                self.follow(body, call("c1"), result("c1"), row, sid=f"m{n}")
                self.assertEqual(self.decision(), (reason, "off"))

    def test_request_level_refusals_hold_on_a_follow_up(self):
        for n, (extra, reason) in enumerate(((dict(top_logprobs=5), "unknown_request_field"), (dict(n=2), "n_not_1"),
                                             (dict(response_format={"type": "json_object"}), "response_format"),
                                             (dict(tool_choice={"type": "function"}), "tool_choice"))):
            with self.subTest(reason):
                body = self.turn(sid=f"r{n}")
                self.follow(dict(body, **extra), call("c1"), result("c1"), sid=f"r{n}")
                self.assertEqual(self.decision(), (reason, "off"))

    def test_strict_trust_still_pins_and_refuses_history(self):
        self.policy.trust = dict(self.policy.trust, system_prompt="pinned", tools="pinned")
        body = self.turn()
        self.assertEqual(self.decision(), ("system_prompt_unpinned", "off"))
        self.policy.trust = dict(self.policy.trust, system_prompt="host")
        body = self.turn(tid="t2")
        self.assertEqual(self.decision(), ("tools_unpinned", "off"))
        self.policy.trust = dict(self.policy.trust, tools="host", history="none")
        body = self.turn(tid="t3")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.follow(body, call("c1"), result("c1"), tid="t3")
        self.assertEqual(self.decision(), ("history_present", "off"))


class TestCompactionMidTurn(Base):
    def test_a_new_session_id_keeps_his_turn(self):
        body = self.turn(sid="s1", tid="s1:task:0001")
        # Hermes compacts mid-turn: a new session id, the same turn id, a summary row in place of older rows.
        compacted = dict(body, messages=[body["messages"][0], {"role": "user", "content": SUMMARY}, result("c9")])
        self.follow(compacted, call("c10"), result("c10"), sid="s2", tid="s1:task:0001")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.assertTrue(self.rows[-1]["owner_verified"])
        self.policy.end(session_id="s2", turn_id="s1:task:0001")  # post_llm_call names the new session
        self.follow(compacted, sid="s2", tid="s1:task:0001")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))

    def test_an_unknown_turn_id_under_a_new_session_is_off(self):
        body = self.turn(sid="s1", tid="t1")
        self.follow(body, call("c1"), result("c1"), sid="s2", tid="t9")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))

    def test_a_turn_id_held_by_two_turns_matches_neither(self):
        body = self.turn(sid="a", tid="t")
        self.turn(sid="b", tid="t")
        self.follow(body, call("c1"), result("c1"), sid="c", tid="t")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))


class TestCompactionCarrier(Base):
    """A turn-start compaction can fold the summary into his row (r6g _merge_summary_into_tail_row)."""

    def test_both_merge_branches_qualify(self):
        cases = {"summary first": SUMMARY + "\n\n" + SUMMARY_END + "\n\n" + OWN + RECALL,
                 "row first": MERGED_PRIOR + "\n" + OWN + "\n\n" + MERGED_DELIMITER + "\n\n" + SUMMARY + "\n\n"
                              + SUMMARY_END + RECALL,
                 "row first, no host context": MERGED_PRIOR + "\n" + OWN + "\n\n" + MERGED_DELIMITER + "\n\n"
                                               + SUMMARY + "\n\n" + SUMMARY_END,
                 "row first, a reply pointer above his words": MERGED_PRIOR + "\n[Replying to: \"x\"]\n\n" + OWN
                                                               + "\n\n" + MERGED_DELIMITER + "\n\n" + SUMMARY}
        for n, (name, last) in enumerate(cases.items()):
            with self.subTest(name):
                self.turn(sid=f"c{n}", last=last)
                self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_the_header_counts_only_at_the_start_and_only_before_his_words(self):
        cases = {"other words after the header": MERGED_PRIOR + "\nSomeone else\n\n" + MERGED_DELIMITER,
                 "his words glued to more text": MERGED_PRIOR + "\n" + OWN + "x\n\n" + MERGED_DELIMITER,
                 "header not at the start": "x" + MERGED_PRIOR + "\n" + OWN + "\n\n" + MERGED_DELIMITER,
                 "header without its line break": MERGED_PRIOR + OWN + "\n\n" + MERGED_DELIMITER}
        for n, (name, last) in enumerate(cases.items()):
            with self.subTest(name):
                self.turn(sid=f"h{n}", last=last)
                self.assertEqual(self.decision(), ("last_not_owner_words", "off"))

    def test_the_header_is_host_text_only_with_host_turn_context(self):
        self.policy.trust = dict(self.policy.trust, turn_context="none")
        self.assertFalse(self.policy._owner_words(MERGED_PRIOR + "\n" + OWN + "\n\nx", OWN))


class TestChildren(Base):
    """delegate_task: subagent_start (adopt) fires before the child's first turn (begin with parent_session_id)."""

    def child(self, *, parent="s", parent_turn="t", child="k", tid="kt", begin_parent=None, goal="Research the venue."):
        self.policy.adopt(parent_session_id=parent, parent_turn_id=parent_turn, child_session_id=child,
                          child_subagent_id="sa-" + child, child_goal=goal)
        self.policy.begin(session_id=child, turn_id=tid, user_message=goal, platform="subagent",
                          parent_session_id=parent if begin_parent is None else begin_parent)
        body = {"model": "bench", "messages": [{"role": "system", "content": "Child prompt."},
                                               {"role": "user", "content": goal}], "tools": self.tools}
        self.send(body, sid=child, tid=tid, platform="subagent")
        return body

    def test_a_child_of_his_turn_is_owner_on_every_call(self):
        parent = self.turn()
        self.follow(parent, call("d1", "delegate_task", '{"goal": "Research the venue."}'))
        body = self.child()
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.follow(body, call("c1"), result("c1"), sid="k", tid="kt", platform="subagent")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_a_grandchild_follows_too(self):
        self.turn()
        self.child()
        self.child(parent="k", parent_turn="kt", child="g", tid="gt")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_a_child_of_a_compacted_parent_turn(self):
        body = self.turn(sid="s1", tid="s1:task:1")
        self.follow(body, call("d1", "delegate_task"), sid="s2", tid="s1:task:1")
        self.child(parent="s2", parent_turn="s1:task:1")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_children_that_are_not_his_work_stay_off(self):
        cases = {"parent turn never passed": dict(setup="unanchored"),
                 "parent is someone else": dict(setup="guest"),
                 "another turn of the parent": dict(parent_turn="other"),
                 "begin names another parent": dict(begin_parent="x"),
                 "never adopted": dict(setup="no_adopt")}
        for n, (name, case) in enumerate(cases.items()):
            with self.subTest(name):
                setup = case.pop("setup", "")
                sid = f"p{n}"
                if setup == "guest":
                    self.policy.authority = Authority(owner=False)
                self.policy.begin(session_id=sid, turn_id="t", user_message=OWN, platform="whatsapp")
                self.policy.authority = Authority()
                if setup != "unanchored":
                    self.send(self.body(), sid=sid)
                if setup == "no_adopt":
                    self.policy.begin(session_id=f"k{n}", turn_id="kt", user_message="goal", platform="subagent",
                                      parent_session_id=sid)
                    self.send(self.body(last="goal"), sid=f"k{n}", tid="kt", platform="subagent")
                else:
                    self.child(parent=sid, child=f"k{n}", **case)
                self.assertEqual(self.decision(), ("owner_unverified", "off"))

    def test_a_child_platform_is_never_an_owner_surface_on_its_own(self):
        self.policy.begin(session_id="k", turn_id="kt", user_message=OWN, platform="subagent")
        self.send(self.body(), sid="k", tid="kt", platform="subagent")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))


class TestSteer(Base):
    """delegate_task(action="steer") from a turn that is not owner work takes owner mode from that child (PM ruling,
    10-28). Hermes delivers the steer as a user row after the child's newest tool result."""

    child = TestChildren.child
    STEER_ROW = {"role": "user", "content": "[OUT-OF-BAND USER MESSAGE]\nChange course.\n[/OUT-OF-BAND USER MESSAGE]"}

    def steer(self, *, sid="s", tid="t", target="sa-k", action="steer", tool="delegate_task"):
        self.assertIsNone(self.policy.steered(tool_name=tool, args={"action": action, "subagent_id": target,
                                                                    "message": "Change course."},
                                              session_id=sid, turn_id=tid, tool_call_id="c", task_id=""))

    def later_turn_not_his(self, sid="s", tid="t2"):
        """His turn has ended; a later turn of the same session is not owner work (unverified)."""
        self.policy.end(session_id="s", turn_id="t")
        self.policy.authority = Authority(owner=False)
        self.policy.begin(session_id=sid, turn_id=tid, user_message=OWN, platform="whatsapp")
        self.policy.authority = Authority()

    def child_call(self, body, child="k", tid="kt"):
        return self.follow(body, call("c1"), result("c1"), self.STEER_ROW, sid=child, tid=tid, platform="subagent")

    def test_his_turn_may_steer_its_child(self):
        self.turn()
        body = self.child()
        self.steer()
        self.child_call(body)
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_a_steer_from_a_later_turn_that_is_not_his_takes_owner_mode_for_good(self):
        self.turn()
        body = self.child()
        follow_up = dict(body, messages=body["messages"] + [call("c0"), result("c0")])
        token, _ = self.policy.decide(follow_up, session_id="k", turn_id="kt", profile="owner-test", route="head")
        self.later_turn_not_his()
        self.steer(tid="t2")
        self.assertEqual(self.policy.consume(token, follow_up, "head"), "off")  # a ticket issued before the steer
        self.child_call(body)
        self.assertEqual(self.decision(), ("non_owner_steer", "off"))
        # A later turn of that child is no longer adopted, and it adopts no children of its own.
        self.policy.begin(session_id="k", turn_id="kt2", user_message="goal", platform="subagent", parent_session_id="s")
        self.send(self.body(last="goal"), sid="k", tid="kt2", platform="subagent")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))
        self.child(parent="k", parent_turn="kt", child="g", tid="gt")
        self.assertEqual(self.decision(), ("owner_unverified", "off"))

    def test_a_steer_from_a_turn_that_never_passed_takes_owner_mode(self):
        self.turn()
        body = self.child()
        self.policy.end(session_id="s", turn_id="t")
        self.turn(tid="t2", last="Someone else's words")
        self.steer(tid="t2")
        self.child_call(body)
        self.assertEqual(self.decision(), ("non_owner_steer", "off"))

    def test_only_the_steered_child_and_only_by_its_exact_id(self):
        self.turn()
        body = self.child()
        self.child(child="k2", tid="k2t")
        self.later_turn_not_his()
        self.steer(tid="t2", target="sa-k2")
        self.steer(tid="t2", target="sa-k-other")
        self.child_call(body)
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.send(self.body(last="goal"), sid="k2", tid="k2t", platform="subagent")
        self.assertEqual(self.decision(), ("non_owner_steer", "off"))

    def test_other_actions_and_tools_change_nothing(self):
        self.turn()
        body = self.child()
        self.later_turn_not_his()
        for action, tool in (("list", "delegate_task"), ("stop", "delegate_task"), ("spawn", "delegate_task"),
                             ("", "delegate_task"), ("steer", "web_extract")):
            self.steer(tid="t2", action=action, tool=tool)
        self.child_call(body)
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))

    def test_the_action_and_id_are_read_as_hermes_reads_them(self):
        self.turn()
        body = self.child()
        self.later_turn_not_his()
        self.steer(tid="t2", action=" Steer ", target=" sa-k ")
        self.child_call(body)
        self.assertEqual(self.decision(), ("non_owner_steer", "off"))

    def test_grandchildren_follow_who_steers_them(self):
        self.turn()
        self.child()
        grandchild = self.child(parent="k", parent_turn="kt", child="g", tid="gt")
        self.steer(sid="k", tid="kt", target="sa-g")  # his child steers its own child: owner work
        self.child_call(grandchild, child="g", tid="gt")
        self.assertEqual(self.decision(), ("owner_clean", "owner-test"))
        self.later_turn_not_his()
        self.steer(tid="t2")  # the child loses owner mode; its steer of the grandchild is no longer owner work
        self.steer(sid="k", tid="kt", target="sa-g")
        self.send(self.body(last="goal"), sid="g", tid="gt", platform="subagent")
        self.assertEqual(self.decision(), ("non_owner_steer", "off"))


class TestRegistration(unittest.TestCase):
    """The plugin wires adopt() to subagent_start, ahead of the child's first turn, and steered() to pre_tool_call."""

    def test_adopt_is_registered_for_subagent_start(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        constants = ModuleType("hermes_constants")
        constants.get_hermes_home = lambda: tmp.name
        saved = sys.modules.get("hermes_constants")
        sys.modules["hermes_constants"] = constants
        self.addCleanup(lambda: sys.modules.__setitem__("hermes_constants", saved) if saved
                        else sys.modules.pop("hermes_constants", None))
        hooks, unload = {}, []

        class Ctx:
            def get_config(self, key, default=None):
                return {"key_file": str(Path(tmp.name) / "key")}.get(key, default)

            def register_hook(self, name, callback):
                hooks.setdefault(name, []).append(callback)

            def register_middleware(self, *_):
                pass

            def on_unload(self, callback):
                unload.append(callback)

        runtime = plugin_module.register(Ctx())
        self.addCleanup(lambda: [callback() for callback in unload])
        self.assertIsInstance(runtime.policy, Policy)
        self.assertEqual(hooks["subagent_start"][0], runtime.adopt)
        self.assertIn(runtime.policy.steered, hooks["pre_tool_call"])


if __name__ == "__main__":
    unittest.main()
