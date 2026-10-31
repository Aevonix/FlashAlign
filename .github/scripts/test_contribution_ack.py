# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import hashlib
import unittest

from check_contribution_ack import DECLARATION, acknowledgment_checked, build_record


class AcknowledgmentTests(unittest.TestCase):
    def test_accepts_existing_wrapped_checked_declaration(self):
        body = "- [X] " + DECLARATION.replace("contribution and", "\n  contribution and")
        self.assertTrue(acknowledgment_checked(body))

    def test_does_not_accept_unchecked_or_incomplete_text(self):
        self.assertFalse(acknowledgment_checked("- [ ] " + DECLARATION))
        self.assertFalse(acknowledgment_checked("- [x] I agree to the terms."))
        self.assertFalse(acknowledgment_checked(None))

    def test_does_not_accept_examples_comments_or_quotations(self):
        checked = "- [x] " + DECLARATION
        for body in ("```markdown\n" + checked + "\n```", "~~~\n" + checked + "\n~~~",
                     "<!--\n" + checked + "\n-->", "> " + checked, "    " + checked,
                     "````markdown\n```\n" + checked + "\n````"):
            with self.subTest(body=body):
                self.assertFalse(acknowledgment_checked(body))

    def test_rejects_contradictory_duplicate_declarations(self):
        self.assertFalse(acknowledgment_checked("- [x] " + DECLARATION + "\n- [ ] " + DECLARATION))

    def event(self, body=None):
        return {
            "repository": {"full_name": "example/project"},
            "sender": {"login": "reviewer"},
            "pull_request": {
                "number": 7, "html_url": "https://github.com/example/project/pull/7",
                "user": {"login": "contributor"}, "body": body or "- [x] " + DECLARATION,
                "head": {"sha": "a" * 40}, "base": {"sha": "b" * 40},
            },
        }

    def test_record_captures_revision_and_terms_without_raw_pr_text(self):
        terms = b"# Terms\n\nVersion 1.0, 2026-10-30\n"
        body = "$(touch should-never-run)\n\n- [x] " + DECLARATION
        record = build_record(self.event(body), terms, "2026-10-31T00:00:00+00:00")
        self.assertEqual(record["terms_sha256"], hashlib.sha256(terms).hexdigest())
        self.assertEqual(record["covered_head_commit"], "a" * 40)
        self.assertEqual(record["observed_event_actor"], "reviewer")
        self.assertEqual(record["submitting_account"], "contributor")
        self.assertTrue(record["maintainer_review_required"])
        self.assertNotIn("body", record)
        self.assertNotIn("touch", str(record))

    def test_terms_version_and_revision_must_be_known(self):
        with self.assertRaises(ValueError):
            build_record(self.event(), b"Version 2.0\n")
        event = self.event()
        event["pull_request"]["head"]["sha"] = "unreviewed"
        with self.assertRaises(ValueError):
            build_record(event, b"Version 1.0\n")


if __name__ == "__main__":
    unittest.main()
