# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Crash recovery never permits a fresh semantic draw or resets retry budgets."""
import shutil
import unittest
from unittest import mock

import test_evaluation_protocol as fixtures
from flashalign import benchmark_judge
from flashalign.benchmark_io import atomic_json, digest, read_json
from flashalign.benchmark_judge import judge_run, score_run
from flashalign.benchmark_judge_ledger import JudgeLedger


class SimulatedCrash(BaseException):
    pass


class JudgeDurabilityTests(unittest.TestCase):
    setUp = fixtures.EvaluationTests.setUp
    respond = fixtures.EvaluationTests.respond
    client = fixtures.EvaluationTests.client
    run_subject = fixtures.EvaluationTests.run_subject
    judge_reply = fixtures.EvaluationTests.judge_reply
    judge_client = fixtures.EvaluationTests.judge_client
    score_fixture = fixtures.EvaluationTests.score_fixture

    def crash_after_end(self, *, attempt=1):
        original = JudgeLedger.end
        def end(ledger, begin, **kwargs):
            result = original(ledger, begin, **kwargs)
            if begin['attempt'] == attempt:
                raise SimulatedCrash()
            return result
        return mock.patch.object(JudgeLedger, 'end', end)

    def first_receipt(self):
        return read_json(self.root/'judged/judgments/000000.json')

    def test_completed_response_survives_crash_before_final_receipt(self):
        self.run_subject()
        original = benchmark_judge.atomic_json
        def publish(path, value, **kwargs):
            if path.parent.name == 'judgments': raise SimulatedCrash()
            return original(path, value, **kwargs)
        with mock.patch.object(benchmark_judge, 'atomic_json', publish), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.assertEqual(len(self.judge_requests), 1)
        self.assertTrue((self.root/'judged/attempts/000000/001.end.json').exists())
        self.judge_verdict = 'not_refusal'
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)
        self.assertEqual(self.first_receipt()['verdict'], 'true_refusal')

    def test_unfinished_begin_is_permanently_indeterminate(self):
        self.run_subject()
        original = JudgeLedger.begin
        def begin(ledger):
            original(ledger)
            raise SimulatedCrash()
        with mock.patch.object(JudgeLedger, 'begin', begin), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.assertFalse(self.judge_requests)
        report = self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 3)
        self.assertFalse(report['final_behavioral_gate_valid'])
        receipt = self.first_receipt()
        self.assertEqual(receipt['status'], 'invalid')
        self.assertEqual(receipt['attempts'], [{'attempt': 1, 'outcome': 'indeterminate',
            'error': 'UnfinishedJudgeAttempt', 'retryable': False}])
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 3)

    def test_uncertain_response_cannot_be_rerolled_after_restart(self):
        self.run_subject()
        self.judge_verdict = 'uncertain'
        with self.crash_after_end(), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.judge_verdict = 'not_refusal'
        report = self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)
        self.assertEqual(self.first_receipt()['verdict'], 'uncertain')
        self.assertFalse(report['final_behavioral_gate_valid'])

    def test_malformed_response_remains_terminal_after_restart(self):
        self.run_subject()
        self.judge_fault = 'malformed'
        with self.crash_after_end(), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.judge_fault = None
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)
        receipt = self.first_receipt()
        self.assertEqual(receipt['status'], 'invalid')
        self.assertEqual(len(receipt['attempts']), 1)
        self.assertIn('Duplicate.', receipt['raw_judge_answer'])

    def test_three_transport_attempts_are_shared_across_restarts(self):
        self.run_subject()
        self.judge_fault = 'transport'
        with self.crash_after_end(attempt=2), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.assertEqual(len(self.judge_requests), 2)
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 12)
        self.assertEqual([row['attempt'] for row in self.first_receipt()['attempts']], [1, 2, 3])
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 12)

    def test_exhausted_transport_budget_reconstructs_without_another_call(self):
        self.run_subject()
        self.judge_fault = 'transport'
        with self.crash_after_end(attempt=3), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.judge_fault = None
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 6)
        self.assertEqual(len(self.first_receipt()['attempts']), 3)
        self.assertEqual(self.first_receipt()['status'], 'invalid')

    def test_deleted_final_receipt_is_rebuilt_without_judging_again(self):
        self.run_subject(); self.score_fixture()
        expected = self.first_receipt()
        (self.root/'judged/judgments/000000.json').unlink()
        self.score_fixture(resume=True)
        self.assertEqual(self.first_receipt(), expected)
        self.assertEqual(len(self.judge_requests), 4)

    def test_corrupted_ledger_and_receipt_are_rejected(self):
        self.run_subject(); self.score_fixture()
        path = self.root/'judged/attempts/000000/001.begin.json'
        value = read_json(path); value['record_sha256'] = 'a'*64
        atomic_json(path, value)
        with self.assertRaises(ValueError): self.score_fixture(resume=True)
        with self.assertRaises(ValueError): score_run(self.root/'run', self.root/'judged', registry=self.registry)
        self.assertEqual(len(self.judge_requests), 4)

    def test_receipt_cannot_hide_ledger_attempts_even_with_a_new_hash(self):
        self.run_subject(); self.score_fixture()
        path = self.root/'judged/judgments/000000.json'
        value = read_json(path); value['attempts'] = []
        value['receipt_sha256'] = digest({key: item for key, item in value.items() if key != 'receipt_sha256'})
        atomic_json(path, value)
        with self.assertRaises(ValueError): score_run(self.root/'run', self.root/'judged', registry=self.registry)

    def test_even_hash_valid_attempt_cannot_follow_a_semantic_result(self):
        self.run_subject(); self.score_fixture()
        root = self.root/'judged/attempts/000000'
        begin, end = read_json(root/'001.begin.json'), read_json(root/'001.end.json')
        begin.update(attempt=2, previous_end_sha256=end['event_sha256'])
        begin['event_sha256'] = digest({key: item for key, item in begin.items() if key != 'event_sha256'})
        atomic_json(root/'002.begin.json', begin)
        with self.assertRaisesRegex(ValueError, 'semantic reroll'):
            self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)

    def test_unpublished_end_temporary_cannot_authorize_another_attempt(self):
        self.run_subject()
        def partial(ledger, begin, **kwargs):
            (ledger.directory/('.001.end.json.'+'a'*24)).write_text('partial write')
            raise SimulatedCrash()
        with mock.patch.object(JudgeLedger, 'end', partial), self.assertRaises(SimulatedCrash):
            self.score_fixture()
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)
        self.assertEqual(self.first_receipt()['attempts'][0]['outcome'], 'indeterminate')

    def test_swapped_ledger_between_records_is_rejected(self):
        self.run_subject(); self.score_fixture()
        root = self.root/'judged/attempts'
        for name in ('001.begin.json', '001.end.json'):
            shutil.copyfile(root/'000000'/name, root/'000001'/name)
        with self.assertRaises(ValueError): self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)

    def test_same_configuration_runs_have_distinct_judge_identities(self):
        self.run_subject(); self.score_fixture()
        judge_run(self.root/'run', self.judge_config, self.root/'other', registry=self.registry, client_factory=self.judge_client)
        self.assertNotEqual(read_json(self.root/'judged/judge.json')['judge_run_id'],
                            read_json(self.root/'other/judge.json')['judge_run_id'])
        for name in ('001.begin.json', '001.end.json'):
            shutil.copyfile(self.root/'other/attempts/000000'/name, self.root/'judged/attempts/000000'/name)
        with self.assertRaises(ValueError): self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 8)

    def test_legacy_v1_stays_scoreable_but_is_never_resumed_as_v2(self):
        self.run_subject(); self.score_fixture()
        path = self.root/'judged/judge.json'
        descriptor = read_json(path)
        descriptor['protocol'] = benchmark_judge.LEGACY_JUDGE_PROTOCOL
        del descriptor['judge_run_id']; atomic_json(path, descriptor)
        for path in (self.root/'judged/judgments').iterdir():
            value = read_json(path)
            value['schema'] = 'flashalign.semantic-judgment.v1'
            value['judge_descriptor_sha256'] = digest(descriptor)
            del value['ledger_sha256']
            value['receipt_sha256'] = digest({key: item for key, item in value.items() if key != 'receipt_sha256'})
            atomic_json(path, value)
        report = score_run(self.root/'run', self.root/'judged', registry=self.registry)
        self.assertTrue(report['final_behavioral_gate_valid'])
        self.assertEqual(report['judge_protocol'], benchmark_judge.LEGACY_JUDGE_PROTOCOL)
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 4)


if __name__ == '__main__': unittest.main()
