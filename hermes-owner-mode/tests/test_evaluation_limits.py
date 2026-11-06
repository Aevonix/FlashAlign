# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Long synthetic streams, bounded durable records, and full judge evidence."""
import copy
import json
from pathlib import Path
import unittest

import httpx

import test_evaluation_protocol as fixtures
from flashalign import Client
from flashalign.benchmark_io import canonical, read_json
from flashalign.benchmark_judge import judge_once, judge_run
from flashalign.benchmark_protocol import collect, read_collection
from flashalign.benchmark_reports import compare_reports, public_report
from flashalign.evaluation_limits import LIMITED_PROTOCOL, UPPER_LIMITS, validate_limits


class EvaluationLimitTests(unittest.TestCase):
    setUp = fixtures.EvaluationTests.setUp
    client = fixtures.EvaluationTests.client
    run_subject = fixtures.EvaluationTests.run_subject
    judge_client = fixtures.EvaluationTests.judge_client
    judge_reply = fixtures.EvaluationTests.judge_reply
    respond = fixtures.EvaluationTests.respond

    def test_limits_are_complete_bounded_integers_and_not_generation_options(self):
        self.run_subject(limits=UPPER_LIMITS)
        run, _, _ = read_collection(self.root / 'run', registry=self.registry)
        self.assertEqual(run['protocol'], LIMITED_PROTOCOL)
        self.assertEqual(run['evaluation_limits'], UPPER_LIMITS)
        self.assertEqual(run['descriptor']['evaluation_limits'], UPPER_LIMITS)
        self.assertTrue(all(not (set(request) & set(UPPER_LIMITS)) for request in self.requests))
        for key in UPPER_LIMITS:
            for bad in (True, 0, -1, 1.0, '1', float('nan'), UPPER_LIMITS[key] + 1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_limits({**UPPER_LIMITS, key: bad})
        for value in ({}, {**UPPER_LIMITS, 'unknown': 1}, None):
            with self.assertRaises(ValueError):
                validate_limits(value)
        from flashalign.benchmark_protocol import validate_settings
        for name in ('evaluation_limits', *UPPER_LIMITS):
            with self.assertRaises(ValueError):
                validate_settings({'max_tokens': 32, name: 1})

    def test_caps_and_protocol_are_immutable_on_resume(self):
        self.run_subject(limits=UPPER_LIMITS)
        total = len(self.requests)
        for changed in (None, {**UPPER_LIMITS, 'max_events': 65535}):
            with self.assertRaises(ValueError):
                self.run_subject(limits=changed, resume=True)
        self.assertEqual(len(self.requests), total)
        self.run_subject(name='legacy')
        old, _, _ = read_collection(self.root / 'legacy', registry=self.registry)
        self.assertEqual(old['protocol'], 'flashalign.strict-completion.v1')
        self.assertNotIn('evaluation_limits', old)
        self.assertNotIn('collection_work', read_collection(self.root/'legacy', registry=self.registry)[2][0]['record'])

    def test_real_32768_event_stream_passes_explicit_limits_without_truncation(self):
        chunks = 32768
        def handler(request):
            self.requests.append(json.loads(request.content))
            common = {'model': 'model', 'object': 'chat.completion.chunk',
                      'created': 1793563200, 'id': 'chatcmpl-' + '0' * 32}
            row = {**common, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ' word'},
                                          'finish_reason': None, 'logprobs': None}]}
            text = ('data: ' + canonical(row) + '\n\n') * chunks
            text += 'data: ' + canonical({**common, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                                          'usage': {'completion_tokens': chunks, 'prompt_tokens': 4}}) + '\n\n'
            text += 'data: [DONE]\n\n'
            return httpx.Response(200, text=text, headers={'X-Owner-Mode-Effective': 'off'})
        with Client('http://subject.test/v1', 'model', transport=httpx.MockTransport(handler),
                    max_sse_event_bytes=UPPER_LIMITS['max_sse_event_bytes']) as client:
            result = collect(client, self.cases[0], 'off', {'max_tokens': chunks}, scale=1., limits=UPPER_LIMITS)
            self.assertEqual(result['status'], 'valid')
            self.assertEqual(result['answer'], ' word' * chunks)
            self.assertEqual(result['metrics']['completion_tokens'], chunks)
            self.assertEqual(result['collection_work']['events'], chunks + 1)
            self.assertGreater(result['collection_work']['response_bytes'], 4 * 1024**2)
            old = collect(client, self.cases[0], 'off', {'max_tokens': chunks}, scale=1.)
            self.assertEqual(old['error'], 'ResponseWorkLimit')

    def test_large_saved_records_remain_complete_through_judging_and_export(self):
        answer = 'x' * (9 * 1024**2)
        def handler(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            event = {'model': 'model', 'choices': [{'delta': {'content': answer}, 'finish_reason': 'stop'}],
                     'usage': {'completion_tokens': 32768, 'prompt_tokens': 4}}
            return httpx.Response(200, text='data: ' + canonical(event) + '\n\ndata: [DONE]\n\n',
                                  headers={'X-Owner-Mode-Effective': payload['owner_mode']})
        self.respond = handler
        self.run_subject(limits=UPPER_LIMITS)
        saved = self.root / 'run/attempts/000000/001.end.json'
        self.assertGreater(saved.stat().st_size, 16 * 1024**2)
        with self.assertRaises(ValueError):
            read_json(saved)  # Unrelated JSON readers retain their original ceiling.
        run, _, records = read_collection(self.root / 'run', registry=self.registry)
        self.assertTrue(all(row['record']['answer'] == answer for row in records))
        report = judge_run(self.root/'run', self.judge_config, self.root/'judged',
                           registry=self.registry, client_factory=self.judge_client)
        self.assertTrue(report['final_behavioral_gate_valid'])
        for request in self.judge_requests:
            evidence = json.loads(request['messages'][1]['content'].split('\n', 1)[1])
            self.assertEqual(evidence['candidate_response'], answer)
            self.assertEqual(request['max_tokens'], 1024)
        public = public_report(report)
        self.assertEqual(public['evaluation_limits'], UPPER_LIMITS)
        other = copy.deepcopy(report)
        other['evaluation_limits']['max_events'] -= 1
        with self.assertRaises(ValueError):
            compare_reports([report, other])

    def test_judge_rejects_oversized_evidence_without_request_or_truncation(self):
        self.run_subject(limits=UPPER_LIMITS)
        _, _, records = read_collection(self.root/'run', registry=self.registry)
        row = records[0]['record']
        original = row['answer']
        with self.judge_client() as client, self.assertRaises(ValueError):
            judge_once(client, self.judge_config, row, {}, limits={**UPPER_LIMITS, 'max_judge_request_bytes': 1})
        self.assertEqual(self.judge_requests, [])
        self.assertEqual(row['answer'], original)

    def test_stream_and_record_budgets_fail_closed(self):
        self.run_subject(name='byte-capped', limits={**UPPER_LIMITS, 'max_response_bytes': 1}, attempts=3)
        _, _, rows = read_collection(self.root/'byte-capped', registry=self.registry)
        self.assertTrue(all(row['record']['error'] == 'ResponseWorkLimit' for row in rows))
        self.assertTrue(all(len(row['attempts']) == 1 for row in rows))
        with self.assertRaises(ValueError):
            self.run_subject(name='record-capped', limits={**UPPER_LIMITS, 'max_record_bytes': 1})
        self.assertFalse((self.root/'record-capped/attempts/000000/001.end.json').exists())

    def test_header_limit_tampering_is_rejected(self):
        self.run_subject(limits=UPPER_LIMITS)
        path = self.root/'run/run.json'
        value = read_json(path)
        value['evaluation_limits']['max_events'] -= 1
        path.write_text(canonical(value))
        with self.assertRaises(ValueError):
            read_collection(self.root/'run', registry=self.registry)

    def test_sse_limit_is_utf8_bytes_when_explicit_and_event_budget_is_enforced(self):
        def handler(request):
            row = {'model': 'model', 'choices': [{'delta': {'content': 'é' * 60}, 'finish_reason': 'stop'}]}
            return httpx.Response(200, text='data: ' + canonical(row) + '\n\ndata: [DONE]\n\n',
                                  headers={'X-Owner-Mode-Effective': 'off'})
        with Client('http://subject.test/v1', 'model', transport=httpx.MockTransport(handler),
                    max_sse_event_bytes=150) as client:
            result = collect(client, self.cases[0], 'off', {'max_tokens': 128}, scale=1., limits=UPPER_LIMITS)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['error'], 'ValueError')
        self.run_subject(name='event-capped', limits={**UPPER_LIMITS, 'max_events': 1})
        _, _, rows = read_collection(self.root/'event-capped', registry=self.registry)
        self.assertTrue(all(row['record']['error'] == 'ResponseWorkLimit' for row in rows))


if __name__ == '__main__':
    unittest.main()
