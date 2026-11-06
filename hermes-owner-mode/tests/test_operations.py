# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Operations observe capability evidence without inventing or leaking it."""
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from test_setup import wizard
from owner_mode.client import Client
from flashalign.operations import doctor, rates, snapshot


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root/'client.json'
        self.config.write_text(json.dumps({'base_url': 'http://localhost/v1', 'model': 'model', 'profile': 'profile'}))
        self.requests = []

    def reply(self, request):
        self.requests.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'model'}]})
        if request.url.path.endswith('/owner-mode/profiles'):
            return httpx.Response(200, json={'data': [{'name': 'profile', 'loaded': True,
                                                     'key': 'DO-NOT-EXPORT'}]})
        return httpx.Response(404)

    def client(self, path):
        return Client.from_config(path, transport=httpx.MockTransport(self.reply))

    def test_default_status_does_not_generate_or_read_signing_key(self):
        result = snapshot(self.config, client_factory=self.client)
        self.assertTrue(result['reachable'])
        self.assertTrue(result['profile_loaded'])
        self.assertIsNone(result['effective_profile'])
        self.assertIsNone(result['runtime'])
        self.assertTrue(all(r.method == 'GET' for r in self.requests))
        self.assertTrue(all('X-Owner-Mode-Auth' not in r.headers for r in self.requests))
        self.assertNotIn('DO-NOT-EXPORT', json.dumps(result))
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_missing_signing_key_is_an_explicit_probe_failure_only(self):
        result = doctor(self.config, probe=True, client_factory=self.client)
        self.assertFalse(result['ok'])
        self.assertIn('activation_probe_failed', [x['code'] for x in result['issues']])
        self.assertFalse(result['activation_checks'][-1]['confirmed'])

    def test_rate_requires_matching_instance_and_monotonic_counters(self):
        old = {'sample_time': 1., 'runtime': {'instance_id': 'instance-a', 'model': 'm',
               'generation': 'g', 'output_tokens_total': 10, 'ttft_seconds_sum': 1.,
               'ttft_seconds_count': 2, 'draft_proposed': 10, 'draft_accepted': 5}}
        new = {'sample_time': 3., 'runtime': {**old['runtime'], 'output_tokens_total': 30,
               'ttft_seconds_sum': 1.5, 'ttft_seconds_count': 4, 'draft_proposed': 20, 'draft_accepted': 11}}
        self.assertEqual(rates(old, new), {'output_tokens_per_second': 10., 'mean_ttft_seconds': .25,
                                         'draft_acceptance': .6})
        new['runtime']['instance_id'] = 'instance-b'
        self.assertTrue(all(x is None for x in rates(old, new).values()))
        new['runtime']['instance_id'] = 'instance-a'
        new['runtime']['output_tokens_total'] = 0
        self.assertIsNone(rates(old, new)['output_tokens_per_second'])

    def test_unknown_data_is_not_treated_as_zero_or_numerical_attestation(self):
        def reply(request):
            if request.url.path.endswith('/flashalign/status'):
                return httpx.Response(200, text='{"schema":"flashalign.runtime-status.v1",'
                                      '"output_tokens_total":Infinity}')
            return self.reply(request)
        result = snapshot(self.config, client_factory=lambda p: Client.from_config(p, transport=httpx.MockTransport(reply)))
        # JSON with non-finite values is not a legal response document.
        self.assertIsNone(result['runtime'])


if __name__ == '__main__':
    unittest.main()
