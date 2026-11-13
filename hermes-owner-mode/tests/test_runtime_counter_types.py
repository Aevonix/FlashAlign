# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Runtime counter payloads distinguish integer counts from elapsed seconds."""
import json
import sys
import unittest
from unittest.mock import patch

from test_runtime_status_counters import registry
from owner_mode.profile_admin_server import runtime_counters


class RuntimeCounterTypeTests(unittest.TestCase):
    def test_payload_counts_are_integers_and_seconds_are_float(self):
        for engine in ('vllm', 'sglang'):
            for values, expected in ((([10., 20.], [.25, .5], [1., 2.]), (30, .75, 3)),
                                     (([0.], [0.], [0.]), (0, 0., 0)),
                                     (([10], [1], [2]), (10, 1., 2))):
                with self.subTest(engine=engine, values=values):
                    with patch.dict(sys.modules, {'prometheus_client': registry(engine, values)}):
                        payload = json.loads(json.dumps(runtime_counters(engine)))
                    self.assertEqual(tuple(payload.values()), expected)
                    self.assertIs(type(payload['output_tokens_total']), int)
                    self.assertIs(type(payload['ttft_seconds_count']), int)
                    self.assertIs(type(payload['ttft_seconds_sum']), float)
