"""Portable regression checks. Mutations are synthetic, never new evidence.

Copyright (C) 2026 Aevonix Inc. SPDX-License-Identifier: AGPL-3.0-only
"""
import copy
import unittest

import recompute

recompute.verify_files()
recompute.load('study_contract')
analyze = recompute.load('analyze')
original = recompute.load('analyze_original')
sampler = recompute.load('memory_sampler')
DATA = recompute.read('analysis-input.json')


class PortableEvidence(unittest.TestCase):
    def test_exact_offline_recomputation(self):
        result = recompute.verify()
        self.assertTrue(result['exact_report_match'])
        self.assertEqual(result['recomputed_intervals'], 13)
        self.assertEqual(result['measured_requests'], 1440)
        self.assertEqual(result['starts'], 12)
        self.assertEqual(result['classifications'], ['pass'] * 13)

    def test_original_order_bug_preserved(self):
        self.assertEqual(original.analyze(DATA)['combined_timing_memory_gate'], 'inconclusive')
        self.assertEqual(analyze.analyze(DATA)['combined_timing_memory_gate'], 'pass')

    def test_missing_and_wrong_order_fail(self):
        for kind in ('missing', 'wrong'):
            data = copy.deepcopy(DATA)
            if kind == 'missing':
                data.pop('arm_order')
            else:
                data['processes'][0]['order'] = 2
            self.assertEqual(analyze.analyze(data)['combined_timing_memory_gate'], 'inconclusive')

    def test_missing_error_and_misaligned_requests_fail(self):
        for kind in ('missing', 'error', 'order'):
            data = copy.deepcopy(DATA)
            windows = data['processes'][0]['workloads']['primary']
            if kind == 'missing':
                windows.pop()
            elif kind == 'error':
                windows[0]['requests'][0]['status'] = 'error'
            else:
                windows.reverse()
            self.assertEqual(analyze.analyze(data)['combined_timing_memory_gate'], 'inconclusive')

    def test_missing_stock_control_fails(self):
        data = copy.deepcopy(DATA)
        data['bare_stock'].pop()
        self.assertEqual(analyze.analyze(data)['combined_timing_memory_gate'], 'inconclusive')

    def test_inadequate_memory_fails(self):
        data = copy.deepcopy(DATA)
        data['processes'][0]['memory']['gpu_used']['measurement_adequate'] = False
        self.assertEqual(analyze.analyze(data)['combined_timing_memory_gate'], 'inconclusive')

    def test_memory_sample_change_rejected(self):
        mem = recompute.read('telemetry.json.gz')['processes']['b1-A']['memory']
        mem['samples'][0]['gpu_used_bytes'] = 2**60
        with self.assertRaises(ValueError):
            sampler.validate_v2_receipt(mem, max_gap_seconds=.25, minimum_paired_coverage=.99)

    def test_inclusive_five_percent_boundaries(self):
        self.assertEqual(analyze.classify_interval({'lower_95': 1.05, 'upper_95': 1.1}, 'higher'), 'pass')
        self.assertEqual(analyze.classify_interval({'lower_95': 1., 'upper_95': 1.05}, 'lower'), 'pass')
        for direction in ('higher', 'lower'):
            self.assertEqual(analyze.classify_interval({'lower_95': 1.04, 'upper_95': 1.06}, direction), 'inconclusive')

    def test_synthetic_workload_token_counts(self):
        workload = recompute.read('workload.json')
        for name, count in [('short_cold', 33), ('primary', 258), ('long_cold', 8193)]:
            self.assertEqual(workload['shapes'][name]['input_tokens'], count)
            self.assertEqual(len(workload['shapes'][name]['rendered_token_ids']), count)


if __name__ == '__main__':
    unittest.main()
