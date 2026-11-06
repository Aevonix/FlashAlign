#!/usr/bin/env python3
"""Offline verification only. No network, model, accelerator or subprocess use.

Copyright (C) 2026 Aevonix Inc. SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    data = (ROOT / name).read_bytes()
    if name.endswith('.gz'):
        data = gzip.decompress(data)
    return json.loads(data)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_files():
    names = set()
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        need(name not in names and Path(name).name == name, 'Invalid inventory')
        names.add(name)
        path = ROOT / name
        need(not path.is_symlink() and path.is_file() and sha(path.read_bytes()) == digest,
             'Bundle hash mismatch: ' + name)
    need({'analyze.py', 'study_contract.py', 'memory_sampler.py', 'analysis-input.json',
          'expected-analysis.json', 'workload.json', 'telemetry.json.gz', 'provenance.json'} <= names,
         'Incomplete bundle')


def verify_observations(data, telemetry, provenance, workload, sampler):
    rows = data['processes'] + data['bare_stock']
    processes = {r['process_id']: r for r in rows}
    sequence = provenance['sequence']
    need(len(rows) == len(processes) == len(sequence) == 12 and len(set(sequence)) == 12,
         'Exactly twelve assigned starts required')
    need(set(processes) == set(sequence) == set(telemetry['processes']), 'Process assignment')
    need(data['arm_order'] == provenance['arm_order'] == [b['arm_order'] for b in workload['blocks']], 'Order binding')
    need(data['order_draw_sha256'] == provenance['order_draw_sha256']
         and data['manifest_sha256'] == provenance['original_manifest_sha256'], 'Manifest binding')
    bindings = {r['process_id']: r for r in provenance['process_bindings']}
    need(set(bindings) == set(processes) and len({b['process_birth'] for b in bindings.values()}) == 12,
         'Distinct exported process identities')
    counters = {'starts': 12, 'measured_windows': 0, 'measured_requests': 0,
                'host_samples': 0, 'memory_samples': 0, 'memory_eligible_ticks': 0,
                'warmup_requests': 0, 'per_window_host_checks': 0}
    for name in sequence:
        p, t, b = processes[name], telemetry['processes'][name], bindings[name]
        need(b['host'] == 'host-A' and b['device'] == 'device-A', 'Device equality')
        for key in ('arm', 'block', 'control_slot', 'order'):
            need(p[key] == b[key], 'Process binding')
        counters['warmup_requests'] += t['warmup_requests']
        if p['arm'] == 'B':
            need(type(t['primary_graph_batches']) is int and t['primary_graph_batches'] > 0, 'Graph evidence')
        expected_conditions = {'primary'} if p['arm'] == 'stock' else {
            'primary', 'short_cold', 'long_cold', 'primary_warm_prefix', 'primary_concurrency_two', 'loaded_off'}
        need({k for k, v in p['workloads'].items() if v} == expected_conditions, 'Condition assignment')
        proposed = accepted = 0
        for condition in expected_conditions:
            windows = p['workloads'][condition]
            order = workload['blocks'][p['block'] - 1]['case_order']
            need([w['case_id'] for w in windows] == [f"b{p['block']}-measured-{condition}-{i:02}" for i in order],
                 'Case order changed')
            counters['measured_windows'] += len(windows)
            for window in windows:
                need(len(window['requests']) == (2 if condition == 'primary_concurrency_two' else 1), 'Member count')
                for request in window['requests']:
                    counters['measured_requests'] += 1
                    m = request['response_metadata']
                    active = p['arm'] != 'stock' and condition != 'loaded_off'
                    need(m['status_code'] == 200 and m['effective_profile'] == (
                        provenance['profile_name'] if active else 'off'), 'Profile binding')
                    if p['arm'] == 'B' and active:
                        need(m['generation'] == b['generation'] and
                             m['profile_digest'] == provenance['canonical_profile_digest'], 'Generation binding')
                    shape = condition if condition in ('short_cold', 'long_cold') else 'primary'
                    need(request['prompt_tokens'] == workload['shapes'][shape]['input_tokens'], 'Prompt count')
                    if condition != 'loaded_off':
                        proposed += request['speculative']['spec_num_proposed_drafts']
                        accepted += request['speculative']['spec_num_correct_drafts']
        need(proposed == t['counters']['draft_proposed'] > 0 and
             accepted == t['counters']['draft_accepted'] > 0, 'Speculation counts')
        host = t['host']
        need(host['passed'] is True and host['error'] is None and host['sample_hz'] == 1
             and host['max_gap_seconds'] == 2 and host['samples'] == len(host['rows']) >= 2, 'Host telemetry')
        need(max(b['at_monotonic'] - a['at_monotonic'] for a, b in zip(host['rows'], host['rows'][1:])) <= 2,
             'Host telemetry gap')
        for sample in host['rows'] + t['quiet_checks']:
            need(sample['scope'] == 'quiet_host' and sample['background_absence_verified'] is True
                 and sample['engine_tree_verified'] is True and sample['unapproved_compute_processes'] == 0
                 and all(sample['compute_process_counts'][d] == 0 for d in ('device-B', 'device-C', 'device-D')),
                 'Quiet host evidence')
        counters['host_samples'] += len(host['rows'])
        counters['per_window_host_checks'] += len(t['quiet_checks'])
        if p['arm'] != 'stock':
            mem = t['memory']
            need(sampler.validate_v2_receipt(mem, max_gap_seconds=.25, minimum_paired_coverage=.99), 'Memory adequacy')
            need(mem['domains'] == p['memory'], 'Memory summary differs from samples')
            counters['memory_samples'] += len(mem['samples'])
            counters['memory_eligible_ticks'] += mem['eligible_scheduled_ticks']
    need(counters['measured_requests'] == 1440 and counters['measured_windows'] == 1260, 'Fixed sample totals')
    return counters


def verify():
    verify_files()
    load('study_contract')
    analyzer, sampler = load('analyze'), load('memory_sampler')
    data, telemetry = read('analysis-input.json'), read('telemetry.json.gz')
    provenance, workload = read('provenance.json'), read('workload.json')
    counts = verify_observations(data, telemetry, provenance, workload, sampler)
    report = analyzer.analyze(data)
    need(report == read('expected-analysis.json'), 'Computed report differs from reference')
    gates = [report['primary_throughput_gate'], *report['memory_gates'].values()]
    gates.extend(g for metrics in report['latency_gates'].values() for g in metrics.values())
    need(len(gates) == 13, 'Thirteen intervals required')
    return {'exact_report_match': True, 'recomputed_intervals': len(gates), **counts,
            'classifications': [g['status'] for g in gates],
            'combined_timing_memory_gate': report['combined_timing_memory_gate'],
            'completion_regression_gate': report['completion_regression_gate']['status']}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True, indent=2))
