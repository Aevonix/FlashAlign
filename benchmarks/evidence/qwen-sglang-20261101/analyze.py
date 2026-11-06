#!/usr/bin/env python3
"""Analyze the sealed, private Qwen/SGLang performance experiment; no execution.

Neutral input schema ``flashalign.private-performance-input.v1``::

  {"schema": "flashalign.private-performance-input.v1",
   "protocol_sha256": "<64 lowercase hex>", "manifest_sha256": "<64 lowercase hex>",
   "latency_metrics": ["ttft", "e2e"],
   "memory_domains": [{"name": "gpu_used", "kind": "sampled", "max_gap_seconds": 0.1},
                      {"name": "allocator_peak", "kind": "counter"}],
   "checks": {<every key in EVIDENCE_CHECKS>: true},
   "assumptions": {"independent_blocks": true, "approximately_normal_log_ratios": true},
   "processes": [{"process_id": "b1-a", "block": 1, "arm": "A", "order": 1,
                  "status": "complete", "checks": {<PROCESS_CHECKS>: true},
                  "workloads": {<each WORKLOAD>: [<30 windows>], "loaded_off": [<30 windows>]},
                  "memory": {"gpu_used": {"peak_bytes": 1, "measurement_adequate": true,
                    "sample_hz": 20, "sample_count": 2, "max_gap_seconds": 0.05},
                    "allocator_peak": {"peak_bytes": 1, "measurement_adequate": true}}}],
   "bare_stock": [{"process_id": "stock-1", "block": 1, "status": "complete",
                   "checks": {<PROCESS_CHECKS>: true}, "workloads": {"primary": [<30 windows>]}}]}

Each window is {case_id, t_send, requests:[request]}; concurrency-two has two
members. A request is {status:"complete"|"error"|"interrupted", t_send, t_first,
t_done, completion_tokens, finish_reason, transport_complete, final_usage,
cached_tokens, cache_evidence}. Tokens must come from actual final usage, and
cache_evidence is "usage", "raw_engine", or "verified_build_rule". Timestamps
are seconds on one collector monotonic clock. Window t_send is the common
barrier timestamp; individual t_send remains the actual transmission timestamp.

Memory gap tolerances come from the sealed manifest, never chosen here after
seeing measurements. measurement_adequate additionally records the sampler's
coverage assessment. Sampled maxima and allocator counters remain distinct.
Checks are receipt-backed assertions, not substitutes for the retained source
receipts. This tool neither manufactures these assertions nor verifies GPUs.

Missing, errored, mispaired or inadequate data makes affected gates inconclusive.
Loaded-off/bare-stock controls are descriptive and never pooled with owner arms.
No retry, request bootstrap, data exclusion or optional extension is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
from study_contract import execution_mode, orders, required_check, stationarity as check_stationarity, ISOLATED, FRANCE, FRANCE_QUIET


SCHEMA = 'flashalign.private-performance-input.v1'
REPORT_SCHEMA = 'flashalign.private-performance-report.v1'
T_CRITICAL = 4.302652729911275  # two-sided 95%, df = 2; fixed by protocol
THRESHOLD = 1.05
BLOCKS = (1, 2, 3)
WORKLOADS = ('primary', 'short_cold', 'long_cold', 'primary_warm_prefix', 'primary_concurrency_two')
LATENCY_METRICS = ('ttft', 'e2e')
EVIDENCE_CHECKS = ('sealed_manifest_verified', 'numerical_attestation_passed',
                   'candidate_graph_exercised', 'no_new_fixture_failure')
PROCESS_CHECKS = ('identity_verified', 'fixed_controls_verified', 'warmup_verified',
                  'no_overlap', 'diagnostics_off', 'speculation_observed',
                  'cache_policy_verified', 'intended_execution_verified')
HASH = re.compile(r'[0-9a-f]{64}\Z')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z')
MAX_INPUT_BYTES = 64 * 1024 * 1024


def _number(value, *, positive=False):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        return False


def nearest_rank_p95(values):
    """Nearest-rank p95, rank ceil(.95*n), without interpolation."""
    if not values or any(not _number(value) for value in values):
        raise ValueError('A finite nonnegative sample is required')
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def paired_interval(ratios):
    """The exact preregistered three-process-block log-ratio t interval."""
    if len(ratios) != 3 or any(not _number(value, positive=True) for value in ratios):
        raise ValueError('Exactly three positive finite block ratios are required')
    logs = [math.log(value) for value in ratios]
    mean = statistics.mean(logs)
    sd = statistics.stdev(logs)  # sample standard deviation (n-1 denominator)
    se = sd / math.sqrt(3)
    try:
        lower, point, upper = math.exp(mean - T_CRITICAL * se), math.exp(mean), math.exp(mean + T_CRITICAL * se)
    except OverflowError as error:
        raise ValueError('Ratio interval exceeds finite arithmetic') from error
    if not all(math.isfinite(value) and value > 0 for value in (lower, point, upper)):
        raise ValueError('Ratio interval exceeds finite arithmetic')
    return {'point_ratio': point, 'lower_95': lower, 'upper_95': upper,
            'mean_log_ratio': mean, 'sample_sd_log_ratio': sd, 'standard_error_log_ratio': se,
            'df': 2, 't_critical': T_CRITICAL, 'process_blocks': 3}


def classify_interval(interval, direction):
    """Equality passes; an interval crossing the boundary stays inconclusive."""
    if direction == 'higher':
        if interval['lower_95'] >= THRESHOLD:
            return 'pass'
        if interval['upper_95'] < THRESHOLD:
            return 'fail'
    elif direction == 'lower':
        if interval['upper_95'] <= THRESHOLD:
            return 'pass'
        if interval['lower_95'] > THRESHOLD:
            return 'fail'
    else:
        raise ValueError('Unknown interval direction')
    return 'inconclusive'


def _request(row, workload, barrier):
    if not isinstance(row, dict):
        return None, 'failed'
    if row.get('status') == 'interrupted':
        return None, 'indeterminate'
    if row.get('status') != 'complete':
        return None, 'failed'
    times = [row.get(name) for name in ('t_send', 't_first', 't_done')]
    if any(not _number(value) or value > 10**15 for value in times):
        return None, 'failed'
    sent, first, done = times
    if not sent <= first < done or not _number(barrier) or barrier > sent:
        return None, 'failed'
    if (type(row.get('completion_tokens')) is not int or row['completion_tokens'] != 256
            or row.get('finish_reason') not in ('stop', 'length')
            or row.get('transport_complete') is not True or row.get('final_usage') is not True):
        return None, 'failed'
    cached = row.get('cached_tokens')
    if (type(cached) is not int or cached < 0
            or row.get('cache_evidence') not in ('usage', 'raw_engine', 'verified_build_rule')
            or (cached <= 0 if workload == 'primary_warm_prefix' else cached != 0)):
        return None, 'failed'
    return {'tokens': row['completion_tokens'], 'ttft': first - sent,
            'e2e': done - sent, 'done': done}, None


def _workload(windows, name):
    expected_members = 2 if name == 'primary_concurrency_two' else 1
    rows = windows if isinstance(windows, list) else []
    issues = []
    if not isinstance(windows, list) or len(rows) != 30:
        issues.append('Exactly 30 assigned windows are required')
    ids, good, complete_windows = [], [], []
    attempted = failed = indeterminate = 0
    for window in rows:
        if not isinstance(window, dict):
            issues.append('Malformed window')
            continue
        case = window.get('case_id')
        if not isinstance(case, str) or not ID.fullmatch(case):
            issues.append('Invalid case identity')
        ids.append(case if isinstance(case, str) and ID.fullmatch(case) else None)
        members = window.get('requests')
        if not isinstance(members, list):
            issues.append('Missing request ledger')
            continue
        if len(members) != expected_members:
            issues.append('Window member count differs from the protocol')
        successes = []
        for row in members:
            attempted += 1
            value, error = _request(row, name, window.get('t_send'))
            if error == 'failed':
                failed += 1
            elif error == 'indeterminate':
                indeterminate += 1
            else:
                successes.append(value)
                good.append(value)
        if len(successes) == expected_members and len(members) == expected_members:
            duration = max(value['done'] for value in successes) - window['t_send']
            if duration > 0:
                # Singles use exact request send-to-done time, not setup time
                # before transmission; pairs use the common release barrier.
                duration = successes[0]['e2e'] if expected_members == 1 else duration
                complete_windows.append((sum(value['tokens'] for value in successes), duration))
    if len(ids) != len(set(value for value in ids if isinstance(value, str))):
        issues.append('Case identities are missing or repeated')
    expected_requests = 30 * expected_members
    if failed or indeterminate:
        issues.append('Failed or indeterminate requests remain in the ledger')
    if attempted != expected_requests or len(good) != expected_requests or len(complete_windows) != 30:
        issues.append('Fixed sample is incomplete')
    tokens = sum(value['tokens'] for value in good)
    seconds = sum(value['e2e'] for value in good)
    window_tokens = sum(value[0] for value in complete_windows)
    window_seconds = sum(value[1] for value in complete_windows)
    return {'expected_windows': 30, 'observed_windows': len(rows), 'expected_requests': expected_requests,
            'attempted': attempted, 'successful': len(good), 'failed': failed, 'indeterminate': indeterminate,
            'missing': max(0, expected_requests - attempted), 'extra': max(0, attempted - expected_requests),
            'completed_windows': len(complete_windows), 'completion_tokens': tokens,
            'request_send_to_done_seconds': seconds,
            'window_completion_tokens': window_tokens, 'window_seconds': window_seconds,
            'throughput_tokens_per_second': window_tokens / window_seconds if window_seconds else None,
            'ttft_p95_seconds': nearest_rank_p95([value['ttft'] for value in good]) if good else None,
            'e2e_p95_seconds': nearest_rank_p95([value['e2e'] for value in good]) if good else None,
            'latency_member_count': len(good), 'ordered_case_ids': ids,
            'clean': not issues, 'issues': sorted(set(issues))}


def _memory(raw, declarations):
    values, issues = {}, []
    if not isinstance(raw, dict):
        raw = {}
    for domain in declarations:
        name = domain['name']
        value = raw.get(name)
        valid = isinstance(value, dict)
        peak = value.get('peak_bytes') if valid else None
        valid = valid and type(peak) is int and 0 < peak < 2**63 and value.get('measurement_adequate') is True
        row = {'peak_bytes': peak if type(peak) is int and 0 < peak < 2**63 else None,
               'kind': domain['kind'], 'measurement_adequate': bool(valid)}
        if domain['kind'] == 'sampled':
            rate = value.get('sample_hz') if isinstance(value, dict) else None
            count = value.get('sample_count') if isinstance(value, dict) else None
            gap = value.get('max_gap_seconds') if isinstance(value, dict) else None
            valid = valid and rate == 20 and type(count) is int and 2 <= count <= 10**12 and _number(gap)
            valid = valid and gap <= domain['max_gap_seconds']
            if domain.get('coverage_protocol') == 'request-stop-coverage-v2':
                eligible=value.get('eligible_scheduled_ticks') if isinstance(value,dict) else None
                paired=value.get('paired_valid_samples') if isinstance(value,dict) else None
                coverage=value.get('paired_coverage') if isinstance(value,dict) else None
                minimum=domain.get('minimum_paired_coverage')
                valid=valid and (type(eligible) is int and type(paired) is int and 2<=paired<=count<=eligible
                    and _number(coverage) and _number(minimum) and 0<minimum<=1
                    and abs(coverage-paired/eligible)<1e-15 and coverage>=minimum
                    and value.get('minimum_paired_coverage')==minimum and value.get('error_count')==0
                    and value.get('coverage_protocol')=='request-stop-coverage-v2')
                row.update(eligible_scheduled_ticks=eligible,paired_valid_samples=paired,paired_coverage=coverage,
                           minimum_paired_coverage=minimum)
            row.update(sample_hz=rate if _number(rate) else None,
                       sample_count=count if type(count) is int and 0 <= count <= 10**12 else None,
                       max_gap_seconds=gap if _number(gap) else None,
                       permitted_max_gap_seconds=domain['max_gap_seconds'], measurement_adequate=bool(valid))
        row['clean'] = bool(valid)
        if not valid:
            issues.append(name + ': missing or inadequate steady-memory measurement')
        values[name] = row
    return values, issues


def _process(raw, domains, *, stock=False, mode=ISOLATED):
    if not isinstance(raw, dict):
        raw = {}
    issues = []
    process_id = raw.get('process_id')
    if not isinstance(process_id, str) or not ID.fullmatch(process_id):
        process_id = None
        issues.append('Missing unique process identity')
    if raw.get('status') != 'complete':
        issues.append('Process attempt did not complete')
    checks = raw.get('checks') if isinstance(raw.get('checks'), dict) else {}
    process_checks = tuple(required_check({'execution_mode': mode}) if name == 'no_overlap' else name
                           for name in PROCESS_CHECKS)
    for check in process_checks:
        if checks.get(check) is not True:
            issues.append('Unverified process check: ' + check)
    workloads = raw.get('workloads') if isinstance(raw.get('workloads'), dict) else {}
    names = ('primary',) if stock else (*WORKLOADS, 'loaded_off')
    measured = {name: _workload(workloads.get(name), name) for name in names}
    memory, memory_issues = _memory(raw.get('memory'), domains) if not stock else ({}, [])
    return {'process_id': process_id,
            'block': raw.get('block') if type(raw.get('block')) is int and raw['block'] in BLOCKS else None,
            'arm': raw.get('arm') if raw.get('arm') in ('A', 'B', 'stock') else None,
            'order': raw.get('order') if type(raw.get('order')) is int and raw['order'] in (1, 2) else None,
            'control_slot': raw.get('control_slot'),
            'clean': not issues, 'issues': issues, 'workloads': measured,
            'memory': memory, 'memory_issues': memory_issues}


def _gate(pairs, direction, issues):
    ratios, blocks = [], []
    for block, a, b in pairs:
        ratio = b / a if _number(a, positive=True) and _number(b, positive=True) else None
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
        blocks.append({'block': block, 'A': a, 'B': b, 'ratio_B_over_A': ratio})
        if ratio is not None:
            ratios.append(ratio)
    problems = list(issues)
    if len(pairs) != 3 or len(ratios) != 3:
        problems.append('All three paired process-block metrics are required')
    interval = None
    if not problems:
        try:
            interval = paired_interval(ratios)
        except ValueError:
            problems.append('The interval cannot be represented with finite arithmetic')
    return {'status': classify_interval(interval, direction) if interval is not None else 'inconclusive',
            'direction': direction, 'threshold_B_over_A': THRESHOLD, 'blocks': blocks,
            'interval': interval, 'issues': sorted(set(problems))}


def _combine(statuses):
    if 'fail' in statuses:
        return 'fail'
    return 'pass' if statuses and all(value == 'pass' for value in statuses) else 'inconclusive'


def analyze(document):
    """Return a numerical report; malformed/missing samples never become passes."""
    if not isinstance(document, dict):
        document = {}
    common = []
    try:
        mode = execution_mode(document)
    except ValueError:
        mode = ISOLATED
        common.append('Unknown study execution mode')
    try:
        expected_orders = orders(document)
    except (ValueError, TypeError):
        expected_orders = None
        common.append('Missing or invalid preregistered arm order')
    if document.get('schema') != SCHEMA:
        common.append('Input schema is missing or unsupported')
    for key in ('manifest_sha256', 'protocol_sha256'):
        if not isinstance(document.get(key), str) or not HASH.fullmatch(document[key]):
            common.append('Missing sealed ' + key)
    if document.get('latency_metrics') not in (list(LATENCY_METRICS), list(reversed(LATENCY_METRICS))):
        common.append('Both TTFT and end-to-end latency metrics must be declared')
    checks = document.get('checks') if isinstance(document.get('checks'), dict) else {}
    for name in EVIDENCE_CHECKS:
        if checks.get(name) is not True:
            common.append('Unverified evidence check: ' + name)
    assumptions = document.get('assumptions') if isinstance(document.get('assumptions'), dict) else {}
    for name in ('independent_blocks', 'approximately_normal_log_ratios'):
        if assumptions.get(name) is not True:
            common.append('Unsupported inference assumption: ' + name)
    declarations, names = [], set()
    domains = document.get('memory_domains')
    if not isinstance(domains, list) or not domains:
        common.append('Steady-memory domains must be preregistered')
    else:
        for row in domains:
            if (not isinstance(row, dict) or not isinstance(row.get('name'), str)
                    or not ID.fullmatch(row['name']) or row['name'] in names
                    or row.get('kind') not in ('sampled', 'counter')
                    or row.get('kind') == 'sampled' and not _number(row.get('max_gap_seconds'), positive=True)):
                common.append('Invalid or duplicate steady-memory declaration')
                continue
            names.add(row['name'])
            declarations.append(row)
    raw_processes = document.get('processes')
    rows = raw_processes if isinstance(raw_processes, list) else []
    if len(rows) != 6:
        common.append('Exactly six assigned owner process starts are required')
    processes = [_process(row, declarations, mode=mode) for row in rows]
    ids = [row['process_id'] for row in processes]
    if None in ids or len(set(ids)) != len(ids):
        common.append('Owner process identities must be unique')
    paired = {}
    for row in processes:
        block, arm = row['block'], row['arm']
        if type(block) is not int or block not in BLOCKS or arm not in ('A', 'B'):
            common.append('Unexpected owner arm or process block')
            continue
        expected_order = expected_orders[block].index(arm) + 1 if expected_orders is not None else None
        if type(row['order']) is not int or row['order'] != expected_order:
            row['issues'].append('Process arm order differs from preregistration')
            row['clean'] = False
        if (block, arm) in paired:
            common.append('An assigned owner process was repeated')
        else:
            paired[block, arm] = row
    if len(paired) != 6:
        common.append('Each process block requires exactly one A and one B start')

    def workload_pairs(name, metric):
        values, issues = [], list(common)
        for block in BLOCKS:
            a, b = paired.get((block, 'A')), paired.get((block, 'B'))
            av = a['workloads'][name] if a else None
            bv = b['workloads'][name] if b else None
            if a is None or b is None or not a['clean'] or not b['clean']:
                issues.append('Block ' + str(block) + ': process evidence is incomplete')
            if av is None or bv is None or not av['clean'] or not bv['clean']:
                issues.append('Block ' + str(block) + ': workload sample is incomplete')
            elif av['ordered_case_ids'] != bv['ordered_case_ids']:
                issues.append('Block ' + str(block) + ': matched case order differs')
            values.append((block, av.get(metric) if av else None, bv.get(metric) if bv else None))
        return values, issues

    values, issues = workload_pairs('primary', 'throughput_tokens_per_second')
    throughput = _gate(values, 'higher', issues)
    latency = {}
    for name in WORKLOADS:
        latency[name] = {}
        for metric in LATENCY_METRICS:
            values, issues = workload_pairs(name, metric + '_p95_seconds')
            latency[name][metric] = _gate(values, 'lower', issues)
    memory = {}
    for domain in declarations:
        name = domain['name']
        values, issues = [], list(common)
        for block in BLOCKS:
            a, b = paired.get((block, 'A')), paired.get((block, 'B'))
            av = a['memory'].get(name) if a else None
            bv = b['memory'].get(name) if b else None
            if (a is None or b is None or not a['clean'] or not b['clean']
                    or av is None or bv is None or not av['clean'] or not bv['clean']):
                issues.append('Block ' + str(block) + ': memory or process evidence is inadequate')
            values.append((block, av.get('peak_bytes') if av else None, bv.get('peak_bytes') if bv else None))
        memory[name] = _gate(values, 'lower', issues)
        memory[name]['measurement_kind'] = domain['kind']

    fixture = ('pass' if checks.get('no_new_fixture_failure') is True else
               'fail' if checks.get('no_new_fixture_failure') is False else 'inconclusive')
    statuses = [throughput['status'], fixture, *(row['status'] for workload in latency.values() for row in workload.values()),
                *(row['status'] for row in memory.values())]
    loaded = [{'process_id': row['process_id'], 'block': row['block'], 'arm': row['arm'],
               **row['workloads']['loaded_off']} for row in processes]
    raw_stock = document.get('bare_stock')
    stock = [_process(row, [], stock=True, mode=mode) for row in raw_stock] if isinstance(raw_stock, list) else []
    control_issues = []
    if len(loaded) != 6 or any(not row['clean'] for row in loaded) or any(not row['clean'] for row in processes):
        control_issues.append('Loaded-off controls are incomplete')
    stationarity = []
    if mode in (FRANCE, FRANCE_QUIET):
        expected_slots = {(block, slot) for block in BLOCKS for slot in ('before', 'after')}
        actual_slots = [(row['block'], row['control_slot']) for row in stock]
        if len(stock) != 6 or len(set(actual_slots)) != 6 or set(actual_slots) != expected_slots:
            control_issues.append('Exactly six paired before/after bare-stock starts are required')
        indexed = {(row['block'], row['control_slot']): row for row in stock}
        for block in BLOCKS:
            before = indexed.get((block, 'before'), {}).get('workloads', {}).get('primary', {})
            after = indexed.get((block, 'after'), {}).get('workloads', {}).get('primary', {})
            stationarity.append({'block': block, **check_stationarity(before, after)})
        if not all(row['passed'] for row in stationarity):
            control_issues.append('The fixed ten-percent stationarity check did not pass for every block')
    elif len(stock) != 3 or {row['block'] for row in stock if type(row['block']) is int} != set(BLOCKS):
        control_issues.append('Exactly three separate bare-stock starts are required')
    stock_ids = [row['process_id'] for row in stock]
    if any(not row['clean'] or not row['workloads']['primary']['clean'] for row in stock):
        control_issues.append('Bare-stock process or request evidence is incomplete')
    if (None in stock_ids or len(stock_ids) != len(set(stock_ids)) or set(stock_ids) & set(ids)):
        control_issues.append('Control process identities must be separate from owner starts')
    controls_status = 'inconclusive' if control_issues else 'complete'
    counts_by_arm = {}
    for arm in ('A', 'B'):
        counts_by_arm[arm] = {}
        for name in WORKLOADS:
            relevant = [row['workloads'][name] for row in processes if row['arm'] == arm]
            assigned = 90 * (2 if name == 'primary_concurrency_two' else 1)
            totals = {key: sum(row[key] for row in relevant)
                      for key in ('attempted', 'successful', 'failed', 'indeterminate')}
            totals.update(assigned=assigned, missing=max(0, assigned - totals['attempted']),
                          extra=max(0, totals['attempted'] - assigned))
            denominator = totals['attempted']
            totals['failure_rate_per_attempt'] = totals['failed'] / denominator if denominator else None
            totals['indeterminate_rate_per_attempt'] = totals['indeterminate'] / denominator if denominator else None
            counts_by_arm[arm][name] = totals
    return {'schema': REPORT_SCHEMA, 'execution_mode': mode,
            'hardware_scope': 'Declared concurrent France host; not isolated-server performance' if mode == FRANCE else 'Quiet France host with paired stock controls' if mode == FRANCE_QUIET else 'Isolated host',
            'protocol_sha256': document.get('protocol_sha256') if isinstance(document.get('protocol_sha256'), str)
                and HASH.fullmatch(document['protocol_sha256']) else None,
            'manifest_sha256': document.get('manifest_sha256') if isinstance(document.get('manifest_sha256'), str)
                and HASH.fullmatch(document['manifest_sha256']) else None,
            'definitions': {
                'primary_throughput': 'sum(actual final completion tokens) / sum(request t_done - t_send)',
                'concurrency_throughput': 'sum(actual tokens in complete two-request windows) / sum(last t_done - common window t_send)',
                'p95': 'nearest rank ceil(0.95*n), no interpolation; all 60 member request latencies for concurrency-two',
                'inference_unit': 'three independent paired process blocks, not individual requests or windows',
                'interval': 'exp(mean(log(B/A)) +/- 4.302652729911275 * sample_sd(log(B/A))/sqrt(3)); two-sided 95%, df=2',
                'memory': 'declared steady-window peak bytes, with sampled maxima and allocator counters kept separate',
                'successful_request': '256 actual completion tokens, final usage, complete transport and valid finish, timing and cache evidence',
                'incomplete_metrics': 'successful-request/window metrics are descriptive only; incomplete samples cannot pass'},
            'assumptions': {'required': ['independent process blocks', 'approximately normal process-block log-ratios'],
                            'limitation': 'Three process pairs provide weak evidence and do not validate normality or global tail probabilities.',
                            'supplied': {name: assumptions.get(name) if type(assumptions.get(name)) is bool else None
                                         for name in ('independent_blocks', 'approximately_normal_log_ratios')}},
            'input_issues': sorted(set(common)), 'processes': processes, 'counts_by_arm': counts_by_arm,
            'primary_throughput_gate': throughput, 'latency_gates': latency, 'memory_gates': memory,
            'deterministic_fixture_gate': fixture,
            'combined_timing_memory_gate': ('inconclusive' if mode in (FRANCE, FRANCE_QUIET) and control_issues else _combine(statuses)),
            'controls': {'status': controls_status, 'issues': control_issues, 'loaded_off': loaded, 'bare_stock': stock,
                         'pooled_into_owner_estimate': False, 'stationarity': stationarity},
            'completion_regression_gate': {'status': 'not_established', 'margin_percentage_points': 1,
                'reason': 'This three-block timing design does not establish the separate one-percentage-point completion-regression confidence gate.'},
            'automatic_reruns': False, 'request_bootstrap': False}


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate input field')
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError('Nonfinite JSON input')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def _read_input(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_INPUT_BYTES:
            raise ValueError('Input must be a bounded regular file')
        chunks, remaining = [], MAX_INPUT_BYTES + 1
        while remaining:
            value = os.read(fd, min(65536, remaining))
            if not value:
                break
            chunks.append(value)
            remaining -= len(value)
        data = b''.join(chunks)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError('Input exceeds the bounded private report size')
        return data
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = _read_input(args.input)
    report = analyze(_decode(data))
    report['input_sha256'] = hashlib.sha256(data).hexdigest()
    report['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'report_written': True, 'timing_memory_gate': report['combined_timing_memory_gate'],
                      'completion_regression_gate': 'not_established'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
