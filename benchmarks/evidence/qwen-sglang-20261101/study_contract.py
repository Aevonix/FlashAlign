"""Explicit layout for original isolated and amended concurrent-host studies."""
import re
import math

ISOLATED = 'isolated'
FRANCE = 'france_declared_concurrent_host_v1'
FRANCE_QUIET = 'france_quiet_host_v1'
LEGACY_ORDER = {1: ['A', 'B'], 2: ['B', 'A'], 3: ['A', 'B']}


def execution_mode(manifest):
    mode = manifest.get('execution_mode', ISOLATED)
    if mode not in (ISOLATED, FRANCE, FRANCE_QUIET):
        raise ValueError('Unknown study execution mode')
    return mode


def orders(manifest):
    if execution_mode(manifest) == ISOLATED:
        return LEGACY_ORDER
    values = manifest.get('arm_order')
    if (not isinstance(values, list) or len(values) != 3
            or any(value not in (['A', 'B'], ['B', 'A']) for value in values)
            or sum(value == ['A', 'B'] for value in values) not in (1, 2)
            or not re.fullmatch('[0-9a-f]{64}', manifest.get('order_draw_sha256', ''))):
        raise ValueError('The amended near-balanced order must be sealed')
    return {index: value for index, value in enumerate(values, 1)}


def required_check(manifest):
    return 'no_overlap' if execution_mode(manifest) == ISOLATED else 'declared_host_envelope_verified'


def process_slots(manifest):
    slots = {(block, arm, None) for block in (1, 2, 3) for arm in ('A', 'B')}
    controls = (None,) if execution_mode(manifest) == ISOLATED else ('before', 'after')
    return slots | {(block, 'stock', control) for block in (1, 2, 3) for control in controls}


def stationarity(before, after):
    comparisons = {}
    for metric in ('throughput_tokens_per_second', 'ttft_p95_seconds', 'e2e_p95_seconds'):
        a, b = before.get(metric), after.get(metric)
        valid = (type(a) in (int, float) and type(b) in (int, float)
                 and math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0)
        ratio = b / a if valid else None
        within = ratio is not None and (abs(ratio - 1) <= .10 or math.isclose(abs(ratio - 1), .10, rel_tol=0, abs_tol=1e-12))
        comparisons[metric] = {'before': a, 'after': b, 'after_over_before': ratio, 'within_ten_percent': within}
    return {'passed': all(row['within_ten_percent'] for row in comparisons.values()), 'metrics': comparisons}
