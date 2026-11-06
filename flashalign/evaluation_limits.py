# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Explicit evaluation resource budgets, separate from generation options."""

LEGACY_PROTOCOL = "flashalign.strict-completion.v1"
LIMITED_PROTOCOL = "flashalign.strict-completion.v2"
MIB = 1024 * 1024
LEGACY_LIMITS = {
    "max_response_bytes": 4 * MIB,
    "max_events": 65536,
    "max_sse_event_bytes": 16 * MIB,
    "max_record_bytes": 16 * MIB,
    "max_judge_request_bytes": None,
}
UPPER_LIMITS = {
    "max_response_bytes": 16 * MIB,
    "max_events": 65536,
    "max_sse_event_bytes": 16 * MIB,
    "max_record_bytes": 64 * MIB,
    "max_judge_request_bytes": 64 * MIB,
}


def validate_limits(value):
    """Require a complete, bounded declaration; bools and floats are invalid."""
    if not isinstance(value, dict) or set(value) != set(UPPER_LIMITS):
        raise ValueError("Evaluation limits must declare every supported resource budget")
    if any(type(value[name]) is not int or not 0 < value[name] <= maximum
           for name, maximum in UPPER_LIMITS.items()):
        raise ValueError("Evaluation resource budgets must be positive bounded integers")
    return dict(value)


def limits_for_run(run):
    """Read old runs without changing their caps or silently upgrading them."""
    descriptor = run.get("descriptor")
    if not isinstance(descriptor, dict) or descriptor.get("protocol") != run.get("protocol"):
        raise ValueError("Evaluation protocol binding mismatch")
    if run.get("protocol") == LEGACY_PROTOCOL:
        if "evaluation_limits" in descriptor or "evaluation_limits" in run:
            raise ValueError("Legacy evaluation runs cannot declare new resource budgets")
        return dict(LEGACY_LIMITS)
    if run.get("protocol") != LIMITED_PROTOCOL:
        raise ValueError("Unsupported evaluation protocol")
    limits = validate_limits(descriptor.get("evaluation_limits"))
    if run.get("evaluation_limits") != limits:
        raise ValueError("Evaluation resource budget binding mismatch")
    return limits
