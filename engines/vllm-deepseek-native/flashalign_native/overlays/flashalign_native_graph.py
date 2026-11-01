# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Native-profile graph identity; no model or tensor operations.

Graph identity alone does not establish serving qualification. Authentication
and the final GPU row map are validated by the native runtime.
"""
from __future__ import annotations

from dataclasses import replace
import re

OFF, OWNER, MIXED = 0, 1, 2
ALIGNMENT_CASES = (OFF, OWNER, MIXED)


def validate_alignment_identity(alignment_case, profile_digest, expected_digest=None):
    if (isinstance(alignment_case, bool) or not isinstance(alignment_case, int)
            or alignment_case not in ALIGNMENT_CASES):
        raise ValueError("Invalid native alignment execution case")
    if not isinstance(profile_digest, str) or (
            profile_digest and re.fullmatch(r"[0-9a-f]{64}", profile_digest) is None):
        raise ValueError("Invalid native alignment profile digest")
    if alignment_case != OFF and not profile_digest:
        raise ValueError("Native alignment execution requires a profile digest")
    if expected_digest is not None and profile_digest != expected_digest:
        raise ValueError("Native alignment graph/runtime profile mismatch")


def native_profile_digest(runtime, parallel_config):
    if runtime is None:
        return ""
    expected = {"data_parallel_size": 1, "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1, "decode_context_parallel_size": 1}
    for name, value in expected.items():
        if getattr(parallel_config, name, None) != value:
            raise ValueError("Native alignment graph path requires DP1/TP4/PP1/DCP1")
    if getattr(parallel_config, "prefill_context_parallel_size", 1) != 1:
        raise ValueError("Native alignment graph path does not support prefill CP")
    digest = runtime.profile.digest
    validate_alignment_identity(OWNER, digest)
    return digest


def classify_scheduled_alignment(runtime, scheduled_tokens, parallel_config, *, dummy=False):
    """Use authenticated CPU selections before graph dispatch; never read GPU data."""
    digest = native_profile_digest(runtime, parallel_config)
    if runtime is None:
        return OFF, digest
    if not isinstance(scheduled_tokens, dict) or any(
            not isinstance(key, str) or type(count) is not int or count < 0
            for key, count in scheduled_tokens.items()):
        raise ValueError("Native alignment classification requires CPU scheduled counts")
    if dummy or not any(scheduled_tokens.values()):
        return OFF, digest
    case = runtime.classify_scheduled(scheduled_tokens)
    validate_alignment_identity(case, digest, digest)
    return int(case), digest


def expand_alignment_descriptors(descriptors, profile_digest):
    """Capture every native execution class independently of existing LoRA keys."""
    validate_alignment_identity(OFF, profile_digest)
    if not profile_digest:
        return list(descriptors)
    return [replace(desc, alignment_case=case, alignment_profile_digest=profile_digest)
            for desc in descriptors for case in ALIGNMENT_CASES]


def validate_runtime_descriptor(runtime, descriptor, parallel_config):
    digest = native_profile_digest(runtime, parallel_config)
    validate_alignment_identity(descriptor.alignment_case,
                                descriptor.alignment_profile_digest, digest)


def record_native_execution(runtime, input_batch, descriptor, *, dummy=False, is_profile=False):
    """Optional post-forward qualification callback, never part of capture.

    The runtime callback is disabled by default and must return before device
    reads unless its explicit diagnostics option is enabled.
    """
    if runtime is None or dummy or is_profile:
        return
    validate_alignment_identity(descriptor.alignment_case,
                                descriptor.alignment_profile_digest, runtime.profile.digest)
    graph_mode = descriptor.cg_mode.name
    if graph_mode not in ("NONE", "FULL", "PIECEWISE"):
        raise ValueError("Unknown native diagnostic graph execution mode")
    runtime.record_execution(input_batch, descriptor.alignment_case,
                             descriptor.alignment_profile_digest, graph_mode=graph_mode)
