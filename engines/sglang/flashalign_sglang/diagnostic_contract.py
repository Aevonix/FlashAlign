# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Numerical conformance checks without request-content logging."""


def sample_rows(plan, maximum=4):
    on = [i for i, slot in enumerate(plan.slots) if slot]
    off = [i for i, slot in enumerate(plan.slots) if not slot]
    selected = (on[:2] + off[:2]) if on and off else (on or off)[:maximum]
    return selected + [None] * (maximum - len(selected))


def expected_site_rows(plan, active_slots):
    return sum(slot in active_slots for slot in plan.slots)


def check_counter_delta(before, after, expected):
    if len(before) != len(after) or len(after) != len(expected):
        raise ValueError("Diagnostic site count mismatch")
    deltas = [end - start for start, end in zip(before, after)]
    if deltas != list(expected):
        raise ValueError("Missing or duplicate per-site projection in target forward")
    return deltas


def check_text_input_boundary(batch):
    for name in ("input_embeds", "replace_embeds", "mm_input_embeds"):
        if getattr(batch, name, None) is not None:
            raise ValueError("Graph path only qualifies ordinary text token inputs")
    inputs = getattr(batch, "mm_inputs", None)
    if inputs is not None and (not isinstance(inputs, (tuple, list)) or any(item is not None for item in inputs)):
        raise ValueError("Graph path does not qualify image or multimodal input preparation")
