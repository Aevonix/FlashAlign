# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Host-only execution classification; never reads a device token count.

This is a planning component. It does not install hooks or grant authorization.
Selections must come from the existing authenticated request gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import re
from typing import Mapping


class ExecutionClass(IntEnum):
    OFF = 0
    OWNER = 1
    MIXED = 2


@dataclass(frozen=True)
class Selection:
    profile_name: str
    profile_digest: str
    strength: float


def _identity(name: str, digest: str) -> None:
    if not isinstance(name, str) or not name or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid native profile identity")


def classify_requests(scheduled: Mapping[str, int], selections: Mapping[str, Selection],
                      *, profile_name: str, profile_digest: str) -> ExecutionClass:
    """Classify scheduled request ownership, not adaptive device token counts.

    Scheduled budgets are only a host-side superset. Device row ownership must
    still be produced from final query starts. A mixed class may execute with no
    owner tokens after adaptation; it must never treat every row as authorized.
    """
    _identity(profile_name, profile_digest)
    owners = []
    for request, budget in scheduled.items():
        if not isinstance(request, str) or not request or type(budget) is not int or budget < 0:
            raise ValueError("Invalid scheduled request or token budget")
        if budget == 0:
            continue
        selected = selections.get(request)
        if selected is None:
            owners.append(False)
            continue
        if (not isinstance(selected, Selection) or selected.profile_name != profile_name
                or selected.profile_digest != profile_digest
                or type(selected.strength) not in (float, int) or selected.strength != 1):
            raise ValueError("Native selection identity or strength mismatch")
        owners.append(True)
    if not owners or not any(owners):
        return ExecutionClass.OFF
    return ExecutionClass.OWNER if all(owners) else ExecutionClass.MIXED


def request_owners(request_ids: list[str], selections: Mapping[str, Selection],
                   *, profile_name: str, profile_digest: str) -> tuple[bool, ...]:
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Duplicate live request ids")
    return tuple(classify_requests({r: 1}, selections, profile_name=profile_name,
                                  profile_digest=profile_digest) == ExecutionClass.OWNER
                 for r in request_ids)


@dataclass(frozen=True)
class GraphIdentity:
    profile_digest: str
    execution_class: ExecutionClass

    def __post_init__(self):
        _identity("native", self.profile_digest)
        if type(self.execution_class) is not ExecutionClass:
            raise ValueError("Unknown execution class")


def expand_capture_descriptors(descriptors, *, enabled: bool):
    """Duplicate dataclass graph descriptors on a dedicated identity field.

    The engine patch supplies `alignment_case`; LoRA fields remain untouched.
    OFF is retained exactly once. Without the native operator, no copies or
    field rewrites occur. Descriptor ordering within each class is preserved.
    """
    rows = list(descriptors)
    if not enabled:
        return rows
    if any(getattr(row, "alignment_case", None) != 0 for row in rows):
        raise ValueError("Capture descriptors must start in OFF class")
    return [replace(row, alignment_case=int(mode))
            for mode in ExecutionClass for row in rows]


def require_native_runtime_qualified(*, native_kernels: bool, graph_variants: bool,
                                     row_mapping: bool, tp_receipts: bool) -> None:
    """Prevent preparation components being mistaken for a usable adapter."""
    if not all(type(x) is bool and x for x in
               (native_kernels, graph_variants, row_mapping, tp_receipts)):
        raise RuntimeError("Native WO-B runtime is not qualified")
