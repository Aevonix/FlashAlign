# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CPU reference for device-side ownership and CED mapping contracts.

Never use this reference to read back GPU starts in a serving path. Runtime
must reuse the existing device prepare_rows kernel and a GPU compaction map.
"""
from __future__ import annotations

from dataclasses import dataclass


def expand_final_starts(starts: tuple[int, ...], owners: tuple[bool, ...], capacity: int):
    if (type(capacity) is not int or capacity < 0 or len(starts) != len(owners) + 1
            or not starts or starts[0] != 0
            or any(type(x) is not int for x in starts)
            or any(type(x) is not bool for x in owners)
            or any(a > b for a, b in zip(starts, starts[1:]))
            or starts[-1] > capacity):
        raise ValueError("Invalid final query boundary contract")
    out = [False] * capacity
    for owner, start, stop in zip(owners, starts, starts[1:]):
        out[start:stop] = [owner] * (stop - start)
    return tuple(out)


def map_layer_rows(owner_rows: tuple[bool, ...], row_count: int,
                   row_indices: tuple[int, ...] | None = None):
    if type(row_count) is not int or row_count < 0:
        raise ValueError("Invalid layer row count")
    if row_indices is None:
        if row_count > len(owner_rows):
            raise ValueError("Layer rows exceed ownership capacity")
        return owner_rows[:row_count]
    if len(row_indices) != row_count or any(type(i) is not int for i in row_indices):
        raise ValueError("CED map does not match layer rows")
    # Invalid/sentinel indices are unowned. This mirrors the fail-closed GPU
    # kernel behavior and never aliases Python's negative indexing.
    return tuple(owner_rows[i] if 0 <= i < len(owner_rows) else False for i in row_indices)


@dataclass(frozen=True)
class CompactionReference:
    source_rows: tuple[int, ...]
    active: int
    capacity: int


def compact_owner_reference(layer_owners: tuple[bool, ...], capacity: int):
    if (type(capacity) is not int or capacity < len(layer_owners)
            or any(type(x) is not bool for x in layer_owners)):
        raise ValueError("Invalid compaction capacity")
    selected = tuple(i for i, selected in enumerate(layer_owners) if selected)
    return CompactionReference(selected + (-1,) * (capacity - len(selected)),
                               len(selected), capacity)


def merge_selected_reference(stock: tuple[bytes, ...], edited: tuple[bytes, ...],
                             compacted: CompactionReference):
    """Byte-oriented proof of owner-only scatter; no numerical operation."""
    if (len(edited) != compacted.capacity or len(compacted.source_rows) != compacted.capacity
            or not 0 <= compacted.active <= compacted.capacity):
        raise ValueError("Invalid edited output capacity")
    active = compacted.source_rows[:compacted.active]
    if (len(active) != len(set(active)) or any(type(i) is not int or not 0 <= i < len(stock)
                                             for i in active)
            or any(i != -1 for i in compacted.source_rows[compacted.active:])):
        raise ValueError("Invalid reverse mapping")
    result = list(stock)
    for source, output in zip(active, edited):
        result[source] = output
    return tuple(result)
