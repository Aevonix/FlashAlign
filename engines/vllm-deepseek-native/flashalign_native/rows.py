# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Request metadata and a CPU reference for native adaptive/CED row mapping."""
import math


def output_view(output):
    if output.ndim == 2:
        return output
    if output.ndim == 3 and output.shape[2] == 1:
        return output[:, :, 0]
    raise ValueError('Unsupported native projection output layout')


def batch_requests(request_ids, selections, profile_name, profile_digest):
    owners, strengths = [], []
    for request_id in request_ids:
        selection = selections.get(request_id)
        if selection is None:
            owners.append(False)
            strengths.append(0.0)
            continue
        if selection.name != profile_name or selection.digest != profile_digest:
            raise ValueError("Worker profile identity mismatch")
        if not math.isfinite(selection.scale) or not 0 < selection.scale <= 1:
            raise ValueError("Invalid worker profile strength")
        owners.append(True)
        strengths.append(float(selection.scale))
    return owners, strengths


def reference_rows(boundaries, owners, strengths, capacity, row_indices=None):
    """Use final boundaries, not the CPU upper-bound allocation for drafts."""
    if (len(boundaries) != len(owners) + 1 or len(owners) != len(strengths)
            or boundaries[0] != 0 or any(a > b for a, b in zip(boundaries, boundaries[1:]))
            or boundaries[-1] > capacity):
        raise ValueError("Invalid row boundaries")
    result = [(False, 0.0)] * capacity
    for index, (owner, strength) in enumerate(zip(owners, strengths)):
        for row in range(boundaries[index], boundaries[index + 1]):
            result[row] = bool(owner), float(strength) if owner else 0.0
    if row_indices is None:
        return result
    return [result[index] if 0 <= index < capacity else (False, 0.0)
            for index in row_indices]


def reference_project(output, basis, site_scale, rows):
    """Independent scalar reference; stock rows never enter arithmetic."""
    result = [list(row) for row in output]
    for index, (owner, alpha) in enumerate(rows):
        if not owner:
            continue
        dot = sum(value * direction for value, direction in zip(output[index], basis))
        result[index] = [value - alpha * site_scale * dot * direction
                         for value, direction in zip(output[index], basis)]
    return result
