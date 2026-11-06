# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Batch row ownership and immutable generation slot allocation."""
from dataclasses import dataclass
import math
from typing import Callable, Optional


class UnsupportedLayout(ValueError):
    pass


@dataclass(frozen=True)
class ProfileIdentity:
    name: str
    digest: str


@dataclass(frozen=True)
class Selection:
    name: str
    digest: str
    scale: float


@dataclass(frozen=True)
class RowPlan:
    slots: tuple[int, ...]
    strengths: tuple[float, ...]
    live_rows: int
    capacity: int

    @property
    def selected_rows(self):
        return sum(slot != 0 for slot in self.slots[:self.live_rows])

    @property
    def padded_slots(self):
        return self.slots + (0,) * (self.capacity - self.live_rows)

    @property
    def padded_strengths(self):
        return self.strengths + (0.,) * (self.capacity - self.live_rows)


class SlotRegistry:
    """Bounded slots for immutable generations; zero means stock mode.

    Slots can be reclaimed only after the lifecycle adapter proves no request,
    cache, graph submission or GPU completion still references the generation.
    """

    def __init__(self, profiles=(), *, capacity=None):
        ordered = tuple(sorted(profiles, key=lambda p: (p.name, p.digest)))
        if len({p.name for p in ordered}) != len(ordered):
            raise ValueError("Expected distinct preloaded profile names")
        if any(not p.name or not p.digest for p in ordered):
            raise ValueError("Missing profile identity")
        self.capacity = len(ordered) if capacity is None else capacity
        if type(self.capacity) is not int or not 1 <= self.capacity <= 256 or len(ordered) > self.capacity:
            raise ValueError("Invalid graph slot capacity")
        self._slots = {}
        for identity in ordered:
            self.reserve(identity)

    @property
    def identities(self):
        return tuple(identity for _, identity in self.entries())

    def entries(self):
        return tuple(sorted((slot, identity) for identity, slot in self._slots.items()))

    def reserve(self, identity):
        if not identity.name or not identity.digest:
            raise ValueError("Missing generation identity")
        if identity in self._slots:
            return self._slots[identity]
        if any(value.name == identity.name for value in self._slots):
            raise ValueError("A generation identifier cannot change content")
        occupied = set(self._slots.values())
        slot = next((slot for slot in range(1, self.capacity + 1) if slot not in occupied), None)
        if slot is not None:
            self._slots[identity] = slot
        return slot

    def release(self, identity, *, references, completed):
        if type(references) is not int or references != 0 or completed is not True:
            raise ValueError("Generation still has live graph or GPU references")
        return self._slots.pop(identity, None)

    def slot_for(self, selected):
        identity = ProfileIdentity(selected.name, selected.digest)
        try:
            slot = self._slots[identity]
        except KeyError:
            raise ValueError("Selection does not identify a loaded profile") from None
        alpha = selected.scale
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 < alpha <= 2:
            raise ValueError("Invalid authenticated request strength")
        return slot, float(alpha)


def graph_profile_reason(profile, captured_sites):
    if not set(profile.sites) <= set(captured_sites):
        return "uncaptured_sites"
    if any(tuple(op["q"].shape) != (1, profile.hidden) for op in profile.sites.values()):
        return "rank_not_supported_by_fused_kernel"
    return None


def graph_batch_reason(selections, available, *, rows, capacity):
    if rows > capacity:
        return "row_capacity"
    for selected in selections:
        if selected is not None and ProfileIdentity(selected.name, selected.digest) not in available:
            return "generation_requires_eager"
    return None


def graph_memory_bytes(widths, *, slots, rows, budget, diagnostic=False):
    if type(slots) is not int or not 1 <= slots <= 256:
        raise ValueError("Invalid graph slot capacity")
    if type(rows) is not int or not 1 <= rows <= 4 * 1024**2:
        raise ValueError("Invalid graph row capacity")
    if type(budget) is not int or budget < 1 or any(type(width) is not int or width < 1 for width in widths):
        raise ValueError("Invalid graph memory budget or width")
    total = (slots + 1) * sum((width + 1) * 4 for width in widths) + rows * 8
    if diagnostic:
        total += sum(widths) * 4 * 8 + len(widths) * 4 + 32
    if total > budget:
        raise ValueError("Graph row and profile banks exceed their configured memory budget")
    return total


def request_widths(*, mode, request_count, rows, extend_lengths=None, verify_width=None):
    """Explicit static row layouts from the pinned Qwen/SGLang adapter."""
    if any(type(n) is not int or n < 0 for n in (request_count, rows)):
        raise UnsupportedLayout("Invalid batch dimensions")
    if mode == "idle":
        if request_count or rows:
            raise UnsupportedLayout("Idle batches must contain no live requests or rows")
        return ()
    if mode == "decode":
        lengths = (1,) * request_count
    elif mode == "target_verify":
        if type(verify_width) is not int or verify_width <= 0:
            raise UnsupportedLayout("Static verification width required")
        lengths = (verify_width,) * request_count
    elif mode == "extend":
        if extend_lengths is None:
            raise UnsupportedLayout("Extension lengths required")
        lengths = tuple(extend_lengths)
        if any(type(n) is not int or n < 0 for n in lengths):
            raise UnsupportedLayout("Extension lengths must be nonnegative integers")
    else:
        raise UnsupportedLayout("Forward mode has not been qualified")
    if len(lengths) != request_count or sum(lengths) != rows:
        raise UnsupportedLayout("Rows do not match explicit request ownership")
    return lengths


def compile_plan(registry, *, selections, lengths, rows, capacity):
    """Build a complete snapshot. No previous batch state can survive padding."""
    if type(rows) is not int or rows < 0:
        raise UnsupportedLayout("Invalid live row count")
    if type(capacity) is not int or capacity < 1 or rows > capacity:
        raise UnsupportedLayout("Row-table capacity exceeded")
    selections, lengths = tuple(selections), tuple(lengths)
    if len(selections) != len(lengths) or any(type(n) is not int or n < 0 for n in lengths):
        raise UnsupportedLayout("Inconsistent request ownership")
    if sum(lengths) != rows:
        raise UnsupportedLayout("Inconsistent live row count")
    slots, strengths = [], []
    for selected, count in zip(selections, lengths):
        slot, alpha = (0, 0.) if selected is None else registry.slot_for(selected)
        slots.extend([slot] * count)
        strengths.extend([alpha] * count)
    # The device publisher clears the previous live prefix before copying this
    # compact snapshot. Untouched capacity stays initialized to zero. Avoid
    # allocating/copying 65K CPU slots for a one-row decode step.
    return RowPlan(tuple(slots), tuple(strengths), rows, capacity)


def plan_from_batch(batch, *, registry, decode_salt: Callable, capacity):
    """Resolve authenticated salts once before capture-independent dispatch."""
    salts = getattr(batch, "flashalign_cache_salts", None)
    count = batch.batch_size
    rows = batch.input_ids.numel()
    mode = batch.forward_mode
    if mode.is_idle():
        kind = "idle"
    elif mode.is_target_verify():
        kind = "target_verify"
    elif mode.is_decode():
        kind = "decode"
    elif mode.is_extend():
        kind = "extend"
    else:
        raise UnsupportedLayout("Forward mode has not been qualified")
    if salts is None:
        # A missing mapping on a live target forward is an integration failure,
        # not permission to silently execute the wrong profile.
        if count:
            raise UnsupportedLayout("Missing authenticated request row metadata")
        salts = []
    if len(salts) != count:
        raise UnsupportedLayout("Request count does not match row metadata")
    info = getattr(batch, "spec_info", None)
    width = getattr(info, "num_tokens_per_req", None)
    if width is None:
        width = getattr(info, "draft_token_num", None)
    lengths = request_widths(
        mode=kind, request_count=count, rows=rows,
        extend_lengths=getattr(batch, "extend_seq_lens_cpu", None), verify_width=width,
    )
    return compile_plan(
        registry, selections=[decode_salt(salt) for salt in salts],
        lengths=lengths, rows=rows, capacity=capacity,
    )
