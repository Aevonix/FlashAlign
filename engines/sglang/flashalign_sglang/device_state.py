# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fixed-address row publication for one CUDA submission stream."""
from contextlib import contextmanager
import threading

import torch


class DeviceRowState:
    def __init__(self, capacity, profile_count, *, device):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Positive row capacity required")
        self.capacity, self.profile_count = capacity, profile_count
        self.slots = torch.zeros(capacity, device=device, dtype=torch.int32)
        self.strengths = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.unused_counter = torch.zeros(1, device=device, dtype=torch.int32)
        self._lock = threading.Lock()
        self._stream = None
        self._live_rows = 0

    def clear_retired(self):
        """Call only after the worker's GPU fence and zero-reference check."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot retire during graph submission")
        try:
            self.slots.zero_()
            self.strengths.zero_()
            self._live_rows = 0
        finally:
            self._lock.release()

    @contextmanager
    def submission(self, plan):
        """Keep publish and forward submission inseparable, without GPU sync.

        CUDA FIFO ordering protects the previous graph while the next snapshot
        is copied. Multi-stream/overlapped target runners are rejected instead
        of racing a global table. Host staging uses synchronous pageable copies whose lifetime ends
        after their ordered submission.
        """
        if plan.capacity != self.capacity or len(plan.slots) != plan.live_rows or len(plan.strengths) != plan.live_rows:
            raise ValueError("Row-plan capacity differs from captured state")
        if any(slot < 0 or slot > self.profile_count for slot in plan.slots):
            raise ValueError("Unknown profile slot")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Overlapping target forward is unqualified")
        try:
            stream = torch.cuda.current_stream(self.slots.device).cuda_stream
            if self._stream is not None and stream != self._stream:
                raise RuntimeError("Cross-stream target forward is unqualified")
            self._stream = stream
            previous_rows = self._live_rows
            count = plan.live_rows
            # If either copy fails, a later attempt must still clear every
            # possibly written row before it may dispatch a forward.
            self._live_rows = max(previous_rows, count)
            # Clear only the previous live prefix. The rest of this fixed
            # allocation was initialized to zero and has never been written.
            self.slots[:previous_rows].zero_()
            self.strengths[:previous_rows].zero_()
            if count:
                self.slots[:count].copy_(torch.tensor(plan.slots, dtype=torch.int32), non_blocking=False)
                self.strengths[:count].copy_(torch.tensor(plan.strengths, dtype=torch.float32), non_blocking=False)
            self._live_rows = count
            yield
        finally:
            self._lock.release()


def build_site_bank(registry, profiles, site, hidden, *, device):
    """Install immutable rank-one banks before any graph capture occurs."""
    bank = torch.zeros((registry.capacity + 1, hidden), dtype=torch.float32, device=device)
    scales = torch.zeros(registry.capacity + 1, dtype=torch.float32, device=device)
    for slot, identity in registry.entries():
        profile = profiles[identity.name]
        if profile.digest != identity.digest:
            raise ValueError("Profile content changed after registry creation")
        operator = profile.sites.get(site)
        if operator is None:
            continue
        q = operator["q"]
        if tuple(q.shape) != (1, hidden):
            raise ValueError("Fused graph kernels accept rank-one sites; retain eager execution")
        bank[slot].copy_(q[0])
        scales[slot] = operator["scale"]
    return bank, scales
