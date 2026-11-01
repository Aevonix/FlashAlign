# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Stable row buffers and host classification for explicit native profiles."""
from __future__ import annotations

from contextlib import contextmanager
import os

from .planning import Selection, classify_requests, request_owners
from .diagnostics import ExecutionDiagnostics


class NativeRuntime:
    def __init__(self, runner, profile, selections):
        import torch
        self.profile = profile
        self.selections = selections
        self.preparing = False
        self._execution_diagnostics = ExecutionDiagnostics(
            os.environ.get("FLASHALIGN_NATIVE_DIAGNOSTICS"), profile.digest)
        self.device = runner.device
        self.capacity = runner.max_num_tokens
        self.owner_rows = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)
        self.strength_rows = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.owner_requests = torch.zeros(runner.max_num_reqs, dtype=torch.int32, device=self.device)
        self.strength_requests = torch.zeros(runner.max_num_reqs, dtype=torch.float32, device=self.device)

    def _selections(self, source=None):
        source = self.selections if source is None else source
        return {key: Selection(item.name, item.digest, item.scale) for key, item in source.items()}

    def classify_scheduled(self, scheduled):
        return int(classify_requests(scheduled, self._selections(),
                                    profile_name=self.profile.name, profile_digest=self.profile.digest))

    @contextmanager
    def preparation(self):
        """Explicit startup-only scope; never entered by request handling."""
        previous = self.preparing
        self.preparing = True
        try:
            yield
        finally:
            self.preparing = previous

    def prepare(self, batch, selections=None):
        import numpy as np
        from flashalign_native.kernels import prepare_rows
        from vllm.utils.torch_utils import async_tensor_h2d
        if (batch.num_reqs > self.owner_requests.numel()
                or batch.num_tokens_after_padding > self.capacity):
            raise ValueError("Native ownership batch exceeds stable buffer capacity")
        owners = request_owners(batch.req_ids, self._selections(selections),
                                profile_name=self.profile.name, profile_digest=self.profile.digest)
        async_tensor_h2d(np.asarray(owners, dtype=np.int32),
                         out=self.owner_requests[:batch.num_reqs])
        async_tensor_h2d(np.asarray(owners, dtype=np.float32),
                         out=self.strength_requests[:batch.num_reqs])
        # The actual final GPU starts include adaptive verification changes.
        prepare_rows(batch.query_start_loc, self.owner_requests, self.strength_requests,
                     self.owner_rows, self.strength_rows, batch.num_reqs)

    def prepare_capture(self, alignment_case, num_reqs, num_tokens):
        from .planning import ExecutionClass
        if (type(alignment_case) is not int or alignment_case not in (0, 1, 2)
                or not 0 <= num_tokens <= self.capacity
                or not 0 <= num_reqs <= self.owner_requests.numel()):
            raise ValueError("Invalid native capture class or capacity")
        self.owner_rows.zero_(); self.strength_rows.zero_()
        self.owner_requests.zero_(); self.strength_requests.zero_()
        if alignment_case == ExecutionClass.OWNER:
            self.owner_rows[:num_tokens].fill_(1)
            self.strength_rows[:num_tokens].fill_(1)
        elif alignment_case == ExecutionClass.MIXED:
            # Dummy capture values exercise both ownership branches. Graph
            # replay reads the same buffers rewritten from real query starts.
            self.owner_rows[:num_tokens:2].fill_(1)
            self.strength_rows[:num_tokens:2].fill_(1)

    def record_execution(self, batch, case, digest, graph_mode):
        """Qualification-only sync after real forward; never call in a graph."""
        diagnostic = self._execution_diagnostics
        if diagnostic.directory is None or not diagnostic.pending(case, graph_mode):
            return None
        if self.preparing:
            return None

        def read_counts():
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Execution diagnostics must run outside CUDA graph capture")
            starts = batch.query_start_loc
            if (not starts.is_cuda or starts.device != self.owner_rows.device
                    or starts.ndim != 1 or starts.dtype not in (torch.int32, torch.int64)
                    or starts.numel() <= batch.num_reqs):
                raise ValueError("Execution diagnostics require final device query boundaries")
            live = starts[batch.num_reqs].to(torch.int64)
            indices = torch.arange(self.capacity, device=self.device, dtype=torch.int64)
            active = indices < live
            owner = torch.count_nonzero(active & (self.owner_rows == 1)).to(torch.int64)
            invalid = torch.count_nonzero(active & ((self.owner_rows != 0) &
                                                    (self.owner_rows != 1))).to(torch.int64)
            # A single aggregate transfer; no request IDs or per-row arrays.
            values = torch.stack((live, owner, live-owner, invalid)).cpu().tolist()
            return dict(zip(("live_tokens", "owner_tokens", "off_tokens", "invalid_flags"), values))

        return diagnostic.record(case, digest, read_counts, graph_mode=graph_mode,
                                 requests=batch.num_reqs,
                                 max_rows=min(self.capacity, batch.num_tokens_after_padding))
