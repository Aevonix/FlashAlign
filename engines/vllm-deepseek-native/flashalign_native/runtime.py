# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Persistent request rows consumed by existing full and piecewise graphs."""
import numpy as np
import torch

from .rows import batch_requests, output_view


class Runtime:
    def __init__(self, runner, profile):
        from . import kernels
        self.kernels = kernels
        self.profile = profile
        self.device = runner.device
        self.capacity = runner.max_num_tokens
        self.owner_rows = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)
        self.strength_rows = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self.owner_requests = torch.zeros(runner.max_num_reqs, dtype=torch.int32, device=self.device)
        self.strength_requests = torch.zeros(runner.max_num_reqs, dtype=torch.float32, device=self.device)
        self.operators = {}
        for site, name in profile.sites.items():
            q, scale = profile.operators[name]
            if tuple(q.shape) == (1, 5120):
                # Keep the existing rank-one representation and kernel unchanged.
                basis = q[0].contiguous().to(self.device)
            elif tuple(q.shape) == (2, 5120):
                basis = q.contiguous().to(self.device)
            else:
                raise ValueError("Unsupported native profile rank or site")
            self.operators[site] = basis, scale

    def clear(self):
        self.owner_rows.zero_()
        self.strength_rows.zero_()

    def prepare(self, batch, selections):
        from vllm.utils.torch_utils import async_tensor_h2d
        if batch.num_reqs > self.owner_requests.numel() or batch.num_tokens_after_padding > self.capacity:
            raise ValueError("Batch exceeds owner row-state capacity")
        owners, strengths = batch_requests(batch.req_ids, selections,
                                           self.profile.name, self.profile.digest)
        async_tensor_h2d(np.asarray(owners, dtype=np.int32),
                         out=self.owner_requests[:batch.num_reqs])
        async_tensor_h2d(np.asarray(strengths, dtype=np.float32),
                         out=self.strength_requests[:batch.num_reqs])
        # Adaptive verification may rewrite query_start_loc on the GPU. The
        # CPU query_start_loc_np is only an upper bound and must not be used.
        self.kernels.prepare_rows(batch.query_start_loc, self.owner_requests,
            self.strength_requests, self.owner_rows, self.strength_rows, batch.num_reqs)

    def apply(self, output, layer, site, row_indices=None):
        operator = self.operators.get((layer, site))
        if operator is None:
            return output
        q, scale = operator
        apply = self.kernels.apply if q.ndim == 1 else self.kernels.apply_rank2
        apply(output_view(output), q, self.owner_rows, self.strength_rows,
              row_indices, scale, None)
        return output

    def apply_embedding(self, output, is_multimodal):
        operator = self.operators.get((0, 'embedding_out'))
        if operator is None:
            return output
        q, scale = operator
        excluded = None if is_multimodal is None else is_multimodal.to(self.device, non_blocking=True)
        apply = self.kernels.apply if q.ndim == 1 else self.kernels.apply_rank2
        return apply(output, q, self.owner_rows, self.strength_rows,
                     None, scale, excluded)
