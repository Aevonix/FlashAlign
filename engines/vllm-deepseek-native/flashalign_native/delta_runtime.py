# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Low-rank profiles beside the matrix profile: one row state per profile, two native sites.

Each profile keeps its own request rows, so any subset can be selected per request.
Both sites add to this rank's local writer output before the tensor-parallel
reduction, inside the engine's opaque attention and MoE operations. Profiles are
applied in the order they were loaded.

The MoE operation takes no row map, so the attention operation of the same layer
stages one in a persistent buffer. Nothing in traced model code writes that buffer.
"""
import json
import math

from .delta_profile import local_columns
from .rows import output_view


def parse_selection(raw, profiles):
    """Check the middleware's low-rank selections in the worker. Returns {name: strength}."""
    try:
        rows = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        rows = None
    if not isinstance(rows, list) or not rows or len(rows) > len(profiles):
        raise ValueError('Invalid worker low-rank selection')
    result = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError('Invalid worker low-rank selection')
        name, digest, scale = row
        profile = profiles.get(name) if isinstance(name, str) else None
        if profile is None or profile.digest != digest or name in result:
            raise ValueError('Worker profile identity mismatch')
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError('Invalid worker profile strength')
        result[name] = float(scale)
    return result


def request_rows(request_ids, selections, name):
    """One (selected, strength) pair per scheduled request for one profile."""
    owners, strengths = [], []
    for request_id in request_ids:
        scale = (selections.get(request_id) or {}).get(name)
        owners.append(scale is not None)
        strengths.append(float(scale) if scale is not None else 0.0)
    return owners, strengths


class DeltaRuntime:
    def __init__(self, runner, profiles, *, tp_rank, tp_size):
        import torch
        from . import delta_kernels
        if not profiles:
            raise ValueError('No low-rank profile to attach')
        self.kernels, self.profiles = delta_kernels, profiles
        self.names = tuple(profiles)
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.device, self.capacity = runner.device, runner.max_num_tokens
        count = len(self.names)
        self.owner_rows = torch.zeros((count, self.capacity), dtype=torch.int32, device=self.device)
        self.strength_rows = torch.zeros((count, self.capacity), dtype=torch.float32, device=self.device)
        self.owner_requests = torch.zeros((count, runner.max_num_reqs), dtype=torch.int32, device=self.device)
        self.strength_requests = torch.zeros((count, runner.max_num_reqs), dtype=torch.float32, device=self.device)
        # Which request row each model row came from. Staged per targeted layer.
        self.row_map = torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        self.staged_rows = 0
        self.sites = sorted({key for profile in profiles.values() for key in profile.sites})
        self.operators = {}

    def bind(self, layer, site, width):
        """Place every profile's operator for one site on the device, cut to this rank's columns."""
        operators = []
        for index, profile in enumerate(self.profiles.values()):
            operator = profile.sites.get((layer, site))
            if operator is None:
                continue
            q, b, scale = operator
            local = local_columns(b, self.tp_rank, self.tp_size, width)
            operators.append((index, q.to(self.device).contiguous(), local.to(self.device).contiguous(), scale))
        if not operators:
            raise ValueError('No low-rank operator for the bound site')
        self.operators[(layer, site)] = operators

    def prepare(self, batch, selections):
        import numpy as np
        from vllm.utils.torch_utils import async_tensor_h2d
        from .kernels import prepare_rows
        if batch.num_reqs > self.owner_requests.shape[1] or batch.num_tokens_after_padding > self.capacity:
            raise ValueError('Batch exceeds low-rank row-state capacity')
        for index, name in enumerate(self.names):
            owners, strengths = request_rows(batch.req_ids, selections, name)
            async_tensor_h2d(np.asarray(owners, dtype=np.int32),
                             out=self.owner_requests[index, :batch.num_reqs])
            async_tensor_h2d(np.asarray(strengths, dtype=np.float32),
                             out=self.strength_requests[index, :batch.num_reqs])
            # Adaptive verification may rewrite query_start_loc on the GPU; use the final boundaries.
            prepare_rows(batch.query_start_loc, self.owner_requests[index], self.strength_requests[index],
                         self.owner_rows[index], self.strength_rows[index], batch.num_reqs)

    def attention(self, layer, output, source, row_indices):
        """Inside the attention operation of a targeted layer, on its local result before the reduction."""
        x = output_view(output)
        # Record which request row each row of this layer came from. The MoE
        # writer of the same layer runs later in its own operation and reads it.
        self.kernels.stage_rows(self.row_map, x, row_indices)
        self.staged_rows = x.shape[0]
        self._add(layer, 'attn_out', x, source)

    def shared_expert(self, layer, output, source):
        """Inside the MoE operation, on the shared expert's local result before the reduction."""
        if (layer, 'shared_expert_out') not in self.operators:
            raise ValueError('Low-rank site was not bound')
        x = output_view(output)
        if x.shape[0] != self.staged_rows:
            raise ValueError('Shared expert rows differ from the attention rows of this layer')
        self._add(layer, 'shared_expert_out', x, source)

    def _add(self, layer, site, x, source):
        """Each loaded profile's delta for this site, in load order."""
        for index, q, b, scale in self.operators.get((layer, site), ()):
            self.kernels.delta_inplace(x, source, b, q, self.owner_rows[index], self.strength_rows[index],
                                       self.row_map, scale)


def _wrap_down_projection(down, runtime, layer):
    original = down.forward
    if getattr(original, '_flashalign_delta', False):
        raise ValueError('Shared expert projection is already bound')

    def forward(input_):
        result = original(input_)
        output = result[0] if isinstance(result, tuple) else result
        runtime.shared_expert(layer, output, input_)
        return result

    forward._flashalign_delta = True
    down.forward = forward


def attach_deltas(runner, profiles):
    """Bind low-rank profiles to the pinned model. No profile means no change at all."""
    if not profiles:
        return None
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    base = runner.model.language_model.model
    if type(base).__module__ != 'vllm.models.deepseek_v4_1.nvidia.model':
        raise ValueError('Expected pinned native B12X model')
    if set(profiles) & set(getattr(runner, '_flashalign_profiles', {})):
        raise ValueError('Duplicate profile name')
    runtime = DeltaRuntime(runner, profiles, tp_rank=get_tensor_model_parallel_rank(),
                           tp_size=get_tensor_model_parallel_world_size())
    for index, site in runtime.sites:
        layer = base.layers[index]
        attention = layer.attn
        if attention.layer_id != index:
            raise ValueError('Main-layer numbering differs from the profile')
        # The attention operation of every targeted layer stages the row map, with or without an operator of its own.
        attention._flashalign_delta, attention._flashalign_delta_layer = runtime, index
        if site == 'attn_out':
            runtime.bind(index, site, attention.n_local_groups * attention.o_lora_rank)
        else:
            down = getattr(getattr(layer.ffn, 'shared_experts', None), 'down_proj', None)
            if (down is None or not down.input_is_parallel
                    or (down.reduce_results and runtime.tp_size > 1)):
                raise ValueError('Shared expert output is not a local result before the reduction')
            runtime.bind(index, site, int(down.input_size_per_partition))
            _wrap_down_projection(down, runtime, index)
        layer._flashalign_delta = runtime
    runner._flashalign_delta_runtime = runtime
    runner._flashalign_delta_profiles = profiles
    runner._flashalign_delta_selections = {}
    return runtime
