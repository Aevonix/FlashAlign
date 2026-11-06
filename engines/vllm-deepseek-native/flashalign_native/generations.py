# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Generation-owned native matrices and graph captures for the pinned runner.

The same checkpoint and topology checks used at startup apply to every stage.
CPU tests cover ownership and dispatch; CUDA capture needs GPU qualification.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json

from owner_mode.profile_lifecycle import GenerationBusy, LifecycleError, ResourceLimit
from flashalign_native_wo.loader import NativePayload
from flashalign_native_wo.execution import install_native_weights
from flashalign_native_wo.runtime import NativeRuntime
from .lifecycle import WorkerRegistry
from .native_profile import NativeProfile, validate_native_selection


@dataclass(frozen=True)
class Snapshot:
    members: tuple

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.members, separators=(',', ':')).encode()).hexdigest()

    def slot(self, generation):
        for index, (gid, _) in enumerate(self.members, 1):
            if gid == generation:
                return index
        raise LifecycleError('Request generation is outside the captured binding set')


@dataclass
class NativeHandle:
    profile: NativeProfile
    receipts: tuple


class GenerationRuntime(NativeRuntime):
    """Reuse stable row-buffer storage already captured by the startup graph."""
    def __init__(self, original, registry):
        self.__dict__.update(original.__dict__)
        self.registry = registry
        self.active_snapshot = None

    @property
    def graph_digest(self):
        return self.active_snapshot.digest if self.active_snapshot is not None else self.profile.digest

    def _slot(self, selected):
        if selected is None:
            return 0
        if self.active_snapshot is None:
            validate_native_selection(self.profile, selected.scale)
            if selected.name != self.profile.name or selected.digest != self.profile.digest:
                raise LifecycleError('Native startup profile identity mismatch')
            return 1
        generation = getattr(selected, 'generation', None)
        handle = self.registry.resolve(generation, selected.digest)
        validate_native_selection(handle.profile, selected.scale)
        return self.active_snapshot.slot(generation)

    def classify_scheduled(self, scheduled):
        slots = [self._slot(self.selections.get(request)) for request, count in scheduled.items() if count]
        return 0 if not any(slots) else (1 if all(slots) else 2)

    def prepare(self, batch, selections=None):
        import numpy as np
        from .kernels import prepare_rows
        from vllm.utils.torch_utils import async_tensor_h2d
        if batch.num_reqs > self.owner_requests.numel() or batch.num_tokens_after_padding > self.capacity:
            raise LifecycleError('Native generation batch exceeds stable row capacity')
        selections = self.selections if selections is None else selections
        owners = [self._slot(selections.get(request)) for request in batch.req_ids]
        async_tensor_h2d(np.asarray(owners, dtype=np.int32), out=self.owner_requests[:batch.num_reqs])
        async_tensor_h2d(np.asarray([float(x != 0) for x in owners], dtype=np.float32),
                         out=self.strength_requests[:batch.num_reqs])
        prepare_rows(batch.query_start_loc, self.owner_requests, self.strength_requests,
                     self.owner_rows, self.strength_rows, batch.num_reqs)

    def bindings(self, attention):
        if self.active_snapshot is None:
            return None
        stored = getattr(attention, '_flashalign_native_generations', {})
        result = []
        for slot, (gid, _) in enumerate(self.active_snapshot.members, 1):
            handle = self.handles[gid]
            if attention.layer_id in handle.profile.spec.layers:
                binding = stored.get(gid)
                if binding is None:
                    raise LifecycleError('Native generation has no layer binding')
                result.append((slot, binding[2]))
        return tuple(result)

    def prepare_capture(self, alignment_case, num_reqs, num_tokens):
        super().prepare_capture(alignment_case, num_reqs, num_tokens)
        # Exercise all loaded matrix bindings during graph capture. Real replay
        # rewrites these same row buffers from final query boundaries.
        if self.active_snapshot is not None and alignment_case:
            count = len(self.active_snapshot.members)
            for offset in range(count):
                self.owner_rows[offset:num_tokens:count].fill_(offset + 1)
            if alignment_case == 2:
                self.owner_rows[:num_tokens:2].zero_()

    def record_execution(self, batch, case, digest, graph_mode):
        if self.active_snapshot is None:
            return super().record_execution(batch, case, digest, graph_mode)
        if self.preparing or self._execution_diagnostics.directory is None:
            return None
        from flashalign_native_wo.diagnostics import ExecutionDiagnostics
        diagnostics = getattr(self, '_generation_diagnostics', {})
        if digest not in diagnostics:
            diagnostics[digest] = ExecutionDiagnostics(self._execution_diagnostics.directory, digest)
            self._generation_diagnostics = diagnostics
        diagnostic = diagnostics[digest]
        if not diagnostic.pending(case, graph_mode):
            return None
        def counts():
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Execution diagnostics cannot run during capture')
            live = batch.query_start_loc[batch.num_reqs].to(torch.int64)
            active = torch.arange(self.capacity, device=self.device) < live
            owner = torch.count_nonzero(active & (self.owner_rows > 0))
            invalid = torch.count_nonzero(active & ((self.owner_rows < 0) |
                (self.owner_rows > len(self.active_snapshot.members))))
            values = torch.stack((live, owner, live-owner, invalid)).cpu().tolist()
            return dict(zip(('live_tokens','owner_tokens','off_tokens','invalid_flags'), values))
        return diagnostic.record(case,digest,counts,graph_mode=graph_mode,requests=batch.num_reqs,
                                 max_rows=min(self.capacity,batch.num_tokens_after_padding))


class NativeAdapter:
    def __init__(self, runner, profiles):
        self.runner, self.profiles = runner, profiles
        self.identity = next(iter(profiles.values())).spec.model
        self.base = runner.model.language_model.model
        self.handles = {}
        self.pending = None
        self.registry = None
        self.runtime = None

    def inventory(self):
        import torch
        return {'model_id': self.identity[0], 'model_revision': self.identity[1],
                'operators': ['native_matrix_override_v1'],
                'available_device_bytes': torch.cuda.mem_get_info(self.runner.device)[0]}

    def validate(self, artifact):
        if (artifact.model_id, artifact.model_revision) != self.identity or artifact.operator != 'native_matrix_override_v1':
            raise LifecycleError('Native generation targets a different checkpoint or operator')
        profile = NativeProfile(artifact.directory)
        validate_native_selection(profile, 1.0)

    def statistics(self):
        import torch
        sites = set(layer for handle in self.handles.values() for layer in handle.profile.spec.layers)
        config = self.runner.vllm_config
        spec = config.speculative_config
        return {'graph_mode': str(config.compilation_config.cudagraph_mode).split('.')[-1],
                'speculative_method': spec.method, 'expected_sites': len(sites),
                'installed_sites': len(sites), 'active_requests': len(self.runner.req_states.req_id_to_index),
                'gpu_memory_bytes': torch.cuda.memory_allocated(self.runner.device)}

    def prepare(self, gid, artifact):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        self.validate(artifact)
        profile = NativeProfile(artifact.directory)
        size = (profile.directory / profile.spec.payload_file).stat().st_size
        if size // profile.spec.tp_world_size > torch.cuda.mem_get_info(self.runner.device)[0]:
            raise ResourceLimit('Insufficient native generation overlap memory')
        payload = NativePayload(profile.directory, profile.spec)
        receipts = []
        try:
            for layer in self.base.layers:
                receipt = install_native_weights(layer.attn, profile.spec, payload,
                    get_tensor_model_parallel_rank(), generation_id=gid)
                if receipt is not None:
                    receipts.append(receipt)
        except BaseException:
            for layer in self.base.layers:
                getattr(layer.attn, '_flashalign_native_generations', {}).pop(gid, None)
            raise
        finally:
            payload.close()
        profile.runtime_digest, profile.digest = profile.digest, artifact.digest
        handle = NativeHandle(profile, tuple(receipts))
        self.handles[gid] = handle
        try:
            self.prepare_existing(gid, artifact, handle)
        except BaseException:
            self.retire(gid, handle)
            raise
        return handle

    def prepare_existing(self, gid, artifact, handle):
        snapshot = Snapshot(tuple(sorted((key, value.profile.digest) for key, value in self.handles.items())))
        self.pending = self._capture(snapshot)

    @contextmanager
    def selected_snapshot(self, snapshot):
        old = self.runtime.active_snapshot
        self.runtime.active_snapshot = snapshot
        try:
            yield
        finally:
            self.runtime.active_snapshot = old

    def _capture(self, snapshot):
        import torch
        old = self.runner.cudagraph_manager
        if old is None:
            raise LifecycleError('Native graph manager is not ready')
        if self.runtime.active_snapshot == snapshot:
            return snapshot, old
        torch.cuda.synchronize(self.runner.device)
        manager = type(old)(old.vllm_config, old.device, old.cudagraph_mode,
            decode_query_len=old.decode_query_len, lora_capture_cases=old.lora_capture_cases,
            varlen_decode=old.varlen_decode, ubatch_runner=old.ubatch_runner,
            specialize_full_decode=old.specialize_full_decode,
            single_request_prefill_tokens=old.single_request_prefill_tokens,
            alignment_profile_digest=snapshot.digest)
        with self.selected_snapshot(snapshot), self.runtime.preparation():
            # Only decoder graphs change. DSpark's existing draft graphs and
            # adaptive verification remain enabled and keep their own storage.
            manager.capture(self.runner.model, self.runner.model_state,
                self.runner.input_buffers, self.runner.intermediate_tensors,
                self.runner.block_tables, self.runner.attn_groups, self.runner.kv_cache_config,
                pcp_manager=self.runner.pcp_manager, has_lora=False,
                use_aux_hidden_state_outputs=self.runner.use_aux_hidden_state_outputs,
                lora_capture_hook=None, alignment_capture_hook=self.runtime.prepare_capture)
        torch.cuda.synchronize(self.runner.device)
        return snapshot, manager

    def commit(self, gid, handle, transaction):
        import torch
        if transaction.get('operation') == 'deactivate':
            self.pending = None
            return
        if self.pending is None:
            current = self.runtime.active_snapshot
            if current is not None and gid in {key for key, _ in current.members}:
                return
            raise LifecycleError('Native commit has no prepared graph generation')
        snapshot, manager = self.pending
        if gid not in {row[0] for row in snapshot.members}:
            raise LifecycleError('Prepared graphs do not contain the committed generation')
        torch.cuda.synchronize(self.runner.device)
        self.runtime.active_snapshot = snapshot
        self.runner.cudagraph_manager = manager
        self.pending = None
        # Startup's compatibility bindings are no longer a separate owner of
        # storage once generation-pinned graphs take over.
        for layer in self.base.layers:
            for attribute in ('_flashalign_native_weights', '_flashalign_native_install'):
                if hasattr(layer.attn, attribute):
                    delattr(layer.attn, attribute)
        payload = getattr(self.runtime, '_native_payload', None)
        if payload is not None:
            payload.close()
            self.runtime._native_payload = None

    def references(self, gid):
        live = sum(getattr(value, 'generation', None) == gid for value in self.runner._flashalign_selections.values())
        graphs = int(self.runtime.active_snapshot is not None and
                     gid in {key for key, _ in self.runtime.active_snapshot.members})
        return live + graphs

    def all_references(self):
        return len(self.runner.req_states.req_id_to_index)

    def prune_graphs(self, keep):
        if self.all_references():
            raise GenerationBusy('Native requests must drain before graph pruning')
        snapshot = Snapshot(tuple(sorted((key, self.handles[key].profile.digest) for key in keep)))
        self.pending = self._capture(snapshot)
        if keep:
            self.commit(next(iter(keep)), self.handles[next(iter(keep))], {})
        else:
            self.runtime.active_snapshot, self.runner.cudagraph_manager = self.pending
            self.pending = None

    def recover(self, wanted):
        if wanted:
            self.pending = self._capture(Snapshot(tuple(sorted((gid, self.handles[gid].profile.digest) for gid in wanted))))
            self.commit(next(iter(wanted)), self.handles[next(iter(wanted))], {})

    def retire(self, gid, handle):
        import torch
        if self.runtime.active_snapshot is not None and gid in {key for key, _ in self.runtime.active_snapshot.members}:
            raise GenerationBusy('Captured native graphs still reference the generation')
        torch.cuda.synchronize(self.runner.device)
        self.pending = None
        self.handles.pop(gid, None)
        for layer in self.base.layers:
            getattr(layer.attn, '_flashalign_native_generations', {}).pop(gid, None)


def install_generations(runner, profiles):
    from vllm.distributed import get_tensor_model_parallel_rank
    adapter = NativeAdapter(runner, profiles)
    registry = WorkerRegistry(adapter, str(get_tensor_model_parallel_rank()))
    adapter.registry = registry
    runtime = GenerationRuntime(runner._flashalign_native_runtime, registry)
    runtime.handles = adapter.handles
    adapter.runtime = runtime
    runner._flashalign_lifecycle = registry
    runner._flashalign_native_runtime = runner.model._flashalign_native_runtime = runtime
    for layer in adapter.base.layers:
        layer._flashalign_native_runtime = layer.attn._flashalign_native_runtime = runtime
    return registry
