# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Pinned B12X hooks; native graph dispatch and DSpark remain unchanged."""
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path

from .core import Selection, image_project, load_profiles

REGISTERED = False


def validate_config(config):
    parallel = config.parallel_config
    if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
            parallel.data_parallel_size) != (4, 1, 1):
        raise ValueError('FlashAlign native DeepSeek requires TP4, PP1, DP1')
    if parallel.enable_expert_parallel or parallel.use_ubatching:
        raise ValueError('EP and microbatching are not qualified')
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError('Context parallelism is not qualified')
    spec = config.speculative_config
    if (spec is None or spec.method != 'dspark' or spec.num_speculative_tokens != 7
            or not spec.enable_adaptive_verification):
        raise ValueError('The native adaptive DSpark7 baseline must remain enabled')
    if str(config.compilation_config.cudagraph_mode).split('.')[-1] != 'FULL_AND_PIECEWISE':
        raise ValueError('The native full/piecewise graph mode must remain enabled')
    if config.model_config.enforce_eager or config.lora_config is not None:
        raise ValueError('Eager override and LoRA are not qualified')
    if config.kv_transfer_config is not None:
        raise ValueError('External KV transfer is not qualified for this integration')
    if str(config.attention_config.backend).split('.')[-1] != 'B12X':
        raise ValueError('The native B12X attention backend must remain enabled')
    if config.kernel_config.linear_backend != 'b12x' or config.kernel_config.moe_backend != 'b12x':
        raise ValueError('The native B12X linear and MoE backends must remain enabled')
    if config.cache_config.cache_dtype != 'fp8':
        raise ValueError('The matched baseline requires FP8 KV cache')
    if config.engram_config.projection_tp:
        raise ValueError('Sharded Engram projections need separate profile qualification')


def validate_profile_contract(profile):
    expected = {(i, site) for i in range(40) for site in ('attn_out', 'moe_out')}
    expected |= {(0, 'embedding_out'), (0, 'vision_aligner_out'),
                 (0, 'image_marker_out'), (1, 'engram_value_out'), (14, 'engram_value_out')}
    if not profile.sites or not set(profile.sites) <= expected:
        raise ValueError('Profile declares a site without a qualified native hook')
    if any(tuple(basis.shape) not in ((1, 5120), (2, 5120))
           for basis, _ in profile.operators.values()):
        raise ValueError('Native grouped operators require rank one or two')
    for row in profile.meta['sites']:
        layer, site = row['layer'], row['site']
        valid = False
        if site == 'attn_out':
            valid = row.get('tensor') == f'layers.{layer}.attn.wo_b.weight'
        elif site == 'moe_out':
            valid = (row.get('tensor_pattern') == f'layers.{layer}.ffn.*.w2.weight'
                     and row.get('components') == ['routed_experts', 'shared_expert']
                     and row.get('application') == 'combined_moe_output_before_mhc_residual')
        elif site == 'embedding_out':
            valid = row.get('tensor') == 'embed.weight'
        elif site == 'engram_value_out':
            valid = (row.get('tensor') == f'layers.{layer}.engram.wkv.weight'
                     and row.get('output_slice') == [20480, 25600])
        elif site == 'vision_aligner_out':
            valid = (row.get('tensor') == 'aligner.w2.weight'
                     and row.get('bias_tensor') == 'aligner.w2.bias'
                     and row.get('application') == 'before_unchanged_bias')
        elif site == 'image_marker_out':
            valid = (row.get('tensors') == ['image_start', 'image_end', 'image_newline']
                     and row.get('application') == 'typed_image_span_delimiters_without_aligner_bias')
        if not valid:
            raise ValueError('Profile tensor does not match its native site')


def attach(runner, profiles):
    from .runtime import Runtime
    validate_config(runner.vllm_config)
    if len(profiles) != 1:
        raise ValueError('Native integration requires one explicit profile')
    profile = next(iter(profiles.values()))
    if getattr(profile, 'is_native_wo', False):
        from .native_attach import attach_native
        return attach_native(runner, profile)
    validate_profile_contract(profile)
    model = runner.model
    base = model.language_model.model
    if type(base).__module__ != 'vllm.models.deepseek_v4_1.nvidia.model':
        raise ValueError('Expected pinned native B12X model')
    runtime = Runtime(runner, profile)
    runner._flashalign_runtime = runtime
    runner._flashalign_profiles = profiles
    runner._flashalign_selections = {}
    runner._flashalign_mm_types = {}
    model._flashalign_runtime = runtime
    for index, layer in enumerate(base.layers):
        layer._flashalign_runtime = runtime
        layer._flashalign_layer = index
        if layer.engram is not None:
            layer.engram._flashalign_runtime = runtime
            layer.engram._flashalign_layer = index
    for name, (basis, scale) in list(profile.operators.items()):
        profile.operators[name] = basis.to(runner.device), scale
    runner.model_state.encoder_runner._flashalign_runner = runner
    runner.model_state.encoder_runner.encoder_cache._flashalign_runner = runner


def register():
    global REGISTERED
    if REGISTERED or not os.environ.get('FLASHALIGN_DEEPSEEK_PROFILES'):
        return
    if os.environ.get('FLASHALIGN_NATIVE_SECURE_LAUNCH') != '1':
        raise RuntimeError('Use the FlashAlign native launcher')
    import vllm
    engine_root = Path(vllm.__file__).resolve().parent
    pins = json.loads((Path(__file__).parent / 'source-pins.json').read_text())
    for name, expected in pins.items():
        if hashlib.sha256((engine_root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unrecognized pinned engine source: '+name)
    from .overlay import install
    install(engine_root)
    native_requested = any(getattr(p, 'is_native_wo', False) for p in
                           load_profiles(json.loads(os.environ['FLASHALIGN_DEEPSEEK_PROFILES'])).values())
    if native_requested:
        if os.environ.get('VLLM_PLUGINS', '').strip() != 'flashalign_native':
            raise ValueError('The native profile requires only the flashalign_native plugin')
        from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
        from .native_attach import wrap_native_setup
        wrap_native_setup(DeepseekV4Attention)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    from vllm.multimodal.utils import get_mm_features_in_window

    original_load = GPUModelRunner.load_model
    @wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        attach(self, load_profiles(json.loads(os.environ['FLASHALIGN_DEEPSEEK_PROFILES'])))
        return result
    GPUModelRunner.load_model = load

    original_add = GPUModelRunner.add_requests
    @wraps(original_add)
    def add(self, scheduler_output):
        result = original_add(self, scheduler_output)
        for request in scheduler_output.scheduled_new_reqs:
            self._flashalign_selections.pop(request.req_id, None)
            extra = getattr(request.sampling_params, 'extra_args', None) or {}
            name = extra.get('_flashalign_profile')
            if name is not None:
                profile = self._flashalign_profiles.get(name)
                scale = extra.get('_flashalign_scale')
                if profile is None or profile.digest != extra.get('_flashalign_digest'):
                    raise ValueError('Worker profile identity mismatch')
                if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
                    raise ValueError('Invalid worker profile strength')
                if getattr(profile, 'is_native_wo', False):
                    from .native_profile import validate_native_selection
                    validate_native_selection(profile, scale)
                self._flashalign_selections[request.req_id] = Selection(name, profile.digest, float(scale))
            for feature in request.mm_features:
                if feature.data is not None and 'types' in feature.data:
                    self._flashalign_mm_types[feature.identifier] = feature.data['types'].data.detach().clone().flatten()
        return result
    GPUModelRunner.add_requests = add

    original_remove = GPUModelRunner._remove_request
    @wraps(original_remove)
    def remove(self, request_id):
        self._flashalign_selections.pop(request_id, None)
        return original_remove(self, request_id)
    GPUModelRunner._remove_request = remove

    original_prepare = GPUModelRunner.prepare_inputs
    @wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        batch = original_prepare(self, *args, **kwargs)
        native = getattr(self, '_flashalign_native_runtime', None)
        if native is not None:
            native.prepare(batch, self._flashalign_selections)
        else:
            self._flashalign_runtime.prepare(batch, self._flashalign_selections)
        return batch
    GPUModelRunner.prepare_inputs = prepare

    original_gather = EncoderRunner.gather_mm_embeddings
    @wraps(original_gather)
    def gather(self, req_ids, total, num_scheduled, query_starts, prefill_lens,
               num_computed, draft_lookahead=0):
        outputs, mask = original_gather(self, req_ids, total, num_scheduled,
            query_starts, prefill_lens, num_computed, draft_lookahead)
        runner = getattr(self, '_flashalign_runner', None)
        if runner is not None and getattr(runner, '_flashalign_native_runtime', None) is not None:
            return outputs, mask
        if runner is None or draft_lookahead or not any(r in runner._flashalign_selections for r in req_ids):
            return outputs, mask
        descriptions = []
        for index, request_id in enumerate(req_ids):
            if num_computed[index] >= prefill_lens[index]:
                continue
            start = int(num_computed[index])
            end = start + int(num_scheduled[index])
            features = self.encoder_cache.mm_features[request_id]
            lo, hi = get_mm_features_in_window(features, start=start, end=end)
            for feature in features[lo:hi]:
                pos = feature.mm_position
                a, b = max(start-pos.offset, 0), min(end-pos.offset, pos.length)
                first, last = pos.get_embeds_indices_in_range(a, b)
                if first == last:
                    continue
                cached = self.encoder_cache.encoder_outputs.get(feature.identifier)
                if cached is None:
                    if pos.offset >= end:
                        continue
                    raise ValueError('Encoder cache miss during owner projection')
                if pos.is_embed is None:
                    first, last = a, b
                descriptions.append((runner._flashalign_selections.get(request_id),
                                     feature.identifier, first, last, cached))
        if len(descriptions) != len(outputs):
            raise ValueError('Image batch ordering differs from the pinned path')
        result = []
        for output, (selection, identifier, start, stop, cached) in zip(outputs, descriptions):
            if selection is None:
                result.append(output)
                continue
            types = runner._flashalign_mm_types.get(identifier)
            if types is None or types.numel() != cached.shape[0]:
                raise ValueError('Missing cached image token types')
            result.append(image_project(output, types[start:stop].to(output.device),
                runner._flashalign_profiles[selection.name], selection, runner.model.aligner.w2.bias))
        return result, mask
    EncoderRunner.gather_mm_embeddings = gather

    original_free = EncoderCache.free_encoder_cache
    @wraps(original_free)
    def free(self, identifier):
        runner = getattr(self, '_flashalign_runner', None)
        if runner is not None:
            runner._flashalign_mm_types.pop(identifier, None)
        return original_free(self, identifier)
    EncoderCache.free_encoder_cache = free
    original_reset = EncoderCache.reset_encoder_cache
    @wraps(original_reset)
    def reset(self):
        runner = getattr(self, '_flashalign_runner', None)
        if runner is not None:
            runner._flashalign_mm_types.clear()
        return original_reset(self)
    EncoderCache.reset_encoder_cache = reset
    REGISTERED = True
