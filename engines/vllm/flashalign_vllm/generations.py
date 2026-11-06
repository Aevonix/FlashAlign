# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Generation-owned projection hooks for the standard vLLM runner."""
from pathlib import Path

from owner_mode.profile_lifecycle import LifecycleError, ResourceLimit
from .core import Profile, project
from .lifecycle import WorkerRegistry


class ProjectionAdapter:
    def __init__(self, runner, profiles):
        self.runner, self.profiles = runner, profiles
        identities = {(p.meta['model']['id'], p.meta['model']['revision']) for p in profiles.values()}
        if len(identities) != 1:
            raise ValueError('Generations must target one pinned model revision')
        self.identity = next(iter(identities))
        self.model = runner.get_model()
        self.modules = dict(self.model.named_modules())
        self.prefix = 'language_model.model' if hasattr(self.model, 'language_model') else 'model'
        self.hf = runner.vllm_config.model_config.hf_text_config
        self.hooks = {}
        self.registry = None

    def inventory(self):
        import torch
        free = torch.cuda.mem_get_info(self.runner.device)[0] if str(self.runner.device).startswith('cuda') else 1 << 40
        return {'model_id': self.identity[0], 'model_revision': self.identity[1],
                'operators': ['projection'], 'available_device_bytes': free}

    def target(self, layer, site):
        if site == 'embedding_out':
            return self.prefix + '.embed_tokens', 'model.language_model.embed_tokens.weight'
        if site == 'mlp_out':
            part = 'mlp.down_proj'
        else:
            kinds = getattr(self.hf, 'layer_types', ())
            part = 'linear_attn.out_proj' if kinds and kinds[layer] == 'linear_attention' else 'self_attn.o_proj'
        return f'{self.prefix}.layers.{layer}.{part}', f'model.language_model.layers.{layer}.{part}.weight'

    def statistics(self):
        import torch
        profiles = [row.handle for row in self.registry.generations.values() if row.committed]
        expected = set(site for profile in profiles for site in profile.sites)
        exercised = set(key for profile in profiles for key, count in profile.applied_rows.items() if count)
        config = self.runner.vllm_config
        spec = getattr(config, 'speculative_config', None)
        return {'graph_mode': 'eager', 'speculative_method': getattr(spec, 'method', None),
                'expected_sites': len(expected), 'installed_sites': len(self.hooks),
                'exercised_sites': len(exercised), 'active_requests': len(getattr(self.runner, 'requests', {})),
                'gpu_memory_bytes': torch.cuda.memory_allocated(self.runner.device)
                    if str(self.runner.device).startswith('cuda') else 0}

    def validate(self, artifact):
        if ((artifact.model_id, artifact.model_revision) != self.identity
                or artifact.operator != 'projection' or artifact.requirements):
            raise LifecycleError('Profile differs from the loaded model or supported operator')

    def prepare(self, generation_id, artifact):
        import torch
        self.validate(artifact)
        profile = Profile(artifact.directory)
        if profile.hidden != self.hf.hidden_size:
            raise LifecycleError('Profile width differs from the loaded model')
        required = sum(q.numel() * q.element_size() for q, _, _ in profile.sites.values())
        if str(self.runner.device).startswith('cuda') and required > torch.cuda.mem_get_info(self.runner.device)[0]:
            raise ResourceLimit('Insufficient device memory for overlapping generations')
        targets = {}
        for (layer, site), (basis, scale, tensor) in profile.sites.items():
            if layer >= self.hf.num_hidden_layers:
                raise LifecycleError('Profile layer is outside the loaded model')
            name, expected = self.target(layer, site)
            if tensor != expected or name not in self.modules:
                raise LifecycleError('Profile target differs from the loaded module')
            targets[name] = (layer, site)
        # Allocate everything before changing hooks. Hooks are site-generic and
        # resolve the immutable generation carried by each scheduled request.
        profile.sites = {key: (basis.to(self.runner.device), scale, tensor)
                         for key, (basis, scale, tensor) in profile.sites.items()}
        profile.runtime_digest, profile.digest = profile.digest, artifact.digest
        # Reserve tracking entries before a hook can become attached. Keep
        # publication local until every registration and dictionary allocation
        # succeeds, so rollback never mutates the existing hook map.
        pending = {name: None for name in targets if name not in self.hooks}
        try:
            for name in pending:
                layer, site = targets[name]
                def hook(module, args, output, layer=layer, site=site):
                    store = self.registry if self.registry is not None else self.profiles
                    if isinstance(output, tuple):
                        first = project(output[0], store, layer, site)
                        return output if first is output[0] else (first, *output[1:])
                    return project(output, store, layer, site)
                pending[name] = self.modules[name].register_forward_hook(hook)
            merged = self.hooks.copy()
            merged.update(pending)
        except BaseException:
            for handle in reversed(pending.values()):
                if handle is not None:
                    handle.remove()
            raise
        self.hooks = merged
        return profile

    def references(self, generation_id):
        return sum(1 for request in getattr(self.runner, 'requests', {}).values()
                   if (getattr(request.sampling_params, 'extra_args', None) or {}).get('_flashalign_generation') == generation_id)

    def all_references(self):
        return len(getattr(self.runner, 'requests', {}))

    def retire(self, generation_id, handle):
        import torch
        if str(self.runner.device).startswith('cuda'):
            # The worker executes internal RPC between scheduler steps. Drain
            # queued GPU work before releasing the last tensor reference.
            torch.cuda.synchronize(self.runner.device)


def install_generations(runner, profiles):
    adapter = ProjectionAdapter(runner, profiles)
    registry = WorkerRegistry(adapter, '0')
    adapter.registry = registry
    runner._flashalign_lifecycle = registry
    # Startup hooks preserve the previous static API for callers that do not
    # enable the local admin plane. Runtime preparation adds any new sites.
    for profile in profiles.values():
        for layer, site in profile.sites:
            name, _ = adapter.target(layer, site)
            adapter.hooks[name] = True
    return registry
