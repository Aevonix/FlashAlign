# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Normalized GLM5-Next output hooks for pinned SGLang and native MTP."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MethodType

import torch
from safetensors.torch import load_file

from .gate import Gate, PREFIX, read_key
from .flashalign_replay import deployment_replay_store

_active = ContextVar('flashalign_glm_forward', default=None)
_audit_seen = set()
ROOT = Path(__file__).resolve().parents[1]
MODEL = {'id': 'zai-org/GLM-5.3-Flash', 'revision': '46a050bec13002ad29ee795c16e61cf4fa3c4a9c'}
REQUIRES = {'row_gain', 'row_norm_preconditioning'}


@dataclass
class Profile:
    name: str
    digest: str
    model: dict
    sites: dict


def load_profile(directory):
    path = Path(directory)
    raw = (path/'profile.json').read_bytes()
    if len(raw) > 4*1024*1024:
        raise ValueError('Oversized profile metadata')
    meta = json.loads(raw)
    if meta.get('model') != MODEL:
        raise ValueError('Profile does not match the supported GLM model')
    if meta.get('transform') != 'row_normalized_projection_v1' or set(meta.get('requires', [])) != REQUIRES:
        raise ValueError('Unsupported GLM normalized operator')
    if type(meta.get('hidden_size')) is not int or meta['hidden_size'] != 4096:
        raise ValueError('Unsupported GLM hidden width')
    name = meta.get('name', '')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', name) or name == 'off':
        raise ValueError('Invalid profile name')
    binary = path/'profile.safetensors'
    if binary.stat().st_size > 16*1024*1024:
        raise ValueError('Oversized GLM operator')
    tensors = load_file(str(binary), device='cpu')
    expected = {(s['layer'], s['site']): s['tensor'] for s in json.loads((ROOT/'expected-sites.json').read_text())}
    sites, keys = {}, set()
    for site in meta.get('sites', []):
        layer, kind = site.get('layer'), site.get('site')
        if type(layer) is not int or (layer, kind) not in expected or site.get('tensor') != expected[(layer, kind)]:
            raise ValueError('Unsupported GLM writer mapping')
        if (layer, kind) in sites or type(site.get('rank')) is not int or site['rank'] != 1:
            raise ValueError('Duplicate writer or unsupported GLM rank')
        strength = site.get('scale')
        if type(strength) not in (int, float) or not math.isfinite(strength):
            raise ValueError('Invalid fitted projection strength')
        prefix = f'L{layer}.{kind}.'
        names = {label: prefix+label for label in ('Q', 'm', 'g')}
        if not set(names.values()) <= set(tensors):
            raise ValueError('Missing normalized operator tensor')
        q, m, g = (tensors[names[label]] for label in ('Q', 'm', 'g'))
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (q, m, g)):
            raise ValueError('Operator tensors must be finite FP32')
        if tuple(q.shape) != (1, 4096) or tuple(m.shape) != (4096,) or tuple(g.shape) != (4096,):
            raise ValueError('Invalid GLM operator shape')
        if not torch.allclose(q@q.T, torch.eye(1), atol=2e-5, rtol=0) or not (m > 0).all() or not (g > 0).all():
            raise ValueError('Invalid GLM normalized basis/gain/norm')
        sites[(layer, kind)] = {'q': q, 'm': m, 'g': g, 'scale': float(strength)}
        keys.update(names.values())
    if set(sites) != set(expected) or set(tensors) != keys:
        raise ValueError('Incomplete GLM selected writer/MTP coverage or unexpected tensors')
    return Profile(name, hashlib.sha256(raw+binary.read_bytes()).hexdigest(), MODEL, sites)


@lru_cache(maxsize=1)
def registry():
    profiles = {}
    for value in os.environ.get('FLASHALIGN_PROFILES', '').split(','):
        if value.strip():
            profile = load_profile(value.strip())
            if profile.name in profiles:
                raise ValueError('Duplicate profile name')
            profiles[profile.name] = profile
    metadata = json.loads((ROOT/'model-metadata.json').read_text())
    scope = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if profiles and os.environ.get('FLASHALIGN_MODEL_SCOPE') != scope:
        raise ValueError('Profile does not match running GLM model fingerprints')
    key = read_key(os.environ.get('FLASHALIGN_KEY_FILE', '')) if profiles else None
    replay_store = deployment_replay_store(key) if profiles else None
    return profiles, Gate(profiles, key, model_scope=scope, replay_store=replay_store)


def check_server_args(args):
    profiles, _ = registry()
    if not profiles:
        return
    for field in ('pp_size', 'dp_size', 'dcp_size', 'attn_cp_size', 'nnodes', 'tokenizer_worker_num'):
        if getattr(args, field, 1) != 1:
            raise ValueError('FlashAlign GLM backend requires one worker and PP/DP/CP=1')
    if getattr(args, 'tp_size', 1) != 4 or getattr(args, 'ep_size', 1) != 1:
        raise ValueError('FlashAlign GLM backend requires TP4/EP1')
    if not getattr(args, 'disable_shared_experts_fusion', False):
        raise ValueError('GLM selected shared-expert writers require shared-expert fusion off')
    if getattr(args, 'moe_runner_backend', 'auto') != 'triton':
        raise ValueError('FlashAlign GLM backend requires the separate shared-expert Triton path')
    if not getattr(args, 'disable_cuda_graph', False):
        if any(getattr(args, f'cuda_graph_backend_{phase}', None) != 'disabled' for phase in ('decode', 'prefill')):
            raise ValueError('FlashAlign GLM backend requires eager execution; active speculation is preserved')
    algorithm = getattr(args, 'speculative_algorithm', None)
    if algorithm not in ('EAGLE', 'NEXTN'):
        raise ValueError('Unsupported active speculative algorithm; it will not be disabled or replaced')
    if algorithm is not None and getattr(args, 'speculative_eagle_topk', 1) != 1:
        raise ValueError('FlashAlign GLM native MTP backend requires top-k one')
    draft = getattr(args, 'speculative_draft_model_path', None)
    if draft is not None and draft != getattr(args, 'model_path', None):
        raise ValueError('GLM native MTP must use the pinned checkpoint')
    for field in ('enable_torch_compile', 'enable_lora', 'enable_custom_logit_processor', 'log_requests',
                  'enable_hierarchical_cache', 'enable_dp_attention', 'enable_eplb', 'enable_two_batch_overlap'):
        if getattr(args, field, False):
            raise ValueError('Unsupported FlashAlign GLM option: '+field)
    if getattr(args, 'moe_a2a_backend', 'none') != 'none' or getattr(args, 'disaggregation_mode', 'null') not in (None, 'null'):
        raise ValueError('FlashAlign GLM backend requires local unsharded-token execution')
    if os.environ.get('SGLANG_RAGGED_VERIFY_MODE', 'static') != 'static':
        raise ValueError('GLM native MTP requires static verification layout')
    if os.environ.get('SGLANG_RUST_SERVER', '').lower() in ('1', 'true', 'on'):
        raise ValueError('FlashAlign GLM backend requires its authenticated Python HTTP boundary')
    os.environ['FLASHALIGN_HTTP_GATE'] = '1'


def row_ranges(batch, rows):
    salts = getattr(batch, 'flashalign_cache_salts', None)
    if not salts:
        return []
    count, mode = len(salts), batch.forward_mode
    if mode.is_target_verify() or getattr(mode, 'is_draft_extend_v2', lambda: False)():
        width = getattr(batch.spec_info, 'num_tokens_per_req', None)
        if width is None:
            width = getattr(batch.spec_info, 'draft_token_num', None)
        if type(width) is not int or width <= 0:
            raise ValueError('Unknown GLM native MTP row layout')
        lengths = [width]*count
    elif mode.is_decode():
        if getattr(getattr(batch, 'spec_info', None), 'num_tokens_per_req', 1) != 1:
            raise ValueError('Branching GLM native MTP is unsupported')
        lengths = [1]*count
    elif mode.is_extend():
        lengths = getattr(batch, 'extend_seq_lens_cpu', None)
        if lengths is None:
            raise ValueError('Missing GLM extension lengths')
        lengths = list(lengths)
    elif mode.is_idle():
        return []
    else:
        raise ValueError('Unsupported GLM forward mode')
    if len(lengths) != count or any(type(x) is not int or x < 0 for x in lengths) or sum(lengths) > rows:
        raise ValueError('GLM row count differs from request layout')
    result, start = [], 0
    for salt, length in zip(salts, lengths):
        result.append((salt, start, length)); start += length
    return result


def normalized(value, operator, alpha):
    if alpha == 0:
        return value
    y = value.float()
    q, m, g, scale = (operator[k] for k in ('q', 'm', 'g', 'scale'))
    z = y/m
    transformed = (m*g)*(z-scale*((z@q.T)@q))
    return (y+alpha*(transformed-y)).to(value.dtype)


def audit_record(label, record):
    audit = os.environ.get('FLASHALIGN_AUDIT_DIR')
    if not audit:
        return
    import torch.distributed as distributed
    rank = distributed.get_rank() if distributed.is_initialized() else None
    world = distributed.get_world_size() if distributed.is_initialized() else None
    record = {**record, 'pid': os.getpid(), 'rank': rank, 'world_size': world}
    directory = Path(audit)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory/(hashlib.sha256((label+repr(record)).encode()).hexdigest()+'.json')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(record, handle)


def project(output, batch, site, profiles, gate):
    salts = getattr(batch, 'flashalign_cache_salts', None)
    if not salts or all(not isinstance(s, str) or not s.startswith(PREFIX) or s.startswith(PREFIX+'off.') for s in salts):
        return output
    selected = []
    for salt, start, length in row_ranges(batch, output.shape[0]):
        selection = gate.from_salt(salt)
        if selection is not None and length:
            operator = profiles[selection.name].sites.get(site)
            if operator is not None:
                selected.append((selection, start, length, operator))
    if not selected:
        return output
    result = output.clone()
    for selection, start, length, operator in selected:
        result[start:start+length] = normalized(output[start:start+length], operator, selection.scale)
    audit = os.environ.get('FLASHALIGN_AUDIT_DIR')
    if audit:
        active = _active.get()
        role = 'native_mtp' if active and active[0].is_draft_worker else 'target'
        phase = getattr(batch.forward_mode, 'name', 'unknown')
        identity = (os.getpid(), role, site, phase, tuple(s.scale for s, _, _, _ in selected), len(selected) < len(salts))
        if identity not in _audit_seen:
            _audit_seen.add(identity)
            off = [(a,n) for salt,a,n in row_ranges(batch,output.shape[0]) if gate.from_salt(salt) is None]
            unchanged = all(torch.equal(output[a:a+n].contiguous().view(torch.uint8), result[a:a+n].contiguous().view(torch.uint8)) for a,n in off)
            record = {'role':role,'site':list(site),'phase':phase,'selected_rows':sum(n for _,_,n,_ in selected),
                      'rows':output.shape[0],'mixed_owner_and_stock':len(selected)<len(salts),
                      'stock_rows_unchanged':unchanged,'activation_dtype':str(output.dtype)}
            record['kind'] = 'projection'
            record['stock_rows_compared'] = sum(n for _,n in off)
            audit_record(repr(identity), record)
    return result


def install(runner):
    profiles, gate = registry()
    if not profiles:
        return
    if os.environ.get('FLASHALIGN_HTTP_GATE') != '1':
        raise ValueError('GLM profiles require authenticated HTTP launch')
    draft = runner.is_draft_worker
    expected = 'Glm5NextForConditionalGenerationNextN' if draft else 'Glm5NextForConditionalGeneration'
    if type(runner.model).__name__ != expected:
        raise ValueError('GLM model role/architecture mismatch')
    modules = dict(runner.model.named_modules())
    bindings = {}
    for profile in profiles.values():
        for site, operator in profile.sites.items():
            layer, kind = site
            if (layer == 45) != draft:
                continue
            prefix = 'model.decoder' if draft else f'model.layers.{layer}'
            name = prefix+('.self_attn.o_proj' if kind == 'attn_out' else '.mlp')
            if name not in modules:
                raise ValueError('Missing GLM writer: '+name)
            module = modules[name]
            if kind == 'shared_expert_out':
                if (getattr(module, 'num_fused_shared_experts', None) != 0 or
                    getattr(module, '_fuse_shared_experts_inside_sbo', False) or
                    getattr(module, '_enable_a2a_moe', False) or not hasattr(module, 'shared_experts') or
                    not callable(getattr(module, '_forward_shared_experts', None))):
                    raise ValueError('Shared writer is fused or has no isolated output boundary')
            elif getattr(module, 'output_size', None) != 4096:
                raise ValueError('GLM attention output width mismatch')
            for key in ('q', 'm', 'g'):
                operator[key] = operator[key].to(runner.device)
            if name in bindings and bindings[name] != site:
                raise ValueError('Conflicting GLM writer binding')
            bindings[name] = site
    handles = []
    for name, site in bindings.items():
        module = modules[name]
        if site[1] == 'shared_expert_out':
            original = module._forward_shared_experts
            def shared(self, *args, original=original, site=site, **kwargs):
                output = original(*args, **kwargs)
                active = _active.get()
                if output is None or active is None or active[0] is not runner:
                    return output
                return project(output, active[1], site, profiles, gate)
            module._forward_shared_experts = MethodType(shared, module)
            handles.append((module, original))
        else:
            def hook(module, inputs, output, site=site):
                active = _active.get()
                if active is None or active[0] is not runner:
                    return output
                value = output[0] if isinstance(output, tuple) else output
                updated = project(value, active[1], site, profiles, gate)
                return (updated, *output[1:]) if isinstance(output, tuple) else updated
            handles.append(module.register_forward_hook(hook))
    runner.flashalign_handles = handles
    audit_record('install:'+str(draft), {'kind': 'install',
                 'role': 'native_mtp' if draft else 'target',
                 'installed_sites': [list(site) for site in bindings.values()],
                 'model_class': type(runner.model).__name__})


def owner_forward(function):
    @wraps(function)
    def wrapped(self, forward_batch, *args, **kwargs):
        if not getattr(self, 'flashalign_handles', None):
            return function(self, forward_batch, *args, **kwargs)
        token = _active.set((self, forward_batch))
        try:
            return function(self, forward_batch, *args, **kwargs)
        finally:
            _active.reset(token)
    return wrapped
