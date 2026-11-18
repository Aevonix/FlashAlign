# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Strict loading of low-rank matrix-input profiles and their CPU reference.

A profile of this kind adds a low-rank matrix to a writer matrix:

    W' = W + strength * scale * Q^T B

Q [rank, hidden] is orthonormal and B [rank, input_size] holds the input side.
The runtime never changes W. It adds (x B^T) Q to the writer's output for the
rows that selected the profile.
"""
import hashlib
import json
import math
import re
from pathlib import Path

TRANSFORM = 'input_low_rank_delta_v1'
MODEL = {'id': 'deepseek-ai/DeepSeek-V4.1-Flash-1017', 'revision': '1bae0e1088993b9b82de08f59de5be5a34e42170'}
HIDDEN = 5120
LAYERS = 40
MAX_RANK = 128
MAX_MANIFEST = 4 * 1024 ** 2
MAX_PAYLOAD = 256 * 1024 ** 2
# Site -> (writer tensor, writer input width).
SITES = {'attn_out': ('layers.{layer}.attn.wo_b.weight', 8192),
         'shared_expert_out': ('layers.{layer}.ffn.shared_experts.w2.weight', 2304)}


def is_delta_manifest(declared):
    return isinstance(declared, dict) and declared.get('transform') == TRANSFORM


def read_manifest(directory):
    """Identity checks that need no tensor library. Returns (name, digest, manifest, payload path)."""
    root = Path(directory)
    raw = (root / 'profile.json').read_bytes()
    data = root / 'profile.safetensors'
    if len(raw) > MAX_MANIFEST or data.stat().st_size > MAX_PAYLOAD:
        raise ValueError('Oversized low-rank profile')
    payload = data.read_bytes()
    meta = json.loads(raw)
    if (not isinstance(meta, dict) or meta.get('transform') != TRANSFORM or meta.get('requires') != ['matrix_input']
            or meta.get('schema') is not None or meta.get('format') is not None):
        raise ValueError('Unsupported low-rank profile schema or transform')
    if meta.get('hidden_size') != HIDDEN:
        raise ValueError('DeepSeek hidden width must be 5120')
    if meta.get('model') != MODEL:
        raise ValueError('Profile must identify pinned DeepSeek V4.1 Flash 1017')
    name = meta.get('name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', name) or name == 'off':
        raise ValueError('Invalid profile name')
    hashes = json.loads((root / 'sha256.json').read_text())
    for filename, body in (('profile.json', raw), ('profile.safetensors', payload)):
        if not isinstance(hashes, dict) or hashes.get(filename) != hashlib.sha256(body).hexdigest():
            raise ValueError('Profile checksum mismatch')
    return name, hashlib.sha256(raw + payload).hexdigest(), meta, data


class DeltaProfile:
    is_delta = True

    def __init__(self, directory):
        import torch
        from safetensors.torch import load_file
        name, digest, meta, data = read_manifest(directory)
        tensors = load_file(data, device='cpu')
        self.sites, used = {}, set()
        rows = meta.get('sites')
        if not isinstance(rows, list) or not rows:
            raise ValueError('Empty profile')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Invalid profile site')
            layer, site, rank, scale = row.get('layer'), row.get('site'), row.get('rank'), row.get('scale')
            if type(layer) is not int or not 0 <= layer < LAYERS or site not in SITES:
                raise ValueError('Unsupported low-rank profile site')
            tensor, width = SITES[site]
            if row.get('tensor') != tensor.format(layer=layer) or row.get('input_size') != width:
                raise ValueError('Profile tensor does not match its native site')
            if type(rank) is not int or not 1 <= rank <= MAX_RANK:
                raise ValueError('Invalid profile rank')
            if type(scale) not in (int, float) or not math.isfinite(scale):
                raise ValueError('Invalid site scale')
            if (layer, site) in self.sites:
                raise ValueError('Duplicate profile site')
            prefix = f'L{layer}.{site}'
            q, b = tensors.get(prefix + '.Q'), tensors.get(prefix + '.B')
            if q is None or q.dtype != torch.float32 or tuple(q.shape) != (rank, HIDDEN):
                raise ValueError('Missing or invalid FP32 basis')
            if not bool(torch.isfinite(q).all()) or not torch.allclose(q @ q.T, torch.eye(rank), atol=2e-5, rtol=0):
                raise ValueError('Non-orthonormal profile basis')
            if (b is None or b.dtype != torch.float32 or tuple(b.shape) != (rank, width)
                    or not bool(torch.isfinite(b).all())):
                raise ValueError('Missing or invalid FP32 input factor')
            self.sites[(layer, site)] = (q.contiguous(), b.contiguous(), float(scale))
            used.update((prefix + '.Q', prefix + '.B'))
        if used != set(tensors):
            raise ValueError('Missing or unexpected profile tensors')
        self.name, self.meta, self.digest = name, meta, digest


def load_delta_profiles(paths):
    """Profiles in the given order. That order is the engine's fixed application order."""
    result = {}
    if not isinstance(paths, list) or len(paths) > 7 or any(not isinstance(path, str) for path in paths):
        raise ValueError('Up to seven low-rank profile directories are supported')
    for path in paths:
        profile = DeltaProfile(path)
        if profile.name in result:
            raise ValueError('Duplicate profile name')
        result[profile.name] = profile
    return result


def local_columns(b, tp_rank, tp_size, width):
    """This rank's columns of an input factor B [rank, input_size].

    The engine pads a writer's input axis with zeros to tp_size equal shards of
    `width` columns and gives rank p the columns [p * width, (p + 1) * width).
    B is padded and cut the same way.
    """
    import torch
    if (b.ndim != 2 or type(tp_rank) is not int or type(tp_size) is not int or type(width) is not int
            or not 0 <= tp_rank < tp_size or width <= 0):
        raise ValueError('Invalid tensor-parallel rank')
    pad = width * tp_size - b.shape[1]
    if not 0 <= pad < width:
        raise ValueError('Low-rank profile input differs from the writer partition of this rank')
    if pad:
        b = torch.cat([b, b.new_zeros((b.shape[0], pad))], dim=1)
    return b[:, tp_rank * width:(tp_rank + 1) * width].contiguous()


def reference_delta(output, source, q, b, rows, site_scale):
    """CPU reference for one site: only selected rows enter arithmetic.

    output [n, hidden] and source [n, input] are this rank's local tensors; rows is
    one (selected, strength) pair per row. Returns a new tensor with the dtype of output.
    """
    result = output.clone()
    for index, (selected, strength) in enumerate(rows):
        if not selected:
            continue
        coefficients = b.float() @ source[index].float()
        changed = output[index].float() + (strength * site_scale) * (coefficients @ q.float())
        result[index] = changed.to(output.dtype)
    return result
