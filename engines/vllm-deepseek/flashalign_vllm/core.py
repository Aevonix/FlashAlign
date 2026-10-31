# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Profile loading, HMAC authorization and row-local output projection."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time


class FieldError(ValueError):
    pass


@dataclass(frozen=True)
class Selection:
    name: str
    digest: str
    scale: float


def load_key(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('Signing key must be an owned regular private file')
        if info.st_uid != os.getuid():
            raise ValueError('Signing key belongs to another user')
        key = os.read(fd, 33)
    finally:
        os.close(fd)
    if len(key) != 32:
        raise ValueError('Signing key must contain exactly 32 bytes')
    return key


class Gate:
    def __init__(self, profiles, key, clock=time.time, *, replay_store=None):
        self.profiles, self.key, self.clock = profiles, key, clock
        self.replay_store = replay_store
        self.nonces, self.lock = {}, threading.Lock()

    def authorize(self, raw, body, headers):
        mode = body.get('owner_mode', 'off')
        mode = 'off' if mode is None else mode
        if not isinstance(mode, str):
            raise FieldError('owner_mode must be "off" or an owner-mode profile name')
        scale = body.get('owner_mode_scale', 1.)
        scale = 1. if scale is None else scale
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 2:
            raise FieldError('owner_mode_scale must be a number in (0, 2]')
        if mode == 'off':
            return None
        if not self.profiles:
            raise FieldError('this server was started without owner-mode profiles: owner_mode must be "off"')
        if self.key is None or len(headers) != 1:
            return None
        match = re.fullmatch(r'v1:([0-9]{1,12}):([A-Za-z0-9_-]{16,128}):([0-9a-f]{64})', headers[0])
        if not match:
            return None
        stamp, nonce, supplied = match.groups()
        now = self.clock()
        if abs(now-int(stamp)) > 60:
            return None
        message = stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
        wanted = hmac.new(self.key, message.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(wanted, supplied):
            return None
        if self.replay_store is not None:
            if not self.replay_store.consume(nonce, int(stamp)+61, now=now):
                return None
        else:
            with self.lock:
                self.nonces = {n:until for n,until in self.nonces.items() if until >= now}
                if nonce in self.nonces or len(self.nonces) >= 100_000:
                    return None
                self.nonces[nonce] = int(stamp)+61
        if mode not in self.profiles:
            raise FieldError('owner_mode names no profile this server loaded')
        return Selection(mode, self.profiles[mode].digest, float(scale))


class Profile:
    def __init__(self, directory):
        import torch
        from safetensors.torch import load_file
        root = Path(directory)
        raw = (root/'profile.json').read_bytes()
        if len(raw) > 4*1024*1024:
            raise ValueError('Oversized profile manifest')
        meta = json.loads(raw)
        identity=meta.get('model')
        if not isinstance(identity,dict) or not all(isinstance(identity.get(x),str) and identity[x] for x in ('id','revision')):
            raise ValueError('Profile must identify a pinned model')
        name = meta.get('name', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', name) or name == 'off':
            raise ValueError('Invalid profile name')
        if meta.get('transform', 'projection') != 'projection' or meta.get('requires', []) != []:
            raise ValueError('This adapter only supports standard output projections')
        hidden = meta.get('hidden_size')
        if type(hidden) is not int or hidden <= 0:
            raise ValueError('Invalid profile hidden size')
        data = root/'profile.safetensors'
        if data.stat().st_size > 256*1024*1024:
            raise ValueError('Oversized profile tensors')
        hashes_path=root/'sha256.json'
        if hashes_path.exists():
            hashes=json.loads(hashes_path.read_text())
            for filename in ('profile.json','profile.safetensors'):
                if hashes.get(filename)!=hashlib.sha256((root/filename).read_bytes()).hexdigest():
                    raise ValueError('Profile checksum mismatch')
        tensors = load_file(data, device='cpu')
        self.name, self.hidden, self.meta = name, hidden, meta
        self.digest = hashlib.sha256(raw+data.read_bytes()).hexdigest()
        self.sites = {}
        self.applied_rows = {}
        used = set()
        for row in meta.get('sites', []):
            layer, site = row.get('layer'), row.get('site')
            rank, scale = row.get('rank'), row.get('scale')
            if type(layer) is not int or layer < 0 or site not in ('embedding_out','attn_out','mlp_out'):
                raise ValueError('Unsupported profile site')
            if site == 'embedding_out' and layer != 0:
                raise ValueError('Embedding site must have layer zero')
            if type(rank) is not int or not 1 <= rank <= min(128, hidden):
                raise ValueError('Invalid profile rank')
            if type(scale) not in (int,float) or not math.isfinite(scale):
                raise ValueError('Invalid site scale')
            if (layer,site) in self.sites:
                raise ValueError('Duplicate profile site')
            key = f'L{layer}.{site}.Q'
            q = tensors.get(key)
            if q is None or q.dtype != torch.float32 or tuple(q.shape) != (rank,hidden):
                raise ValueError('Missing or invalid FP32 basis')
            if not bool(torch.isfinite(q).all()) or not torch.allclose(q@q.T, torch.eye(rank), atol=2e-5,rtol=0):
                raise ValueError('Non-orthonormal profile basis')
            self.sites[(layer,site)] = (q, float(scale), row.get('tensor'))
            used.add(key)
        if not used or used != set(tensors):
            raise ValueError('Missing or unexpected profile tensors')


def load_profiles(paths):
    profiles = {}
    for path in paths:
        profile = Profile(path)
        if profile.name in profiles:
            raise ValueError('Duplicate profile names')
        profiles[profile.name] = profile
    return profiles


def cache_key(selection, store, model_identity, original=None):
    mode = 'off' if selection is None else selection.name
    digest = '' if selection is None else selection.digest
    scale = '' if selection is None else selection.scale.hex()
    material = json.dumps([model_identity,mode,digest,scale,original],sort_keys=True,separators=(',',':'))
    return 'flashalign:'+hashlib.sha256(material.encode()).hexdigest()


# Each span is (start, end, Selection). Context belongs to one model call.
_spans = ContextVar('flashalign_vllm_spans', default=())


@contextmanager
def selected_rows(spans):
    token = _spans.set(tuple(spans))
    try:
        yield
    finally:
        _spans.reset(token)


def project(output, profiles, layer, site):
    """Off rows never participate in transform arithmetic."""
    groups = []
    for start, end, selection in _spans.get():
        profile = profiles.get(selection.name)
        if profile is None or profile.digest != selection.digest:
            raise ValueError('Unknown profile selection in worker')
        operator = profile.sites.get((layer,site))
        if operator is not None and end > start:
            groups.append((start,end,selection.scale,operator,profile))
    if not groups:
        return output
    result = output.clone()
    for start,end,alpha,(q,scale,_),profile in groups:
        if start < 0 or end > output.shape[0]:
            raise ValueError('Profile row map differs from model output')
        q = q.to(device=output.device)
        value = output[start:end].float()
        shifted = value-alpha*scale*((value@q.T)@q)
        result[start:end].copy_(shifted.to(output.dtype))
        counter=f'{layer}.{site}'
        profile.applied_rows[counter]=profile.applied_rows.get(counter,0)+(end-start)
    return result


def spans_for_runner(runner, profiles):
    spans = []
    ids = runner.input_batch.req_ids
    starts = runner.query_start_loc.np
    for index, request_id in enumerate(ids):
        request = runner.requests[request_id]
        extra = getattr(request.sampling_params, 'extra_args', None) or {}
        mode = extra.get('_flashalign_profile')
        if mode is None:
            continue
        profile = profiles.get(mode)
        if profile is None or extra.get('_flashalign_digest') != profile.digest:
            raise ValueError('Worker profile identity mismatch')
        scale = extra.get('_flashalign_scale')
        if type(scale) not in (int,float) or not math.isfinite(scale) or not 0 < scale <= 2:
            raise ValueError('Invalid worker profile strength')
        spans.append((int(starts[index]),int(starts[index+1]),Selection(mode,profile.digest,float(scale))))
    return spans
