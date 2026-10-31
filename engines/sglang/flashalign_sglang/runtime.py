# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Target-model output hooks for pinned SGLang eager execution."""
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re

import torch
from safetensors.torch import load_file

from .gate import Gate, PREFIX, read_key
from .flashalign_replay import deployment_replay_store

_active = ContextVar('flashalign_sglang_forward',default=None)


@dataclass
class Profile:
    name: str
    digest: str
    model: dict
    hidden: int
    sites: dict


def load_profile(directory):
    path = Path(directory)
    raw = (path/'profile.json').read_bytes()
    meta = json.loads(raw)
    if len(raw)>4*1024*1024 or meta.get('transform','projection')!='projection' or meta.get('requires',[])!=[]:
        raise ValueError('Unsupported FlashAlign profile transform')
    name = meta.get('name','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',name) or name=='off':
        raise ValueError('Invalid FlashAlign profile name')
    hidden = meta.get('hidden_size')
    if type(hidden) is not int or not 0<hidden<=65536:
        raise ValueError('Invalid FlashAlign profile width')
    tensor_file=path/'profile.safetensors'
    if tensor_file.stat().st_size>256*1024*1024:
        raise ValueError('Oversized FlashAlign profile')
    tensors=load_file(str(tensor_file),device='cpu');sites={};used=set()
    for site in meta.get('sites',[]):
        layer,kind,rank,strength=site.get('layer'),site.get('site'),site.get('rank'),site.get('scale')
        if type(layer) is not int or layer<0 or kind not in ('embedding_out','attn_out','mlp_out'):
            raise ValueError('Unsupported FlashAlign profile site')
        if kind=='embedding_out' and layer!=0:
            raise ValueError('Embedding profile site must use layer zero')
        if type(rank) is not int or not 1<=rank<=min(hidden,128) or type(strength) not in (int,float) or not math.isfinite(strength):
            raise ValueError('Invalid FlashAlign rank or scale')
        key=f'L{layer}.{kind}.Q';q=tensors.get(key)
        if q is None or q.dtype!=torch.float32 or tuple(q.shape)!=(rank,hidden) or not torch.isfinite(q).all():
            raise ValueError('Invalid FlashAlign basis tensor')
        if not torch.allclose(q@q.T,torch.eye(rank),atol=2e-5,rtol=0):
            raise ValueError('FlashAlign basis must be orthonormal')
        if (layer,kind) in sites:
            raise ValueError('Duplicate FlashAlign profile site')
        sites[(layer,kind)]={'q':q,'scale':float(strength),'tensor':site.get('tensor')};used.add(key)
    if not sites or used!=set(tensors):
        raise ValueError('Missing or unexpected FlashAlign tensors')
    return Profile(name,hashlib.sha256(raw+tensor_file.read_bytes()).hexdigest(),meta['model'],hidden,sites)


@lru_cache(maxsize=1)
def registry():
    roots=os.environ.get('FLASHALIGN_PROFILES',os.environ.get('TF_OWNER_PROFILES',''))
    profiles={}
    for root in roots.split(','):
        if root.strip():
            profile=load_profile(root.strip())
            if profile.name in profiles: raise ValueError('Duplicate FlashAlign profile name')
            profiles[profile.name]=profile
    key_path=os.environ.get('FLASHALIGN_KEY_FILE',os.environ.get('TF_OWNER_MODE_KEY_FILE',''))
    key=read_key(key_path) if profiles else None
    return profiles,Gate(profiles,key,
                         replay_store=deployment_replay_store(key) if profiles else None)


def check_server_args(args):
    profiles,_=registry()
    if not profiles:return
    for field in ('tp_size','pp_size','dp_size','dcp_size','attn_cp_size','nnodes','tokenizer_worker_num'):
        if getattr(args,field,1)!=1:
            raise ValueError('FlashAlign SGLang currently requires one model and HTTP worker, TP/PP/DP/CP=1')
    if not getattr(args,'disable_cuda_graph',False):
        decode=getattr(args,'disable_decode_cuda_graph',False) or getattr(args,'cuda_graph_backend_decode',None)=='disabled'
        prefill=getattr(args,'disable_prefill_cuda_graph',False) or getattr(args,'cuda_graph_backend_prefill',None)=='disabled'
        if not (decode and prefill):
            raise ValueError('FlashAlign SGLang requires eager execution; set both CUDA graph backends to disabled. Speculative decoding stays enabled')
    if getattr(args,'enable_torch_compile',False) or getattr(args,'enable_lora',False):
        raise ValueError('FlashAlign SGLang does not yet support torch.compile or LoRA composition')
    if getattr(args,'enable_custom_logit_processor',False):
        raise ValueError('FlashAlign SGLang does not expose client-supplied executable logit processors')
    if getattr(args,'log_requests',False):
        raise ValueError('FlashAlign requires request-content logging to remain disabled')
    if getattr(args,'speculative_algorithm',None) not in (None,'DFLASH'):
        raise ValueError('FlashAlign SGLang currently qualifies DFLASH speculation only; the requested algorithm will not be disabled or replaced')
    if os.environ.get('SGLANG_RAGGED_VERIFY_MODE','static')!='static':
        raise ValueError('FlashAlign SGLang currently requires static verification rows; the requested speculative layout will not be changed')
    if getattr(args,'disaggregation_mode','null') not in (None,'null') or getattr(args,'enable_hierarchical_cache',False):
        raise ValueError('FlashAlign SGLang requires local in-memory cache execution')
    if os.environ.get('SGLANG_RUST_SERVER','').lower() in ('1','true','on'):
        raise ValueError('FlashAlign SGLang requires its authenticated Python HTTP server')
    os.environ['FLASHALIGN_HTTP_GATE']='1'


def row_ranges(batch, rows):
    salts=getattr(batch,'flashalign_cache_salts',None)
    if not salts:
        return []
    count=len(salts)
    if batch.forward_mode.is_target_verify():
        width=getattr(batch.spec_info,'num_tokens_per_req',None)
        if width is None:width=getattr(batch.spec_info,'draft_token_num',None)
        if type(width) is not int or width<=0 or width*count!=rows:
            raise ValueError('Unsupported speculative verification row layout')
        lengths=[width]*count
    elif batch.forward_mode.is_decode():
        if rows!=count: raise ValueError('Decode row count does not match requests')
        lengths=[1]*count
    elif batch.forward_mode.is_extend():
        lengths=getattr(batch,'extend_seq_lens_cpu',None)
        if lengths is None:
            raise ValueError('FlashAlign requires explicit per-request extension row lengths')
        lengths=list(lengths)
        if len(lengths)!=count or sum(lengths)!=rows:
            raise ValueError('Extension row count does not match requests')
    elif batch.forward_mode.is_idle():
        return []
    else:
        raise ValueError('Unsupported FlashAlign forward mode')
    result=[];start=0
    for salt,length in zip(salts,lengths):
        result.append((salt,start,int(length)));start+=int(length)
    return result


def project(output,batch,site,profiles,gate):
    salts=getattr(batch,'flashalign_cache_salts',None)
    if not salts or all(not isinstance(s,str) or not s.startswith(PREFIX) or s.startswith(PREFIX+'off.') for s in salts):
        return output
    selected=[]
    for salt,start,length in row_ranges(batch,output.shape[0]):
        selection=gate.from_salt(salt)
        if selection is None or length==0:continue
        operator=profiles[selection.name].sites.get(site)
        if operator is not None:selected.append((selection,start,length,operator))
    if not selected:return output
    result=output.clone()
    for selection,start,length,operator in selected:
        value=output[start:start+length].float()
        q=operator['q']
        result[start:start+length]=(value-selection.scale*operator['scale']*((value@q.T)@q)).to(output.dtype)
    return result


def install(runner):
    if runner.is_draft_worker:return
    profiles,gate=registry()
    if not profiles:return
    if os.environ.get('FLASHALIGN_HTTP_GATE')!='1':
        raise ValueError('FlashAlign profiles require the authenticated HTTP launch path')
    if type(runner.model).__name__ not in ('Qwen3_5ForConditionalGeneration','Qwen3_5ForCausalLM'):
        raise ValueError('FlashAlign SGLang currently supports dense Qwen3.5-family model classes')
    modules=dict(runner.model.named_modules())
    positions={}
    for profile in profiles.values():
        for (layer,kind),operator in profile.sites.items():
            tensor=operator['tensor']
            if not isinstance(tensor,str) or not tensor.startswith('model.language_model.') or not tensor.endswith('.weight'):
                raise ValueError('FlashAlign profile tensor is not a supported Qwen target')
            module_name=tensor.replace('model.language_model.','model.',1)[:-7]
            if kind=='embedding_out':expected=['model.embed_tokens']
            elif kind=='mlp_out':expected=[f'model.layers.{layer}.mlp.down_proj']
            else:expected=[f'model.layers.{layer}.linear_attn.out_proj',f'model.layers.{layer}.self_attn.o_proj']
            if module_name not in expected:
                raise ValueError('FlashAlign profile tensor and site disagree')
            # This pinned SGLang full-attention layer owns o_proj directly;
            # the checkpoint retains the Hugging Face self_attn component.
            if kind=='attn_out' and module_name.endswith('.self_attn.o_proj') and module_name not in modules:
                module_name=f'model.layers.{layer}.o_proj'
            if module_name not in modules:
                raise ValueError('Missing FlashAlign target module: '+module_name)
            module=modules[module_name]
            width=getattr(module,'embedding_dim',getattr(module,'output_size',None))
            if width!=profile.hidden:
                raise ValueError('FlashAlign profile width does not match loaded module')
            operator['q']=operator['q'].to(runner.device)
            positions[module_name]=(layer,kind)
    handles=[]
    for name,site in positions.items():
        def hook(module,inputs,output,site=site):
            active=_active.get()
            if active is None:return output
            current,batch=active
            if current is not runner:return output
            first=output[0] if isinstance(output,tuple) else output
            modified=project(first,batch,site,profiles,gate)
            return (modified,*output[1:]) if isinstance(output,tuple) else modified
        handles.append(modules[name].register_forward_hook(hook))
    runner.flashalign_handles=handles


def owner_forward(function):
    @wraps(function)
    def wrapped(self,forward_batch,*args,**kwargs):
        if not getattr(self,'flashalign_handles',None):
            return function(self,forward_batch,*args,**kwargs)
        token=_active.set((self,forward_batch))
        try:return function(self,forward_batch,*args,**kwargs)
        finally:_active.reset(token)
    return wrapped
