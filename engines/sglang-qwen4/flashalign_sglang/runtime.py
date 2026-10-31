# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Qwen4Exp target/native-MTP hooks for pinned SGLang eager execution."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
import hashlib
import json
import os
from pathlib import Path
import re
from types import MethodType

import torch
from safetensors.torch import load_file

from .gate import Gate, PREFIX, read_key
from .flashalign_replay import deployment_replay_store

_active = ContextVar('flashalign_qwen4_forward',default=None)
_audit_seen = set()
REQUIRES = {'writer_groups','text_embedding_before_multimodal_merge','ple_value_projection','native_mtp_writer_groups'}


@dataclass
class Profile:
    name: str
    digest: str
    model: dict
    hidden: int
    sites: dict
    compatible_models: list = field(default_factory=list)


def expected_sources(role,layer,kind):
    if kind=='text_embedding_out' and role=='target_and_native_mtp' and layer==0:
        return ['model.language_model.embed_tokens.weight']
    if kind=='ple_value_out' and role=='target' and layer==1:
        return ['model.language_model.layers.1.ple.value_proj.weight']
    if role not in ('target','native_mtp') or not 0<=layer<(48 if role=='target' else 1):
        raise ValueError('Unsupported FlashAlign writer role/layer')
    prefix=f'model.language_model.layers.{layer}' if role=='target' else f'mtp.layers.{layer}'
    if kind=='attn_out':
        attention='self_attn.o_proj' if role=='native_mtp' or (layer+1)%4==0 else 'linear_attn.out_proj'
        return [prefix+'.'+attention+'.weight']
    if kind=='combined_mlp_out':return [prefix+'.mlp.experts.down_proj',prefix+'.mlp.shared_expert.down_proj.weight']
    raise ValueError('Unsupported FlashAlign writer site')


def load_profile(directory):
    path=Path(directory);raw=(path/'profile.json').read_bytes();meta=json.loads(raw)
    if type(meta.get('schema_version')) is not int or meta['schema_version']!=1:raise ValueError('Unsupported profile schema version')
    if len(raw)>4*1024*1024 or meta.get('transform')!='residual_writer_group_projection_v1' or set(meta.get('requires',[]))!=REQUIRES:
        raise ValueError('Unsupported FlashAlign Qwen4 profile contract')
    name=meta.get('name','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',name) or name=='off':raise ValueError('Invalid profile name')
    if type(meta.get('hidden_size')) is not int or type(meta.get('rank')) is not int or type(meta.get('scale')) not in (int,float):raise ValueError('Invalid Qwen4 profile geometry type')
    if meta.get('hidden_size')!=2560 or meta.get('rank')!=1 or meta.get('scale')!=1:raise ValueError('Unsupported Qwen4 profile geometry')
    binary=path/'profile.safetensors'
    if binary.stat().st_size>1024*1024:raise ValueError('Oversized profile operator')
    tensors=load_file(str(binary),device='cpu');q=tensors.get('shared.Q')
    if set(tensors)!={'shared.Q'} or q is None or q.dtype!=torch.float32 or tuple(q.shape)!=(1,2560) or not torch.isfinite(q).all():raise ValueError('Invalid Qwen4 basis')
    if not torch.allclose(q@q.T,torch.eye(1),atol=2e-5,rtol=0):raise ValueError('Basis must be orthonormal')
    sites={}
    for site in meta.get('sites',[]):
        role,layer,kind=site.get('model_role'),site.get('layer'),site.get('site')
        if type(layer) is not int:raise ValueError('Invalid site layer')
        key=(role,layer,kind)
        if key in sites or site.get('source_tensors')!=expected_sources(*key):raise ValueError('Duplicate or mismatched writer source')
        if kind=='combined_mlp_out' and (site.get('expert_count')!=512 or site.get('fused_expert_tensor_axes')!=['expert','output','input']):raise ValueError('Invalid fused expert declaration')
        sites[key]={'q':q,'scale':1.0,'sources':site['source_tensors']}
    expected={('target_and_native_mtp',0,'text_embedding_out'),('target',1,'ple_value_out')}
    expected.update((role,layer,kind) for role,count in (('target',48),('native_mtp',1)) for layer in range(count) for kind in ('attn_out','combined_mlp_out'))
    if set(sites)!=expected:raise ValueError('Incomplete Qwen4/native-MTP writer coverage')
    compatible=meta.get('compatible_models')
    if not isinstance(compatible,list) or not compatible:raise ValueError('Profile needs explicit model compatibility')
    for item in compatible:
        if not isinstance(item,dict) or item.get('model_type')!='qwen4_exp':raise ValueError('Unsupported compatible model type')
        if not isinstance(item.get('id'),str) or not re.fullmatch(r'[0-9a-f]{40}',item.get('revision','')):raise ValueError('Compatible model must be pinned')
        if any(not re.fullmatch(r'[0-9a-f]{64}',item.get(key,'')) for key in ('config_sha256','index_sha256','quantization_config_sha256')):raise ValueError('Compatible model fingerprints are required')
    return Profile(name,hashlib.sha256(raw+binary.read_bytes()).hexdigest(),meta['model'],2560,sites,compatible)


@lru_cache(maxsize=1)
def registry():
    profiles={}
    for root in os.environ.get('FLASHALIGN_PROFILES','').split(','):
        if root.strip():
            p=load_profile(root.strip())
            if p.name in profiles:raise ValueError('Duplicate profile name')
            profiles[p.name]=p
    key=read_key(os.environ.get('FLASHALIGN_KEY_FILE','')) if profiles else None
    scope=os.environ.get('FLASHALIGN_MODEL_SCOPE','')
    if profiles and not scope:raise ValueError('FlashAlign Qwen4 requires a pinned model scope')
    for profile in profiles.values():
        accepted=[]
        for item in profile.compatible_models:
            model={'id':item['id'],'revision':item['revision'],'sha256':{'config.json':item['config_sha256'],'model.safetensors.index.json':item['index_sha256']}}
            accepted.append(hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest())
        if scope not in accepted:raise ValueError('Profile does not match the running model fingerprints')
    return profiles,Gate(profiles,key,model_scope=scope,
                         replay_store=deployment_replay_store(key) if profiles else None)


def check_server_args(args):
    profiles,_=registry()
    if not profiles:return
    for field in ('pp_size','dp_size','dcp_size','attn_cp_size','nnodes','tokenizer_worker_num'):
        if getattr(args,field,1)!=1:raise ValueError('FlashAlign Qwen4 requires one worker and PP/DP/CP=1')
    tp=getattr(args,'tp_size',1);ep=getattr(args,'ep_size',1)
    if tp!=2 or ep!=2:raise ValueError('FlashAlign Qwen4 currently qualifies TP=EP=2')
    if not getattr(args,'disable_cuda_graph',False):
        decode=getattr(args,'disable_decode_cuda_graph',False) or getattr(args,'cuda_graph_backend_decode',None)=='disabled'
        prefill=getattr(args,'disable_prefill_cuda_graph',False) or getattr(args,'cuda_graph_backend_prefill',None)=='disabled'
        if not (decode and prefill):raise ValueError('FlashAlign Qwen4 requires eager execution while preserving active speculation')
    algorithm=getattr(args,'speculative_algorithm',None)
    if algorithm not in (None,'EAGLE','NEXTN'):raise ValueError('FlashAlign Qwen4 supports native MTP only; active speculation will not be disabled or replaced')
    if algorithm is not None and getattr(args,'speculative_eagle_topk',1)!=1:raise ValueError('FlashAlign Qwen4 requires native MTP top-k one')
    if os.environ.get('SGLANG_RAGGED_VERIFY_MODE','static')!='static':raise ValueError('FlashAlign Qwen4 requires the qualified static verification layout')
    if os.environ.get('SGLANG_RUST_SERVER','').lower() in ('1','true','on'):raise ValueError('FlashAlign Qwen4 requires its authenticated Python HTTP boundary')
    for field in ('enable_torch_compile','enable_lora','enable_custom_logit_processor','log_requests','enable_hierarchical_cache'):
        if getattr(args,field,False):raise ValueError('Unsupported FlashAlign Qwen4 option: '+field)
    if getattr(args,'disaggregation_mode','null') not in (None,'null'):raise ValueError('FlashAlign Qwen4 requires local cache execution')
    draft=getattr(args,'speculative_draft_model_path',None)
    if draft is not None and draft!=getattr(args,'model_path',draft):raise ValueError('FlashAlign Qwen4 requires the model native MTP weights')
    os.environ['FLASHALIGN_HTTP_GATE']='1'


def row_ranges(batch,rows,one_per_request=False):
    salts=getattr(batch,'flashalign_cache_salts',None)
    if not salts:return []
    count=len(salts);mode=batch.forward_mode
    if one_per_request:
        if rows!=count:raise ValueError('Embedding refresh does not match request count')
        lengths=[1]*count
    elif mode.is_target_verify() or getattr(mode,'is_draft_extend_v2',lambda:False)():
        width=getattr(batch.spec_info,'num_tokens_per_req',None)
        if width is None:width=getattr(batch.spec_info,'draft_token_num',None)
        if type(width) is not int or width<=0:raise ValueError('Unknown native MTP verification/extension width')
        lengths=[width]*count
    elif mode.is_decode():
        width=getattr(getattr(batch,'spec_info',None),'num_tokens_per_req',1)
        if width!=1:raise ValueError('Native MTP branching is not supported')
        lengths=[1]*count
    elif mode.is_extend():
        lengths=getattr(batch,'extend_seq_lens_cpu',None)
        if lengths is None:raise ValueError('Missing per-request extension lengths')
        lengths=list(lengths)
    elif mode.is_idle():return []
    else:raise ValueError('Unsupported Qwen4 forward mode')
    if len(lengths)!=count or any(type(x) is not int or x<0 for x in lengths) or sum(lengths)>rows:raise ValueError('Forward row count does not match selected requests')
    # TP may append padding rows. They remain stock; real token rows lead the batch.
    result=[];start=0
    for salt,length in zip(salts,lengths):result.append((salt,start,length));start+=length
    return result


def project(output,batch,site,profiles,gate,one_per_request=False):
    salts=getattr(batch,'flashalign_cache_salts',None)
    if not salts or all(not isinstance(s,str) or not s.startswith(PREFIX) or s.startswith(PREFIX+'off.') for s in salts):return output
    selected=[]
    ranges=row_ranges(batch,output.shape[0],one_per_request)
    for salt,start,length in ranges:
        selection=gate.from_salt(salt)
        if selection is None or length==0:continue
        operator=profiles[selection.name].sites.get(site)
        if operator is not None:selected.append((selection,start,length,operator))
    if not selected:return output
    result=output.clone()
    for selection,start,length,operator in selected:
        value=output[start:start+length].float();q=operator['q']
        modified=value-selection.scale*operator['scale']*((value@q.T)@q)
        result[start:start+length]=modified.to(output.dtype)
        audit=os.environ.get('FLASHALIGN_AUDIT_DIR')
        if audit:
            active=_active.get();runner_role='native_mtp' if active and active[0].is_draft_worker else 'target'
            phase=getattr(batch.forward_mode,'name',str(batch.forward_mode))
            mixed=len(selected)<len(ranges)
            key=(os.getpid(),runner_role,site,selection.scale,phase,mixed)
            if key not in _audit_seen:
                _audit_seen.add(key)
                off_ranges=[(a,n) for salt,a,n in ranges if gate.from_salt(salt) is None]
                unchanged=all(torch.equal(output[a:a+n].contiguous().view(torch.uint8),result[a:a+n].contiguous().view(torch.uint8)) for a,n in off_ranges)
                record={'pid':os.getpid(),'runner_role':runner_role,'site':list(site),'scale':selection.scale,
                        'phase':phase,'selected_rows':length,'batch_rows':output.shape[0],
                        'batch_requests':len(ranges),'mixed_owner_and_stock':mixed,
                        'stock_rows_unchanged':unchanged,'activation_dtype':str(output.dtype),
                        'before_component_l2':float(torch.linalg.vector_norm(value@q.T)),
                        'after_component_fp32_l2':float(torch.linalg.vector_norm(modified@q.T)),
                        'after_component_actual_dtype_l2':float(torch.linalg.vector_norm(result[start:start+length].float()@q.T))}
                directory=Path(audit);directory.mkdir(mode=0o700,parents=True,exist_ok=True)
                name=hashlib.sha256(repr(key).encode()).hexdigest()+'.json'
                fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'w') as handle:json.dump(record,handle)
    return result


def install(runner):
    profiles,gate=registry()
    if not profiles:return
    if os.environ.get('FLASHALIGN_HTTP_GATE')!='1':raise ValueError('Qwen4 profiles require the authenticated HTTP launch path')
    role='native_mtp' if runner.is_draft_worker else 'target'
    expected='Qwen4ExpForCausalLMMTP' if runner.is_draft_worker else 'Qwen4ExpForConditionalGeneration'
    if type(runner.model).__name__!=expected:raise ValueError('FlashAlign requires the Qwen4 target/native-MTP architecture')
    modules=dict(runner.model.named_modules());bindings={}
    for profile in profiles.values():
        for site,operator in profile.sites.items():
            site_role,layer,kind=site
            if site_role not in (role,'target_and_native_mtp'):continue
            if kind=='text_embedding_out':name='model.embed_tokens'
            elif kind=='ple_value_out':name=f'model.layers.{layer}.ple.value_proj'
            elif kind=='combined_mlp_out':name=f'model.layers.{layer}'
            else:name=f'model.layers.{layer}.o_proj' if role=='native_mtp' or (layer+1)%4==0 else f'model.layers.{layer}.linear_attn.out_proj'
            if name not in modules:raise ValueError('Missing Qwen4 writer: '+name)
            if kind!='combined_mlp_out':
                width=getattr(modules[name],'embedding_dim',getattr(modules[name],'output_size',None))
                if width!=profile.hidden:raise ValueError('Qwen4 writer width mismatch')
            operator['q']=operator['q'].to(runner.device)
            if name in bindings and bindings[name]!=site:raise ValueError('Conflicting writer binding')
            bindings[name]=site
    handles=[]
    for name,site in bindings.items():
        kind=site[2];module=modules[name]
        if kind=='combined_mlp_out':
            original=module._run_qwen4_exp_mlp
            def wrapped_mlp(self,hidden,batch,original=original,site=site):
                output=original(hidden,batch);active=_active.get()
                return project(output,batch,site,profiles,gate) if active is not None and active[0] is runner else output
            module._run_qwen4_exp_mlp=MethodType(wrapped_mlp,module)
            handles.append((module,original))
        else:
            def hook(module,inputs,output,site=site,kind=kind):
                active=_active.get()
                if active is None or active[0] is not runner:return output
                batch=active[1];first=output[0] if isinstance(output,tuple) else output
                single=kind=='text_embedding_out' and first.shape[0]==len(getattr(batch,'flashalign_cache_salts',[]) or [])
                modified=project(first,batch,site,profiles,gate,one_per_request=single)
                return (modified,*output[1:]) if isinstance(output,tuple) else modified
            handles.append(module.register_forward_hook(hook))
    runner.flashalign_handles=handles


def owner_forward(function):
    @wraps(function)
    def wrapped(self,forward_batch,*args,**kwargs):
        if not getattr(self,'flashalign_handles',None):return function(self,forward_batch,*args,**kwargs)
        token=_active.set((self,forward_batch))
        try:return function(self,forward_batch,*args,**kwargs)
        finally:_active.reset(token)
    return wrapped
