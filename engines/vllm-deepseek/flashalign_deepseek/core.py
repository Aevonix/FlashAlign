# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Strict grouped profile loading and row-local runtime projection."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
import re
from pathlib import Path

# Reuse the already-qualified protocol implementation without mutating it.
from flashalign_vllm.core import FieldError, Selection, cache_key, load_key
from .deepseek_grouped import validate_grouped_profile

REQUIRES={'grouped_basis','combined_moe_output','shared_expert_output','engram_value_slice','vision_pre_bias','typed_image_span','selected_tp_rows'}
SITES={'attn_out','moe_out','shared_expert_out','embedding_out','vision_aligner_out','image_marker_out','engram_value_out'}
_spans=ContextVar('flashalign_deepseek_spans',default=())


def normalize_manifest(meta):
    """Map standard attention/shared profiles without changing their operator."""
    if meta.get('schema')=='flashalign.grouped_projection.v1':
        return meta
    if meta.get('schema') is not None or meta.get('transform','projection')!='projection' or meta.get('requires',[])!=[]:
        raise ValueError('Unsupported DeepSeek profile schema/transform')
    operators={};sites=[]
    for row in meta.get('sites',[]):
        layer,site=row.get('layer'),row.get('site')
        if type(layer) is not int or not 0<=layer<40 or site not in ('attn_out','shared_expert_out'):
            raise ValueError('Unsupported standard DeepSeek profile site')
        expected=f'layers.{layer}.'+('attn.wo_b.weight' if site=='attn_out' else 'ffn.shared_experts.w2.weight')
        if row.get('tensor')!=expected:
            raise ValueError('Standard profile tensor does not match its native DeepSeek site')
        key=f'L{layer}.{site}.Q'
        operators[key]={'basis':key,'rank':row.get('rank'),'scale':row.get('scale')}
        sites.append({**row,'operator':key})
    requires={'grouped_basis','selected_tp_rows'}
    if any(row['site']=='shared_expert_out' for row in sites):requires.add('shared_expert_output')
    return {**meta,'schema':'flashalign.grouped_projection.v1','transform':'projection',
            'requires':sorted(requires),'operators':operators,'sites':sites}


class Profile:
    def __init__(self,directory,allow_research=False):
        import torch
        from safetensors.torch import load_file
        root=Path(directory);raw=(root/'profile.json').read_bytes();payload=(root/'profile.safetensors').read_bytes()
        if len(raw)>4*1024**2 or len(payload)>16*1024**2:raise ValueError('Oversized grouped profile')
        meta=normalize_manifest(json.loads(raw))
        if meta.get('schema')!='flashalign.grouped_projection.v1' or meta.get('transform')!='projection':
            raise ValueError('Unsupported DeepSeek profile schema/transform')
        required=set(meta.get('requires',[]))
        if not required<=REQUIRES:raise ValueError('Unsupported required profile extension')
        if meta.get('status')=='research_candidate_not_deployment_qualified' and not allow_research:
            raise ValueError('Research profile requires explicit --allow-research')
        if meta.get('hidden_size')!=5120:raise ValueError('DeepSeek hidden width must be 5120')
        identity=meta.get('model',{})
        if identity!={'id':'deepseek-ai/DeepSeek-V4.1-Flash-1017','revision':'1bae0e1088993b9b82de08f59de5be5a34e42170'}:
            raise ValueError('Profile must identify pinned DeepSeek V4.1 Flash 1017')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',meta.get('name','')) or meta['name']=='off':
            raise ValueError('Invalid profile name')
        validate_grouped_profile(meta, meta['hidden_size'], meta.get('sites'))
        hashes=json.loads((root/'sha256.json').read_text())
        for name,body in [('profile.json',raw),('profile.safetensors',payload)]:
            if hashes.get(name)!=hashlib.sha256(body).hexdigest():raise ValueError('Profile checksum mismatch')
        values=load_file(root/'profile.safetensors',device='cpu');operators={};used=set()
        for name,operator in meta['operators'].items():
            rank,scale=operator.get('rank'),operator.get('scale');key=operator.get('basis')
            if type(rank) is not int or not 1<=rank<=128 or type(scale) not in [int,float] or not math.isfinite(scale):
                raise ValueError('Invalid grouped operator')
            q=values.get(key)
            if q is None or q.dtype!=torch.float32 or tuple(q.shape)!=(rank,5120):raise ValueError('Invalid basis tensor')
            if not torch.isfinite(q).all() or not torch.allclose(q@q.T,torch.eye(rank),atol=2e-5,rtol=0):
                raise ValueError('Non-orthonormal basis tensor')
            operators[name]=(q,float(scale));used.add(key)
        if used!=set(values):raise ValueError('Missing or unknown profile tensors')
        self.sites={}
        for row in meta['sites']:
            layer,site,operator=row.get('layer'),row.get('site'),row.get('operator')
            if type(layer) is not int or not 0<=layer<40 or site not in SITES or operator not in operators:
                raise ValueError('Unsupported DeepSeek site')
            if (layer,site) in self.sites:raise ValueError('Duplicate profile site')
            if site=='engram_value_out' and (layer not in [1,14] or row.get('output_slice')!=[20480,25600]):
                raise ValueError('Engram profile may target only the value slice')
            if site in ['embedding_out','vision_aligner_out','image_marker_out'] and layer!=0:raise ValueError('Invalid input site layer')
            self.sites[(layer,site)]=operator
        if not self.sites:raise ValueError('Empty profile')
        expected={'grouped_basis','selected_tp_rows'}
        extensions={'moe_out':{'combined_moe_output'},'shared_expert_out':{'shared_expert_output'},
            'engram_value_out':{'engram_value_slice'},'vision_aligner_out':{'vision_pre_bias','typed_image_span'},
            'image_marker_out':{'typed_image_span'}}
        for _,site in self.sites:expected.update(extensions.get(site,set()))
        if required!=expected:raise ValueError('Required capabilities do not match declared sites')
        self.name=meta['name'];self.meta=meta;self.operators=operators
        self.digest=hashlib.sha256(raw+payload).hexdigest();self.applied_rows={}
        self.off_rows_checked=0


def load_profiles(paths,allow_research=None):
    if allow_research is None:allow_research=os.environ.get('FLASHALIGN_DEEPSEEK_ALLOW_RESEARCH')=='1'
    result={}
    for path in paths:
        profile=Profile(path,allow_research)
        if profile.name in result:raise ValueError('Duplicate profile name')
        result[profile.name]=profile
    return result


@contextmanager
def row_context():
    token=_spans.set(())
    try:yield
    finally:_spans.reset(token)


def set_spans(spans):
    _spans.set(tuple(spans))


def get_spans():return _spans.get()


def wrap_target_embedding(method,profiles):
    """Wrap the target method, not its module also shared by the stock drafter."""
    from functools import wraps
    @wraps(method)
    def embed(*args,**kwargs):
        return project(method(*args,**kwargs),profiles,0,'embedding_out')
    return embed


@contextmanager
def scoped_spans(spans):
    token=_spans.set(tuple(spans))
    try:yield
    finally:_spans.reset(token)


def remap_spans(spans,indices):
    result=[]
    for output_row,input_row in enumerate(indices):
        selection=next((s for start,stop,s in spans if start<=input_row<stop),None)
        if selection is None:continue
        if result and result[-1][1]==output_row and result[-1][2]==selection:
            start,_,_=result[-1];result[-1]=(start,output_row+1,selection)
        else:result.append((output_row,output_row+1,selection))
    return result


def project(output,profiles,layer,site):
    """Only selected rows enter transform arithmetic; off returns the same object."""
    result=output
    for start,stop,selection in _spans.get():
        profile=profiles.get(selection.name)
        if profile is None or profile.digest!=selection.digest:raise ValueError('Worker profile identity differs')
        operator=profile.sites.get((layer,site))
        if operator is None or start==stop:continue
        if start<0 or stop>output.shape[0]:raise ValueError('Selected rows do not match model output')
        if result is output:result=output.clone()
        q,scale=profile.operators[operator]
        value=output[start:stop].float()
        changed=value-selection.scale*scale*((value@q.T)@q)
        result[start:stop].copy_(changed.to(output.dtype))
        label=f'{layer}.{site}';profile.applied_rows[label]=profile.applied_rows.get(label,0)+stop-start
    if result is not output and os.environ.get('FLASHALIGN_DEEPSEEK_CHECK_OFF_ROWS')=='1':
        import torch
        off=torch.ones(output.shape[0],device=output.device,dtype=torch.bool)
        for start,stop,_ in _spans.get():off[start:stop]=False
        count=int(off.sum())
        if count and not torch.equal(output[off].view(torch.uint8),result[off].view(torch.uint8)):
            raise RuntimeError('A transform changed off-mode row bytes')
        for name in {selection.name for _,_,selection in _spans.get()}:
            profiles[name].off_rows_checked+=count
    return result


def image_project(output,types,profile,selection,bias):
    """Cached span contains W2*gelu(W1*x+b1)+b2 and unchanged delimiters."""
    if selection is None:return output
    if profile.digest!=selection.digest:raise ValueError('Image profile identity differs')
    result=output.clone()
    if types.ndim!=1 or types.numel()!=output.shape[0] or not bool(((types>=0)&(types<=3)).all()):
        raise ValueError('Invalid typed image span')
    for site,mask in [('vision_aligner_out',types==1),('image_marker_out',types!=1)]:
        operator=profile.sites.get((0,site))
        if operator is None or not bool(mask.any()):continue
        q,scale=profile.operators[operator];value=output[mask].float()
        if site=='vision_aligner_out':value=value-bias.float()
        changed=value-selection.scale*scale*((value@q.T)@q)
        if site=='vision_aligner_out':changed=changed+bias.float()
        result[mask]=changed.to(output.dtype)
        label='0.'+site;profile.applied_rows[label]=profile.applied_rows.get(label,0)+int(mask.sum())
    return result


def spans_for_batch(batch,selections):
    starts=batch.query_start_loc_np
    if len(starts)<len(batch.req_ids)+1:raise ValueError('Incomplete V2 query boundaries')
    result=[]
    for i,request_id in enumerate(batch.req_ids):
        selection=selections.get(request_id)
        if selection is not None:result.append((int(starts[i]),int(starts[i+1]),selection))
    if result and max(stop for _,stop,_ in result)>batch.num_tokens:
        raise ValueError('Adaptive verification row mapping is not qualified')
    return result
