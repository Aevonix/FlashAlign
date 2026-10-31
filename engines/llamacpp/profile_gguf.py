# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Convert a standard output-projection profile into a native adapter container.

This copies runtime factors only. It does not fit a profile or read model payloads.
The matching FlashAlign llama.cpp patch is required by adapter.type.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


def read_profile(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate metadata field')
            result[key]=value
        return result
    root=Path(path)
    raw=(root/'profile.json').read_bytes()
    if len(raw)>4*1024*1024: raise ValueError('Profile metadata too large')
    meta=json.loads(raw,object_pairs_hook=unique)
    if meta.get('transform','projection')!='projection' or meta.get('requires',[])!=[]:
        raise ValueError('Only standard output projection is supported')
    model=meta.get('model')
    if not isinstance(model,dict) or any(not isinstance(model.get(k),str) or not model[k] for k in ['id','revision']):
        raise ValueError('Profile must pin its model and revision')
    name=meta.get('name','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',name): raise ValueError('Invalid profile name')
    hidden=meta.get('hidden_size')
    if type(hidden) is not int or not 1<=hidden<=16384: raise ValueError('Invalid hidden width')
    data=(root/'profile.safetensors').read_bytes()
    if not 10<=len(data)<=256*1024*1024: raise ValueError('Invalid profile size')
    hashes=root/'sha256.json'
    if hashes.exists():
        expected=json.loads(hashes.read_text())
        for filename,content in [('profile.json',raw),('profile.safetensors',data)]:
            if expected.get(filename)!=hashlib.sha256(content).hexdigest():
                raise ValueError('Profile checksum mismatch')
    size=int.from_bytes(data[:8],'little')
    if not 2<=size<=4*1024*1024 or 8+size>len(data): raise ValueError('Invalid tensor header')
    tensors=json.loads(data[8:8+size],object_pairs_hook=unique)
    items=[];used=set();ranges=[];targets=set()
    sites=meta.get('sites')
    if not isinstance(sites,list) or not 1<=len(sites)<=4096: raise ValueError('Invalid sites')
    for site in sites:
        layer=site.get('layer');kind=site.get('site');rank=site.get('rank')
        if type(layer) is not int or not 0<=layer<1024: raise ValueError('Invalid layer')
        if type(rank) is not int or not 1<=rank<=min(128,hidden): raise ValueError('Invalid rank')
        scale=site.get('scale')
        if (type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0
                or scale>float(np.finfo(np.float32).max)/2 or float(np.float32(scale))<=0):
            raise ValueError('Site scales must be finite, positive FP32 values safe at request strength 2')
        scale=float(np.float32(scale))
        tensor=site.get('tensor')
        if kind=='embedding_out' and layer==0 and tensor=='model.language_model.embed_tokens.weight':
            target='token_embd.weight'
        elif kind=='mlp_out' and tensor==f'model.language_model.layers.{layer}.mlp.down_proj.weight':
            target=f'blk.{layer}.ffn_down.weight'
        elif kind=='attn_out' and tensor==f'model.language_model.layers.{layer}.self_attn.o_proj.weight':
            target=f'blk.{layer}.attn_output.weight'
        elif kind=='attn_out' and tensor==f'model.language_model.layers.{layer}.linear_attn.out_proj.weight':
            target=f'blk.{layer}.ssm_out.weight'
        else: raise ValueError('Unsupported output projection site')
        if target in targets: raise ValueError('Duplicate target')
        targets.add(target)
        key=f'L{layer}.{kind}.Q';spec=tensors[key]
        if spec.get('dtype')!='F32' or spec.get('shape')!=[rank,hidden] or any(type(n) is not int for n in spec.get('shape',[])): raise ValueError('Invalid Q tensor')
        start,end=spec['data_offsets']
        if type(start) is not int or type(end) is not int or start<0 or end-start!=rank*hidden*4 or 8+size+end>len(data): raise ValueError('Invalid tensor bounds')
        ranges.append((start,end));used.add(key)
        q=np.frombuffer(data,offset=8+size+start,count=rank*hidden,dtype='<f4').reshape(rank,hidden).copy()
        if not np.isfinite(q).all() or np.max(np.abs(q@q.T-np.eye(rank)))>2e-5:
            raise ValueError('Invalid or nonorthonormal basis')
        items.append((target,q,scale))
    if used!=set(tensors)-{'__metadata__'}: raise ValueError('Unexpected tensors')
    ranges.sort()
    if ranges[0][0]!=0 or ranges[-1][1]!=len(data)-8-size or any(a[1]!=b[0] for a,b in zip(ranges,ranges[1:])):
        raise ValueError('Overlapping or noncontiguous tensors')
    return meta,items,hashlib.sha256(raw+data).hexdigest()


def convert(profile,output,gguf_python,model=None,architecture=None):
    sys.path.insert(0,str(gguf_python))
    import gguf
    meta,items,digest=read_profile(profile)
    if Path(output).exists(): raise ValueError('Output already exists')
    if model is not None:
        source=gguf.GGUFReader(model,'r')
        source_architecture=source.get_field('general.architecture').contents()
        if architecture is not None and architecture!=source_architecture:
            raise ValueError('Explicit architecture differs from the model')
        architecture=source_architecture
        if architecture not in ('qwen35','gemma4'):
            raise ValueError('This package supports qwen35 and gemma4 GGUF architectures only')
        tensors={t.name:t for t in source.tensors}
        for target,q,scale in items:
            tensor=tensors.get(target)
            if tensor is None: raise ValueError('Model lacks '+target)
            width=int(tensor.shape[0] if target=='token_embd.weight' else tensor.shape[1])
            if len(tensor.shape)!=2 or width!=q.shape[1]: raise ValueError('Model shape mismatch at '+target)
    architecture=architecture or 'qwen35'
    if architecture not in ('qwen35','gemma4'):
        raise ValueError('This package supports qwen35 and gemma4 GGUF architectures only')
    if architecture=='gemma4' and any(target=='token_embd.weight' or '.ssm_out.' in target for target,_,_ in items):
        raise ValueError('Gemma profiles support attention and dense MLP outputs only')
    writer=gguf.GGUFWriter(output,architecture)
    writer.add_type('adapter')
    writer.add_name(meta['name'])
    writer.add_string('adapter.type','flashalign_projection')
    writer.add_float32('adapter.lora.alpha',0.)
    writer.add_uint32('flashalign.projection.version',2)
    writer.add_string('flashalign.profile.digest',digest)
    writer.add_string('flashalign.profile.name',meta['name'])
    for target,q,scale in items:
        writer.add_float32('flashalign.projection.scale.'+target,scale)
        writer.add_tensor(target+'.lora_a',np.ascontiguousarray(q,dtype=np.float32))
        writer.add_tensor(target+'.lora_b',np.ascontiguousarray(-q.T,dtype=np.float32))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    return {'name':meta['name'],'model':meta['model'],'sites':len(items),'profile_digest':digest,
            'adapter_sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest(),
            'adapter_type':'flashalign_projection','architecture':architecture,
            'site_scales':{target:scale for target,_,scale in items},'fitted_from_weights':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gguf-python',type=Path,required=True);p.add_argument('--model',type=Path)
    p.add_argument('--architecture',choices=['qwen35','gemma4'])
    a=p.parse_args();r=convert(a.profile,a.output,a.gguf_python,a.model,a.architecture)
    print(json.dumps(r,indent=2))
