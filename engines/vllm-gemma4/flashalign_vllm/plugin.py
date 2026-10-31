# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Version-pinned hooks in vLLM's V1 GPU runner, installed in each worker."""
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from .core import load_profiles, project, selected_rows, spans_for_runner

REGISTERED=False
VERSION='0.32.0'
COMMIT='5df63a7203cdb9328d824a27b682e77c7b9928be'
PINNED_SOURCES={
    'v1/worker/gpu_model_runner.py':'4cf95efe4ea8caa3110dcd231ff95ceb97e8136f5056e6bdd388300a484c4a81',
    'model_executor/models/qwen3_5.py':'ea7e71303c902b9678e85ff3e9a75a7ab6e69b5770605d540ad6ab1084e18187',
    'model_executor/models/qwen3_next.py':'09ea2a385259edc9feadc79e304e86a31582b9770f80b8003d2d641782405818',
    'model_executor/models/gemma4.py':'e3922b9373db89b1aec00b06f4128448056f324e36c9ce7b81927bda04da3a33',
    'model_executor/models/gemma4_mm.py':'21f216a93c2386ee37b983d5da788c685e8465d9c18356c48c3bda912087bdd4',
}


def attach(runner, profiles):
    config=runner.vllm_config
    parallel=config.parallel_config
    if any(getattr(parallel,x,1)!=1 for x in ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size')):
        raise ValueError('FlashAlign vLLM currently supports one GPU and one engine replica')
    if getattr(parallel,'use_ubatching',False):
        raise ValueError('FlashAlign vLLM does not yet support microbatch overlap')
    if getattr(config,'lora_config',None) is not None:
        raise ValueError('FlashAlign vLLM does not yet qualify combined LoRA adapters')
    if not config.model_config.enforce_eager:
        raise ValueError('FlashAlign vLLM requires eager execution for selected-row projections')
    if getattr(config,'kv_transfer_config',None) is not None:
        raise ValueError('FlashAlign vLLM does not yet support external KV transfer')
    if getattr(config.model_config,'enable_prompt_embeds',False):
        raise ValueError('FlashAlign vLLM does not yet support external prompt embeddings')
    spec=config.speculative_config
    if spec is not None and spec.method not in ('ngram','ngram_gpu','eagle','eagle3','dflash','dspark'):
        raise ValueError('The active speculative method is unsupported and will not be disabled')
    model=runner.get_model()
    if type(model).__name__ not in ('Qwen3_5ForConditionalGeneration','Qwen3_5ForCausalLM',
                                    'Gemma4ForConditionalGeneration','Gemma4ForCausalLM'):
        raise ValueError('This experimental FlashAlign adapter supports Qwen3.5/3.8 dense and Gemma4')
    prefix='language_model.model' if hasattr(model,'language_model') else 'model'
    modules=dict(model.named_modules())
    hf=config.model_config.hf_text_config
    for profile in profiles.values():
        if profile.hidden!=hf.hidden_size:
            raise ValueError('Profile width differs from the loaded model')
        for (layer,site),(q,scale,tensor) in profile.sites.items():
            if layer>=hf.num_hidden_layers:
                raise ValueError('Profile layer is outside the loaded model')
            if site=='embedding_out':
                name=prefix+'.embed_tokens';expected='model.language_model.embed_tokens.weight'
            elif site=='mlp_out':
                name=f'{prefix}.layers.{layer}.mlp.down_proj'
                expected=f'model.language_model.layers.{layer}.mlp.down_proj.weight'
            else:
                part='linear_attn.out_proj' if hf.layer_types[layer]=='linear_attention' else 'self_attn.o_proj'
                name=f'{prefix}.layers.{layer}.{part}'
                expected=f'model.language_model.layers.{layer}.{part}.weight'
            if tensor!=expected or name not in modules:
                raise ValueError('Profile target differs from the loaded model module')
            # Store the immutable basis once on the inference device.
            profile.sites[(layer,site)]=(q.to(runner.device),scale,tensor)
    targets={}
    for profile in profiles.values():
        for layer,site in profile.sites:
            if site=='embedding_out':name=prefix+'.embed_tokens'
            elif site=='mlp_out':name=f'{prefix}.layers.{layer}.mlp.down_proj'
            else:
                part='linear_attn.out_proj' if hf.layer_types[layer]=='linear_attention' else 'self_attn.o_proj'
                name=f'{prefix}.layers.{layer}.{part}'
            targets[name]=(layer,site)
    for name,(layer,site) in targets.items():
        def hook(module,args,output,layer=layer,site=site):
            if isinstance(output,tuple):
                first=project(output[0],profiles,layer,site)
                return output if first is output[0] else (first,*output[1:])
            return project(output,profiles,layer,site)
        modules[name].register_forward_hook(hook)
    runner._flashalign_profiles=profiles


def register():
    global REGISTERED
    if REGISTERED:return
    if not os.environ.get('FLASHALIGN_PROFILES'):
        return
    if os.environ.get('FLASHALIGN_SECURE_LAUNCH')!='1':
        raise RuntimeError('Use the FlashAlign launcher to install the mandatory HTTP gate')
    if importlib.metadata.version('vllm')!=VERSION:
        raise RuntimeError('FlashAlign requires pinned vLLM '+VERSION)
    if os.environ.get('VLLM_USE_V2_MODEL_RUNNER','0').lower() not in ('0','false'):
        raise RuntimeError('FlashAlign vLLM supports the V1 GPU runner only')
    import vllm
    for relative,digest in PINNED_SOURCES.items():
        if hashlib.sha256((Path(vllm.__file__).parent/relative).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('FlashAlign does not recognize this modified vLLM source: '+relative)
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    original_ngram_init=NgramProposer.__init__
    @wraps(original_ngram_init)
    def ngram_init(self,*args,**kwargs):
        original_ngram_init(self,*args,**kwargs)
        # Upstream's empty-list warmup does not enter the jitted batch kernel.
        # Prepare a real synthetic batch before the server advertises readiness.
        import numpy as np
        print('[FlashAlign] Preparing ngram speculative kernel',flush=True)
        length=max(8,self.max_n+self.k+1)
        self.batch_propose(1,[0],np.array([length],dtype=np.int32),
            np.zeros((1,self.max_model_len),dtype=np.int32),self.k)
        self.valid_ngram_draft.fill(0)
        self.valid_ngram_num_drafts.fill(0)
    NgramProposer.__init__=ngram_init
    original_load=GPUModelRunner.load_model
    @wraps(original_load)
    def load(self,*args,**kwargs):
        result=original_load(self,*args,**kwargs)
        attach(self,load_profiles(json.loads(os.environ['FLASHALIGN_PROFILES'])))
        return result
    GPUModelRunner.load_model=load
    for name in ('_preprocess','_model_forward'):
        original=getattr(GPUModelRunner,name)
        def wrap(original):
            @wraps(original)
            def call(self,*args,**kwargs):
                profiles=getattr(self,'_flashalign_profiles',{})
                spans=spans_for_runner(self,profiles) if profiles else ()
                diagnostic=os.environ.get('FLASHALIGN_DIAGNOSTIC_FILE')
                before={name:dict(p.applied_rows) for name,p in profiles.items()} if diagnostic else {}
                with selected_rows(spans):
                    result=original(self,*args,**kwargs)
                if diagnostic:
                    delta={name:{k:v-before[name].get(k,0) for k,v in p.applied_rows.items()
                                 if v!=before[name].get(k,0)} for name,p in profiles.items()}
                    with open(diagnostic,'a') as stream:
                        stream.write(json.dumps({'stage':original.__name__,'selected_rows':sum(b-a for a,b,_ in spans),'applied_rows':delta})+'\n')
                return result
            return call
        setattr(GPUModelRunner,name,wrap(original))
    REGISTERED=True
