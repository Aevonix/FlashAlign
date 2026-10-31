# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Version-pinned DeepSeek V4.1 hooks for vLLM's V2 runner on TP4 SM120."""
from functools import wraps
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .core import (Selection,get_spans,image_project,load_profiles,project,
                   remap_spans,row_context,scoped_spans,set_spans,spans_for_batch,
                   wrap_target_embedding)

REGISTERED=False


def diagnostic(row):
    path=os.environ.get('FLASHALIGN_DEEPSEEK_DIAGNOSTIC')
    if path:
        with open(path,'a') as stream:stream.write(json.dumps(row,separators=(',',':'))+'\n')


def tensor_digest(tensor):
    import torch
    raw=tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def attach(runner,profiles):
    import torch
    config=runner.vllm_config;p=config.parallel_config
    if (p.tensor_parallel_size,p.pipeline_parallel_size,p.data_parallel_size)!=(4,1,1):
        raise ValueError('DeepSeek research adapter requires TP4, PP1, DP1')
    if p.enable_expert_parallel or p.use_ubatching or p.prefill_context_parallel_size!=1 or p.decode_context_parallel_size!=1:
        raise ValueError('Unqualified DeepSeek parallel execution mode')
    if not config.model_config.enforce_eager or config.lora_config is not None or config.kv_transfer_config is not None:
        raise ValueError('DeepSeek research adapter requires eager execution without LoRA/KV transfer')
    if torch.cuda.get_device_capability(runner.device)!=(12,0):raise ValueError('Only SM120 is qualified for this research adapter')
    if runner.adaptive_verification is not None:raise ValueError('Adaptive draft row compaction is not yet qualified')
    if config.speculative_config is None or config.speculative_config.method!='dspark':
        raise ValueError('Native DSpark must remain enabled for this research adapter')
    model=runner.model
    if type(model).__name__!='DeepseekV41ForCausalLM' or not hasattr(model,'language_model'):
        raise ValueError('Expected the native DeepSeek V4.1 model')
    base=model.language_model.model
    if base.use_sequence_parallel or base.fuse_mhc_all_reduce:
        raise ValueError('This adapter has not qualified sequence parallelism or deferred MoE finalization')
    for profile in profiles.values():
        for name,(q,scale) in profile.operators.items():profile.operators[name]=(q.to(runner.device),scale)
    runner._flashalign_profiles=profiles
    runner._flashalign_selections={}
    runner._flashalign_mm_types={}
    def hook(layer,site):
        def apply(module,args,out):
            if isinstance(out,tuple):
                changed=project(out[0],profiles,layer,site)
                return out if changed is out[0] else (changed,*out[1:])
            if not isinstance(out,torch.Tensor):
                if get_spans():raise ValueError('Unqualified non-tensor output at profile site')
                return out
            return project(out,profiles,layer,site)
        return apply
    # DSpark aliases the exact embedding module. Wrap only the target's
    # method so draft rows never inherit the target's batch row map.
    base.embed_input_ids=wrap_target_embedding(base.embed_input_ids,profiles)
    for index,layer in enumerate(base.layers):
        layer.attn.wo_b.register_forward_hook(hook(index,'attn_out'))
        layer.ffn.register_forward_hook(hook(index,'moe_out'))
        if any((index,'shared_expert_out') in p.sites for p in profiles.values()):
            if layer.ffn.shared_experts is None:raise ValueError('Declared shared-expert site is absent')
            layer.ffn.shared_experts.down_proj.register_forward_hook(hook(index,'shared_expert_out'))
        if layer.engram is not None:
            def engram_hook(module,args,out,index=index):
                if out.ndim!=2 or out.shape[1]!=25600:raise ValueError('Unexpected Engram WKV output shape')
                shifted=project(out[:,20480:25600],profiles,index,'engram_value_out')
                if not get_spans():return out
                result=out.clone();result[:,20480:25600]=shifted;return result
            layer.engram.wkv.register_forward_hook(engram_hook)
    replay=base.decoder_replay_layers
    if replay is not None:replay._flashalign_enabled=True
    encoder=runner.model_state.encoder_runner
    encoder._flashalign_runner=runner
    encoder.encoder_cache._flashalign_runner=runner
    if tuple(model.aligner.w2.bias.shape)!=(5120,):raise ValueError('Unexpected vision aligner bias')
    diagnostic({'event':'attached','profile_sites':{n:len(p.sites) for n,p in profiles.items()},'runner':'V2','tensor_parallel':4,'speculation':'dspark'})


def register():
    global REGISTERED
    if os.environ.get('FLASHALIGN_DEEPSEEK_SM120_GEOMETRY')=='64':
        from .sm120 import apply_geometry
        apply_geometry()
    if REGISTERED or not os.environ.get('FLASHALIGN_DEEPSEEK_PROFILES'):return
    if os.environ.get('FLASHALIGN_DEEPSEEK_SECURE_LAUNCH')!='1':raise RuntimeError('Use the secured DeepSeek launcher')
    if importlib.metadata.version('vllm')!='0.32.0':raise RuntimeError('Pinned vLLM 0.32.0 is required')
    if os.environ.get('VLLM_USE_V2_MODEL_RUNNER')!='1':raise RuntimeError('The native V2 runner is required')
    import vllm
    pins=json.loads((Path(__file__).parent/'source-pins.json').read_text())
    for name,digest in pins.items():
        if hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Unrecognized modified vLLM source: '+name)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
    from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers
    from vllm.multimodal.utils import get_mm_features_in_window
    original_load=GPUModelRunner.load_model
    @wraps(original_load)
    def load(self,*args,**kwargs):
        result=original_load(self,*args,**kwargs)
        attach(self,load_profiles(json.loads(os.environ['FLASHALIGN_DEEPSEEK_PROFILES'])))
        return result
    GPUModelRunner.load_model=load
    original_initialize=GPUModelRunner.initialize_kv_cache
    @wraps(original_initialize)
    def initialize(self,*args,**kwargs):
        result=original_initialize(self,*args,**kwargs)
        if self.adaptive_verification is not None:
            raise ValueError('Adaptive draft row compaction is not yet qualified')
        return result
    GPUModelRunner.initialize_kv_cache=initialize
    original_remove=GPUModelRunner._remove_request
    @wraps(original_remove)
    def remove(self,request_id):
        getattr(self,'_flashalign_selections',{}).pop(request_id,None)
        return original_remove(self,request_id)
    GPUModelRunner._remove_request=remove
    original_add=GPUModelRunner.add_requests
    @wraps(original_add)
    def add(self,scheduler_output):
        result=original_add(self,scheduler_output)
        for request in scheduler_output.scheduled_new_reqs:
            extra=getattr(request.sampling_params,'extra_args',None) or {}
            name=extra.get('_flashalign_profile')
            if name is not None:
                profile=self._flashalign_profiles.get(name);scale=extra.get('_flashalign_scale')
                if profile is None or profile.digest!=extra.get('_flashalign_digest'):
                    raise ValueError('Worker profile identity mismatch')
                if type(scale) not in [int,float] or not math.isfinite(scale) or not 0<scale<=2:
                    raise ValueError('Invalid worker scale')
                self._flashalign_selections[request.req_id]=Selection(name,profile.digest,float(scale))
            for feature in request.mm_features:
                if feature.data is not None and 'types' in feature.data:
                    self._flashalign_mm_types[feature.identifier]=feature.data['types'].data.detach().clone().flatten()
        return result
    GPUModelRunner.add_requests=add
    original_prepare=GPUModelRunner.prepare_inputs
    @wraps(original_prepare)
    def prepare(self,*args,**kwargs):
        batch=original_prepare(self,*args,**kwargs)
        set_spans(spans_for_batch(batch,self._flashalign_selections))
        return batch
    GPUModelRunner.prepare_inputs=prepare
    original_execute=GPUModelRunner.execute_model
    @wraps(original_execute)
    def execute(self,*args,**kwargs):
        before={name:dict(p.applied_rows) for name,p in self._flashalign_profiles.items()}
        checked={name:p.off_rows_checked for name,p in self._flashalign_profiles.items()}
        with row_context():result=original_execute(self,*args,**kwargs)
        diagnostic({'event':'target_forward','applied_rows':{name:{k:v-before[name].get(k,0) for k,v in p.applied_rows.items()
            if v!=before[name].get(k,0)} for name,p in self._flashalign_profiles.items()},
            'off_rows_checked':{name:p.off_rows_checked-checked[name] for name,p in self._flashalign_profiles.items()}})
        return result
    GPUModelRunner.execute_model=execute
    original_replay=DecoderReplayLayers._run
    @wraps(original_replay)
    def replay(self,*args,**kwargs):
        if not getattr(self,'_flashalign_enabled',False) or not get_spans() or self.replay_batch is None:
            return original_replay(self,*args,**kwargs)
        if self.replay_batch.run_graph is not None:raise ValueError('Captured replay cannot bypass selected-row projections')
        indices=self.replay_batch.rows.detach().cpu().tolist()
        with scoped_spans(remap_spans(get_spans(),indices)):return original_replay(self,*args,**kwargs)
    DecoderReplayLayers._run=replay
    original_gather=EncoderRunner.gather_mm_embeddings
    @wraps(original_gather)
    def gather(self,req_ids,total,num_scheduled,query_starts,prefill_lens,num_computed,draft_lookahead=0):
        outputs,mask=original_gather(self,req_ids,total,num_scheduled,query_starts,prefill_lens,num_computed,draft_lookahead)
        runner=getattr(self,'_flashalign_runner',None)
        # Draft queries are a different layout; keep the stock drafter's
        # image features stock, just like its shared token embeddings.
        if runner is None or draft_lookahead:return outputs,mask
        recording=bool(os.environ.get('FLASHALIGN_DEEPSEEK_DIAGNOSTIC'))
        if not recording and not any(r in runner._flashalign_selections for r in req_ids):return outputs,mask
        descriptions=[]
        computed=num_computed+draft_lookahead
        for i,request_id in enumerate(req_ids):
            if computed[i]>=prefill_lens[i]:continue
            start,end=int(computed[i]),int(computed[i]+num_scheduled[i])
            features=self.encoder_cache.mm_features[request_id]
            lo,hi=get_mm_features_in_window(features,start=start,end=end)
            for feature in features[lo:hi]:
                pos=feature.mm_position;a=max(start-pos.offset,0);b=min(end-pos.offset,pos.length)
                first,last=pos.get_embeds_indices_in_range(a,b)
                if first==last:continue
                cached=self.encoder_cache.encoder_outputs.get(feature.identifier)
                if cached is None:
                    if pos.offset+draft_lookahead>=end:continue
                    raise ValueError('Encoder cache miss during owner projection')
                if pos.is_embed is None:first,last=a,b
                descriptions.append((runner._flashalign_selections.get(request_id),feature.identifier,first,last,cached))
        if len(descriptions)!=len(outputs):raise ValueError('Image batch ordering differs from the pinned path')
        result=[]
        for output,(selection,identifier,start,stop,cached) in zip(outputs,descriptions):
            before=tensor_digest(cached) if recording else None
            if selection is None:changed=output
            else:
                types=runner._flashalign_mm_types.get(identifier)
                if types is None or types.numel()!=cached.shape[0]:raise ValueError('Missing cached image token types')
                profile=runner._flashalign_profiles[selection.name]
                if recording:image_counts=dict(profile.applied_rows)
                changed=image_project(output,types[start:stop].to(output.device),profile,selection,runner.model.aligner.w2.bias)
            result.append(changed)
            if recording:
                after=tensor_digest(cached)
                if before!=after:raise RuntimeError('Owner image projection modified encoder cache storage')
                diagnostic({'event':'image_gather','profile':'off' if selection is None else selection.name,
                    'scale':None if selection is None else selection.scale,'rows':stop-start,
                    'applied_rows':{} if selection is None else {name:value-image_counts.get(name,0)
                        for name,value in profile.applied_rows.items() if value!=image_counts.get(name,0)},
                    'cache_key_hash':hashlib.sha256(identifier.encode()).hexdigest(),
                    'cache_sha256_before':before,'cache_sha256_after':after,
                    'input_view_sha256':tensor_digest(output),
                    'output_view_sha256':tensor_digest(changed),'cache_unchanged':before==after})
        return result,mask
    EncoderRunner.gather_mm_embeddings=gather
    original_prepare_mm=EncoderRunner.prepare_mm_inputs
    @wraps(original_prepare_mm)
    def prepare_mm(self,scheduled_encoder_inputs):
        result=original_prepare_mm(self,scheduled_encoder_inputs)
        if getattr(self,'_flashalign_runner',None) is not None:
            diagnostic({'event':'image_encode_inputs','cache_key_hashes':[
                hashlib.sha256(identifier.encode()).hexdigest() for identifier in result[0]],
                'encoder_calls_required':len(result[0])})
        return result
    EncoderRunner.prepare_mm_inputs=prepare_mm
    original_free=EncoderCache.free_encoder_cache
    @wraps(original_free)
    def free(self,identifier):
        runner=getattr(self,'_flashalign_runner',None)
        if runner is not None:runner._flashalign_mm_types.pop(identifier,None)
        return original_free(self,identifier)
    EncoderCache.free_encoder_cache=free
    original_reset=EncoderCache.reset_encoder_cache
    @wraps(original_reset)
    def reset(self):
        runner=getattr(self,'_flashalign_runner',None)
        if runner is not None:runner._flashalign_mm_types.clear()
        return original_reset(self)
    EncoderCache.reset_encoder_cache=reset
    REGISTERED=True
