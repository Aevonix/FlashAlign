# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import asyncio
import hashlib
import hmac
import json
import math
import os
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from flashalign_vllm.core import (FieldError,Gate,Profile,Selection,cache_key,load_key,
    project,selected_rows,spans_for_runner)
from flashalign_vllm.middleware import OwnerMiddleware
from flashalign_vllm.plugin import attach,register


def write_profile(root):
    rows=[(0,'embedding_out','model.language_model.embed_tokens.weight'),
          (0,'attn_out','model.language_model.layers.0.linear_attn.out_proj.weight'),
          (0,'mlp_out','model.language_model.layers.0.mlp.down_proj.weight')]
    meta={'name':'test','model':{'id':'test-model','revision':'pinned'},'hidden_size':2,
          'sites':[{'layer':layer,'site':site,'tensor':name,'rank':1,'scale':1.}
                   for layer,site,name in rows]}
    (root/'profile.json').write_text(json.dumps(meta))
    save_file({f'L{layer}.{site}.Q':torch.tensor([[1.,0.]]) for layer,site,_ in rows},root/'profile.safetensors')
    return Profile(root)


class Runtime(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.profile=write_profile(self.root)
        self.store={'test':self.profile};self.owner=Selection('test',self.profile.digest,1.)
        self.key=b'k'*32;self.now=1_800_000_000

    def signed(self,raw,nonce='unique_nonce_123456',stamp=None):
        stamp=str(self.now if stamp is None else stamp)
        digest=hmac.new(self.key,(stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()).encode(),hashlib.sha256).hexdigest()
        return f'v1:{stamp}:{nonce}:{digest}'

    def test_off_bypasses_arithmetic_including_nonfinite_values(self):
        y=torch.tensor([[float('nan'),-0.],[float('inf'),1.]])
        with selected_rows(()):self.assertIs(project(y,self.store,0,'attn_out'),y)

    def test_mixed_rows_are_selected_and_output_not_mutated(self):
        y=torch.tensor([[4.,2.],[8.,3.],[9.,4.]])
        with selected_rows([(1,2,self.owner)]):got=project(y,self.store,0,'attn_out')
        torch.testing.assert_close(got,torch.tensor([[4.,2.],[0.,3.],[9.,4.]]))
        torch.testing.assert_close(y,torch.tensor([[4.,2.],[8.,3.],[9.,4.]]))

    def test_exact_strength_interpolates(self):
        owner=Selection('test',self.profile.digest,.25)
        with selected_rows([(0,1,owner)]):got=project(torch.tensor([[8.,3.]]),self.store,0,'mlp_out')
        torch.testing.assert_close(got,torch.tensor([[6.,3.]]))

    def test_foreign_profile_digest_and_row_overrun_reject(self):
        with selected_rows([(0,1,Selection('test','foreign',1.))]),self.assertRaises(ValueError):
            project(torch.zeros(1,2),self.store,0,'attn_out')
        with selected_rows([(0,2,self.owner)]),self.assertRaises(ValueError):
            project(torch.zeros(1,2),self.store,0,'attn_out')

    def test_saved_profile_invalid_basis_and_extension_rejected(self):
        meta=json.loads((self.root/'profile.json').read_text());meta['requires']=['matrix_input']
        (self.root/'profile.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError):Profile(self.root)
        meta.pop('requires');(self.root/'profile.json').write_text(json.dumps(meta))
        save_file({'L0.attn_out.Q':torch.tensor([[2.,0.]])},self.root/'profile.safetensors')
        with self.assertRaises(ValueError):Profile(self.root)

    def test_auth_valid_missing_tampered_expired_and_replay(self):
        raw=b'{"owner_mode":"test"}';body=json.loads(raw)
        gate=Gate(self.store,self.key,clock=lambda:self.now)
        self.assertIsNone(gate.authorize(raw,body,[]))
        self.assertIsNone(gate.authorize(raw+b' ',body,[self.signed(raw)]))
        self.assertIsNone(gate.authorize(raw,body,[self.signed(raw,stamp=self.now-61)]))
        self.assertEqual(gate.authorize(raw,body,[self.signed(raw)]),self.owner)
        self.assertIsNone(gate.authorize(raw,body,[self.signed(raw)]))

    def test_unknown_profile_only_errors_after_valid_auth(self):
        raw=b'{"owner_mode":"unknown"}';body=json.loads(raw)
        gate=Gate(self.store,self.key,clock=lambda:self.now)
        self.assertIsNone(gate.authorize(raw,body,[]))
        with self.assertRaises(FieldError):gate.authorize(raw,body,[self.signed(raw)])

    def test_bad_scale_and_type_fail(self):
        gate=Gate(self.store,self.key)
        for scale in [True,0,3,float('nan')]:
            with self.assertRaises(FieldError):gate.authorize(b'',{'owner_mode':'test','owner_mode_scale':scale},[])
        with self.assertRaises(FieldError):gate.authorize(b'',{'owner_mode':[]},[])

    def test_nonce_store_full_fails_closed(self):
        gate=Gate(self.store,self.key,clock=lambda:self.now)
        gate.nonces={str(i):self.now+60 for i in range(100000)}
        raw=b'{"owner_mode":"test"}'
        self.assertIsNone(gate.authorize(raw,json.loads(raw),[self.signed(raw)]))

    def test_key_permission_and_length(self):
        p=self.root/'key';p.write_bytes(self.key);p.chmod(0o600)
        self.assertEqual(load_key(p),self.key)
        p.chmod(0o644)
        with self.assertRaises(ValueError):load_key(p)
        p.chmod(0o600);p.write_bytes(b'short')
        with self.assertRaises(ValueError):load_key(p)

    def test_cache_namespaces_cover_profile_strength_and_model(self):
        keys={cache_key(x,self.store,m,'client') for x in [None,self.owner,Selection('test',self.profile.digest,.5)] for m in ['a','b']}
        self.assertEqual(len(keys),6)

    def test_runner_map_follows_request_order_and_query_lengths(self):
        params=lambda mode:SimpleNamespace(extra_args=mode)
        marked={'_flashalign_profile':'test','_flashalign_digest':self.profile.digest,'_flashalign_scale':.5}
        r=SimpleNamespace(input_batch=SimpleNamespace(req_ids=['normal','owner']),
            query_start_loc=SimpleNamespace(np=[0,2,5]),requests={
                'normal':SimpleNamespace(sampling_params=params({})),
                'owner':SimpleNamespace(sampling_params=params(marked))})
        self.assertEqual(spans_for_runner(r,self.store),[(2,5,Selection('test',self.profile.digest,.5))])

    def test_actual_module_hooks_cover_embedding_attention_and_mlp(self):
        class Linear(torch.nn.Module):
            def forward(self,x):return x*2,None
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed_tokens=torch.nn.Identity()
                layer=torch.nn.Module();layer.linear_attn=torch.nn.Module();layer.linear_attn.out_proj=Linear()
                layer.mlp=torch.nn.Module();layer.mlp.down_proj=Linear();self.layers=torch.nn.ModuleList([layer])
        model=type('Qwen3_5ForCausalLM',(torch.nn.Module,),{'__init__':lambda s:(torch.nn.Module.__init__(s),setattr(s,'model',Inner()))[-1]})()
        cfg=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1),
            model_config=SimpleNamespace(enforce_eager=True,enable_prompt_embeds=False,
                hf_text_config=SimpleNamespace(hidden_size=2,num_hidden_layers=1,layer_types=['linear_attention'])),
            speculative_config=SimpleNamespace(method='ngram'))
        runner=SimpleNamespace(vllm_config=cfg,device='cpu',get_model=lambda:model)
        attach(runner,self.store)
        with selected_rows([(0,1,self.owner)]):
            for module in [model.model.embed_tokens,model.model.layers[0].linear_attn.out_proj,model.model.layers[0].mlp.down_proj]:
                got=module(torch.tensor([[4.,3.],[5.,6.]]));got=got[0] if isinstance(got,tuple) else got
                self.assertEqual(got[0,0].item(),0.)
                self.assertNotEqual(got[1,0].item(),0.)

    def test_middleware_strips_forged_worker_metadata_and_adds_actual_header(self):
        async def run():
            captured={};sent=[]
            async def app(scope,receive,send):
                captured.update(json.loads((await receive())['body']))
                await send({'type':'http.response.start','status':200,'headers':[]})
                await send({'type':'http.response.body','body':b'{}'})
            env={'FLASHALIGN_PROFILES':json.dumps([str(self.root)]),'FLASHALIGN_KEY_FILE':str(self.root/'key'),
                 'FLASHALIGN_MODEL_ID':'test-model',
                 'FLASHALIGN_REPLAY_STORE':str(self.root/'replay'/'nonces.sqlite3')}
            (self.root/'key').write_bytes(self.key);(self.root/'key').chmod(0o600)
            with patch.dict(os.environ,env),patch('flashalign_vllm.plugin.REGISTERED',True):middleware=OwnerMiddleware(app)
            body={'owner_mode':'test','vllm_xargs':{'_flashalign_profile':'test','_flashalign_scale':1}}
            raw=json.dumps(body).encode()
            async def receive():return {'type':'http.request','body':raw,'more_body':False}
            async def send(m):sent.append(m)
            await middleware({'type':'http','method':'POST','path':'/v1/chat/completions','headers':[]},receive,send)
            self.assertNotIn('vllm_xargs',captured)
            self.assertTrue(captured['cache_salt'].startswith('flashalign:'))
            self.assertIn((b'x-owner-mode-effective',b'off'),sent[0]['headers'])
        asyncio.run(run())

    def test_alternative_api_cannot_reuse_an_owner_cache_salt(self):
        async def run():
            captured={};sent=[]
            async def app(scope,receive,send):
                captured.update(json.loads((await receive())['body']))
                await send({'type':'http.response.start','status':200,'headers':[]})
                await send({'type':'http.response.body','body':b'{}'})
            (self.root/'key').write_bytes(self.key);(self.root/'key').chmod(0o600)
            env={'FLASHALIGN_PROFILES':json.dumps([str(self.root)]),'FLASHALIGN_KEY_FILE':str(self.root/'key'),
                 'FLASHALIGN_MODEL_ID':'test-model',
                 'FLASHALIGN_REPLAY_STORE':str(self.root/'replay'/'nonces.sqlite3')}
            with patch.dict(os.environ,env),patch('flashalign_vllm.plugin.REGISTERED',True):middleware=OwnerMiddleware(app)
            owner_salt=cache_key(self.owner,self.store,'test-model')
            raw=json.dumps({'owner_mode':'test','cache_salt':owner_salt,
                'vllm_xargs':{'_flashalign_profile':'test','_flashalign_digest':self.profile.digest,'_flashalign_scale':1.}}).encode()
            async def receive():return {'type':'http.request','body':raw,'more_body':False}
            async def send(m):sent.append(m)
            await middleware({'type':'http','method':'POST','path':'/v1/responses','headers':[]},receive,send)
            self.assertEqual(captured,{})
            self.assertEqual(sent[0]['status'],400)
            self.assertIn((b'x-owner-mode-effective',b'off'),sent[0]['headers'])
        asyncio.run(run())

    def test_install_does_not_patch_ordinary_vllm_without_profiles(self):
        with patch.dict(os.environ,{},clear=True),patch('flashalign_vllm.plugin.REGISTERED',False),\
             patch('flashalign_vllm.plugin.importlib.metadata.version',side_effect=AssertionError('should not inspect vLLM')):
            register()

    def test_profiles_require_the_secured_launcher(self):
        with patch.dict(os.environ,{'FLASHALIGN_PROFILES':'["profile"]'},clear=True),\
             patch('flashalign_vllm.plugin.REGISTERED',False),self.assertRaisesRegex(RuntimeError,'mandatory HTTP gate'):
            register()

    def test_launcher_preserves_speculation_and_sets_private_defaults(self):
        from flashalign_vllm.cli import main
        (self.root/'key').write_bytes(self.key);(self.root/'key').chmod(0o600)
        captured={}
        module=types.ModuleType('vllm.entrypoints.cli.main')
        module.main=lambda:captured.update(argv=list(sys.argv),environment=dict(os.environ))
        spec={'method':'ngram','num_speculative_tokens':3,'prompt_lookup_min':2,'prompt_lookup_max':3}
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'2'},clear=True),\
             patch.dict(sys.modules,{'vllm.entrypoints.cli.main':module}),\
             patch('flashalign_vllm.plugin.register'),patch.object(sys,'argv',['flashalign-vllm']):
            main(['--model','/local/model','--profile',str(self.root),'--key-file',str(self.root/'key'),
                  '--speculative-config',json.dumps(spec)])
        argv=captured['argv'];environment=captured['environment']
        self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1]),spec)
        self.assertEqual(argv[argv.index('--served-model-name')+1],'test-model')
        self.assertIn('--enable-auto-tool-choice',argv)
        self.assertEqual(argv[argv.index('--tool-call-parser')+1],'qwen3_coder')
        self.assertEqual(environment['NUMBA_NUM_THREADS'],'2')
        self.assertEqual(environment['VLLM_NO_USAGE_STATS'],'1')
        self.assertEqual(environment['HF_HUB_DISABLE_TELEMETRY'],'1')


if __name__=='__main__':unittest.main()
