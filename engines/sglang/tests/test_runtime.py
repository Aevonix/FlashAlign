# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from flashalign_sglang.gate import Gate,FieldError,Selection,read_key
from flashalign_sglang.runtime import Profile,project,row_ranges,load_profile,registry,check_server_args,install,owner_forward
from flashalign_sglang.http import Middleware


class Mode:
    def __init__(self,kind):self.kind=kind
    def is_decode(self):return self.kind=='decode'
    def is_extend(self):return self.kind=='extend'
    def is_target_verify(self):return self.kind=='verify'
    def is_idle(self):return self.kind=='idle'


class Runtime(unittest.TestCase):
    def setUp(self):
        self.profile=Profile('test','digest',{'id':'test','revision':'pin'},4,
            {(0,'attn_out'):{'q':torch.tensor([[1.,0.,0.,0.]]),'scale':1.,'tensor':'model.language_model.layers.0.self_attn.o_proj.weight'}})
        self.profiles={'test':self.profile};self.gate=Gate(self.profiles,b'x'*32,clock=lambda:1000000000)
        self.owner=Selection('test','digest',1.)
    def sign(self,raw,nonce='0'*32,ts='1000000000'):
        msg=ts+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
        return 'v1:'+ts+':'+nonce+':'+hmac.new(b'x'*32,msg.encode(),hashlib.sha256).hexdigest()
    def batch(self,mode,salts,**kwargs):
        return SimpleNamespace(forward_mode=Mode(mode),flashalign_cache_salts=salts,**kwargs)
    def test_valid_and_replayed_signature(self):
        body={'owner_mode':'test'};raw=json.dumps(body).encode();header=self.sign(raw)
        self.assertEqual(self.gate.select(raw,body,[header]),self.owner)
        self.assertIsNone(self.gate.select(raw,body,[header]))
    def test_bad_missing_expired_and_duplicate_headers(self):
        body={'owner_mode':'test'};raw=json.dumps(body).encode()
        for headers in ([],['invalid'],[self.sign(raw,ts='999999800')],[self.sign(raw),self.sign(raw)]):
            self.assertIsNone(self.gate.select(raw,body,headers))
    def test_body_tamper_invalidates_signature(self):
        self.assertIsNone(self.gate.select(b'b',{'owner_mode':'test'},[self.sign(b'a')]))
    def test_unknown_authentication_happens_before_lookup(self):
        body={'owner_mode':'unknown'};raw=json.dumps(body).encode()
        self.assertIsNone(self.gate.select(raw,body,[]))
        with self.assertRaisesRegex(FieldError,'no profile'):
            self.gate.select(raw,body,[self.sign(raw)])
    def test_scale_and_type_are_rejected(self):
        for body in ({'owner_mode':3},{'owner_mode':'test','owner_mode_scale':0},
                     {'owner_mode_scale':True},{'owner_mode_scale':float('nan')}):
            with self.assertRaises(FieldError):self.gate.select(b'',body,[])
    def test_copied_owner_salt_is_wrapped_into_off_namespace(self):
        owner=self.gate.cache_salt(self.owner,None)
        copied=self.gate.cache_salt(None,owner)
        self.assertIsNone(self.gate.from_salt(copied))
        self.assertEqual(self.gate.from_salt(owner),self.owner)
    def test_cache_separates_strength_and_user_salt(self):
        salt=self.gate.cache_salt(self.owner,None)
        self.assertNotEqual(salt,self.gate.cache_salt(Selection('test','digest',.5),None))
        self.assertNotEqual(salt,self.gate.cache_salt(self.owner,'different'))
    def test_internal_salt_tamper_rejected(self):
        salt=self.gate.cache_salt(self.owner,None)
        with self.assertRaises(ValueError):self.gate.from_salt(salt[:-1]+('0' if salt[-1]!='0' else '1'))
    def test_off_no_arithmetic_even_nan_signed_zero(self):
        values=torch.tensor([[float('nan'),-0.,1.,2.]])
        result=project(values,self.batch('decode',[self.gate.cache_salt(None,None)]),(0,'attn_out'),self.profiles,self.gate)
        self.assertIs(result,values)
    def test_off_unknown_layout_keeps_stock_execution(self):
        values=torch.ones(3,4)
        batch=self.batch('unrecognized',[None,None])
        self.assertIs(project(values,batch,(0,'attn_out'),self.profiles,self.gate),values)
    def test_decode_selected_rows(self):
        values=torch.arange(12,dtype=torch.float32).reshape(3,4)
        b=self.batch('decode',[None,self.gate.cache_salt(self.owner,None),None])
        actual=project(values,b,(0,'attn_out'),self.profiles,self.gate)
        expected=values.clone();expected[1,0]=0
        torch.testing.assert_close(actual,expected)
        self.assertTrue(torch.equal(actual[[0,2]],values[[0,2]]))
    def test_prefill_mixed_length_rows(self):
        values=torch.ones(5,4)
        b=self.batch('extend',[self.gate.cache_salt(self.owner,None),None],extend_seq_lens_cpu=[2,3])
        actual=project(values,b,(0,'attn_out'),self.profiles,self.gate)
        self.assertEqual(actual[:,0].tolist(),[0,0,1,1,1])
    def test_speculative_verification_rows_are_per_request(self):
        values=torch.ones(6,4)
        b=self.batch('verify',[None,self.gate.cache_salt(self.owner,None)],spec_info=SimpleNamespace(num_tokens_per_req=3))
        actual=project(values,b,(0,'attn_out'),self.profiles,self.gate)
        self.assertEqual(actual[:,0].tolist(),[1,1,1,0,0,0])
    def test_unknown_spec_layout_rejected(self):
        b=self.batch('verify',[None,None],spec_info=SimpleNamespace(num_tokens_per_req=4))
        with self.assertRaisesRegex(ValueError,'layout'):row_ranges(b,6)
    def test_eager_setup_does_not_disable_speculative_algorithm(self):
        args=SimpleNamespace(disable_cuda_graph=True,speculative_algorithm='DFLASH')
        with patch('flashalign_sglang.runtime.registry',return_value=(self.profiles,self.gate)):
            check_server_args(args)
        self.assertEqual(args.speculative_algorithm,'DFLASH')
    def test_graph_and_distributed_setup_rejected(self):
        with patch('flashalign_sglang.runtime.registry',return_value=(self.profiles,self.gate)):
            with self.assertRaisesRegex(ValueError,'eager'):check_server_args(SimpleNamespace())
            with self.assertRaisesRegex(ValueError,'TP/PP/DP'):check_server_args(SimpleNamespace(disable_cuda_graph=True,tp_size=2))
    def test_profile_load_and_unexpected_tensors(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);manifest={'name':'test','hidden_size':4,'model':{'id':'test','revision':'pin'},'sites':[
                {'layer':0,'site':'attn_out','rank':1,'scale':1.,'tensor':'model.language_model.layers.0.self_attn.o_proj.weight'}]}
            (p/'profile.json').write_text(json.dumps(manifest))
            q=torch.tensor([[1.,0.,0.,0.]])
            save_file({'L0.attn_out.Q':q},p/'profile.safetensors')
            self.assertEqual(load_profile(p).hidden,4)
            save_file({'L0.attn_out.Q':q,'unexpected':q.clone()},p/'profile.safetensors')
            with self.assertRaisesRegex(ValueError,'unexpected'):load_profile(p)
    def test_key_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'key').write_bytes(b'x'*32);(p/'key').chmod(0o600)
            self.assertEqual(read_key(p/'key'),b'x'*32)
            (p/'link').symlink_to(p/'key')
            with self.assertRaises(OSError):read_key(p/'link')
    def test_pinned_full_attention_module_mapping_and_tuple_hook(self):
        class Linear(torch.nn.Linear):
            output_size=4
            def forward(self,value):return super().forward(value),None
        class Qwen3_5ForCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=torch.nn.Module()
                layer=torch.nn.Module();layer.o_proj=Linear(4,4,bias=False)
                self.model.layers=torch.nn.ModuleList([layer])
        class Runner:
            is_draft_worker=False
            device='cpu'
            def __init__(self):self.model=Qwen3_5ForCausalLM()
            @owner_forward
            def forward(self,batch,values):return self.model.model.layers[0].o_proj(values)
        runner=Runner();values=torch.ones(2,4);expected=runner.forward(self.batch('decode',[None,None]),values)[0]
        with patch.dict(os.environ,{'FLASHALIGN_HTTP_GATE':'1'}),patch('flashalign_sglang.runtime.registry',return_value=(self.profiles,self.gate)):install(runner)
        actual,bias=runner.forward(self.batch('decode',[None,self.gate.cache_salt(self.owner,None)]),values)
        self.assertIsNone(bias)
        torch.testing.assert_close(actual[0],expected[0])
        self.assertEqual(actual[1,0].item(),0.)
        torch.testing.assert_close(actual[1,1:],expected[1,1:])
    def test_draft_worker_is_not_transformed(self):
        runner=SimpleNamespace(is_draft_worker=True)
        with patch('flashalign_sglang.runtime.registry',side_effect=AssertionError('draft profile load')):install(runner)
        self.assertFalse(hasattr(runner,'flashalign_handles'))
    def test_asgi_strips_auth_and_sets_effective_header(self):
        seen={};messages=[]
        async def app(scope,receive,send):
            seen['headers']=scope['headers'];seen['body']=json.loads((await receive())['body'])
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        body={'owner_mode':'test','owner_mode_scale':.5};raw=json.dumps(body).encode()
        async def receive():return {'type':'http.request','body':raw,'more_body':False}
        async def send(event):messages.append(event)
        scope={'type':'http','path':'/v1/chat/completions','method':'POST',
               'headers':[(b'x-owner-mode-auth',self.sign(raw).encode()),(b'content-type',b'application/json')]}
        with patch('flashalign_sglang.http.registry',return_value=(self.profiles,self.gate)):
            asyncio.run(Middleware(app)(scope,receive,send))
        self.assertNotIn('owner_mode',seen['body'])
        self.assertFalse(any(k==b'x-owner-mode-auth' for k,v in seen['headers']))
        self.assertEqual(self.gate.from_salt(seen['body']['cache_salt']).scale,.5)
        self.assertIn((b'x-owner-mode-effective',b'test'),messages[0]['headers'])
    def test_other_json_api_cannot_replay_owner_cache_namespace(self):
        seen={};messages=[]
        async def app(scope,receive,send):
            seen.update(json.loads((await receive())['body']))
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        body={'cache_salt':self.gate.cache_salt(self.owner,None),'owner_mode':'test'}
        raw=json.dumps(body).encode()
        async def receive():return {'type':'http.request','body':raw,'more_body':False}
        async def send(event):messages.append(event)
        scope={'type':'http','path':'/v1/responses','method':'POST','headers':[(b'content-type',b'application/json')]}
        with patch('flashalign_sglang.http.registry',return_value=(self.profiles,self.gate)):
            asyncio.run(Middleware(app)(scope,receive,send))
        self.assertEqual(seen,{})
        self.assertEqual(messages[0]['status'],405)
    def test_non_ascii_authentication_serves_stock(self):
        seen={};messages=[]
        async def app(scope,receive,send):
            seen.update(json.loads((await receive())['body']))
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        raw=b'{"owner_mode":"test"}'
        async def receive():return {'type':'http.request','body':raw,'more_body':False}
        async def send(event):messages.append(event)
        scope={'type':'http','path':'/v1/chat/completions','method':'POST','headers':[(b'content-type',b'application/json'),(b'x-owner-mode-auth',b'\xff')]}
        with patch('flashalign_sglang.http.registry',return_value=(self.profiles,self.gate)):
            asyncio.run(Middleware(app)(scope,receive,send))
        self.assertIsNone(self.gate.from_salt(seen['cache_salt']))
        self.assertIn((b'x-owner-mode-effective',b'off'),messages[0]['headers'])


if __name__=='__main__':unittest.main()
