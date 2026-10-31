# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CPU operator/protocol tests; synthetic modules are not a GLM engine test."""
import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flashalign_sglang import runtime as rt
from flashalign_sglang.gate import Gate,Selection,FieldError
from flashalign_sglang.http import Middleware


class Mode:
    def __init__(self,name):self.name=name
    def is_decode(self):return self.name=='decode'
    def is_extend(self):return self.name=='extend'
    def is_target_verify(self):return self.name=='verify'
    def is_draft_extend_v2(self):return self.name=='draft_extend'
    def is_idle(self):return self.name=='idle'


class Runtime(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(90210)
        self.operator={'q':torch.tensor([[1.,0,0,0]]),'m':torch.ones(4),'g':torch.ones(4),'scale':1.}
        self.profile=rt.Profile('test','digest',rt.MODEL,{(12,'attn_out'):self.operator})
        self.profiles={'test':self.profile};self.gate=Gate(self.profiles,b'x'*32,clock=lambda:1000000000)
        self.owner=Selection('test','digest',1.)
    def batch(self,name,salts,**kwargs):return SimpleNamespace(forward_mode=Mode(name),flashalign_cache_salts=salts,**kwargs)
    def sign(self,raw,nonce='0'*32,ts='1000000000'):
        msg=ts+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
        return 'v1:'+ts+':'+nonce+':'+hmac.new(b'x'*32,msg.encode(),hashlib.sha256).hexdigest()
    def owner_salt(self):return self.gate.cache_salt(self.owner,None)
    def test_signature_replay_and_exact_raw_body(self):
        body={'owner_mode':'test'};raw=json.dumps(body).encode();header=self.sign(raw)
        self.assertIsNone(self.gate.select(raw+b' ',body,[header]))
        self.assertEqual(self.gate.select(raw,body,[header]),self.owner)
        self.assertIsNone(self.gate.select(raw,body,[header]))
    def test_missing_invalid_expired_and_duplicate_headers(self):
        body={'owner_mode':'test'};raw=json.dumps(body).encode()
        for headers in ([],['bad'],[self.sign(raw,ts='999999800')],[self.sign(raw),self.sign(raw)]):
            self.assertIsNone(self.gate.select(raw,body,headers))
    def test_unknown_profile_requires_valid_auth_before_lookup(self):
        body={'owner_mode':'absent'};raw=json.dumps(body).encode()
        self.assertIsNone(self.gate.select(raw,body,[]))
        with self.assertRaises(FieldError):self.gate.select(raw,body,[self.sign(raw)])
    def test_cache_isolates_mode_profile_strength_and_copied_salt(self):
        owner=self.owner_salt()
        self.assertIsNone(self.gate.from_salt(self.gate.cache_salt(None,owner)))
        self.assertNotEqual(owner,self.gate.cache_salt(Selection('test','digest',.5),None))
        other=Gate(self.profiles,b'x'*32,model_scope='other')
        with self.assertRaises(ValueError):other.from_salt(owner)
    def test_off_bypasses_all_math_and_preserves_nan_signed_zero(self):
        x=torch.tensor([[float('nan'),-0.,1,2]])
        with patch.object(rt,'normalized',side_effect=AssertionError('off arithmetic')):
            self.assertIs(rt.project(x,self.batch('unknown',[None]),(12,'attn_out'),self.profiles,self.gate),x)
    def test_mixed_decode_leaves_stock_and_padding_bytes_identical(self):
        x=torch.arange(16,dtype=torch.float32).reshape(4,4);x[0,0]=-0.
        y=rt.project(x,self.batch('decode',[None,self.owner_salt(),None]),(12,'attn_out'),self.profiles,self.gate)
        self.assertEqual(y[1,0],0)
        self.assertTrue(torch.equal(y[[0,2,3]].view(torch.uint8),x[[0,2,3]].view(torch.uint8)))
    def test_prefill_lengths(self):
        x=torch.ones(5,4)
        y=rt.project(x,self.batch('extend',[self.owner_salt(),None],extend_seq_lens_cpu=[2,3]),(12,'attn_out'),self.profiles,self.gate)
        self.assertEqual(y[:,0].tolist(),[0,0,1,1,1])
    def test_native_mtp_verify_and_extend_layouts(self):
        for mode in ('verify','draft_extend'):
            x=torch.ones(6,4)
            y=rt.project(x,self.batch(mode,[None,self.owner_salt()],spec_info=SimpleNamespace(num_tokens_per_req=3)),(12,'attn_out'),self.profiles,self.gate)
            self.assertEqual(y[:,0].tolist(),[1,1,1,0,0,0])
    def test_unknown_mtp_geometry_fails(self):
        with self.assertRaises(ValueError):rt.row_ranges(self.batch('verify',[self.owner_salt()],spec_info=SimpleNamespace()),1)
        with self.assertRaises(ValueError):rt.row_ranges(self.batch('verify',[self.owner_salt()],spec_info=SimpleNamespace(num_tokens_per_req=5)),4)
    def test_normalized_formula_matches_explicit_weight_edit(self):
        q=torch.linalg.qr(torch.randn(4,1))[0].T
        op={'q':q,'m':torch.tensor([.2,.7,1.9,3.]),'g':torch.tensor([.8,1.1,1.3,.6]),'scale':1.37}
        w=torch.randn(4,7);x=torch.randn(11,7);identity=torch.eye(4)
        edited=torch.diag(op['m']*op['g'])@(identity-op['scale']*q.T@q)@torch.diag(1/op['m'])@w
        for alpha in (.15,1.,1.7):
            actual=rt.normalized(x@w.T,op,alpha)
            expected=x@(w+alpha*(edited-w)).T
            torch.testing.assert_close(actual,expected,atol=5e-6,rtol=5e-6)
        original=x@w.T;self.assertIs(rt.normalized(original,op,0),original)
    def test_normalized_operator_commutes_with_tp_sum_in_fp32(self):
        op={**self.operator,'m':torch.tensor([.5,2,3,1.]),'g':torch.tensor([.7,1.3,.8,1.]),'scale':.6}
        parts=[torch.randn(6,4) for _ in range(4)]
        torch.testing.assert_close(sum(rt.normalized(p,op,.4) for p in parts),rt.normalized(sum(parts),op,.4),atol=1e-6,rtol=2e-6)
    def args(self,**extra):
        return SimpleNamespace(tp_size=4,ep_size=1,disable_cuda_graph=True,disable_shared_experts_fusion=True,
                               moe_runner_backend='triton',speculative_algorithm='EAGLE',**extra)
    def test_preflight_preserves_active_speculation(self):
        args=self.args()
        with patch.object(rt,'registry',return_value=(self.profiles,self.gate)),patch.dict(os.environ,{},clear=False):rt.check_server_args(args)
        self.assertEqual(args.speculative_algorithm,'EAGLE')
    def test_preflight_refuses_unsupported_spec_and_fused_shared_path(self):
        with patch.object(rt,'registry',return_value=(self.profiles,self.gate)):
            for field,value in [('speculative_algorithm','DFLASH'),('tp_size',2),('ep_size',4),('disable_shared_experts_fusion',False),('moe_runner_backend','auto'),('enable_two_batch_overlap',True)]:
                args=self.args();setattr(args,field,value)
                with self.subTest(field=field),self.assertRaises(ValueError):rt.check_server_args(args)
                self.assertEqual(getattr(args,field),value)
    def test_http_preserves_auth_boundary_and_hides_internal_owner_fields(self):
        seen={};out=[]
        async def app(scope,receive,send):
            seen['body']=json.loads((await receive())['body']);seen['headers']=scope['headers']
            await send({'type':'http.response.start','status':200,'headers':[]});await send({'type':'http.response.body','body':b'{}'})
        raw=b'{"owner_mode":"test","owner_mode_scale":0.5}'
        async def receive():return {'type':'http.request','body':raw,'more_body':False}
        async def send(message):out.append(message)
        scope={'type':'http','path':'/v1/chat/completions','method':'POST','headers':[(b'content-type',b'application/json'),(b'x-owner-mode-auth',self.sign(raw).encode())]}
        with patch('flashalign_sglang.http.registry',return_value=(self.profiles,self.gate)):asyncio.run(Middleware(app)(scope,receive,send))
        self.assertNotIn('owner_mode',seen['body']);self.assertNotIn('owner_mode_scale',seen['body'])
        self.assertFalse(any(k==b'x-owner-mode-auth' for k,v in seen['headers']))
        self.assertEqual(self.gate.from_salt(seen['body']['cache_salt']).scale,.5)
        self.assertIn((b'x-owner-mode-effective',b'test'),out[0]['headers'])


    def test_real_module_hook_boundaries_in_synthetic_main_and_mtp_modules(self):
        # The modules are synthetic; this checks installation, closure role
        # selection and that the routed branch remains outside the shared edit.
        class Writer(torch.nn.Module):
            output_size=4096
            def forward(self,x):return x*2,None
        class Moe(torch.nn.Module):
            num_fused_shared_experts=0
            _fuse_shared_experts_inside_sbo=False
            _enable_a2a_moe=False
            shared_experts=True
            def _forward_shared_experts(self,x):return x*3
            def forward(self,x):return x*7+self._forward_shared_experts(x)
        q=torch.zeros(1,4096);q[0,0]=1
        operator={'q':q,'m':torch.ones(4096),'g':torch.ones(4096),'scale':1.}
        profile=rt.Profile('test','digest',rt.MODEL,{(layer,kind):dict(operator) for layer in (12,45) for kind in ('attn_out','shared_expert_out')})
        profiles={'test':profile};gate=Gate(profiles,b'x'*32)
        salt=gate.cache_salt(self.owner,None)
        for draft,layer in [(False,12),(True,45)]:
            prefix='model.decoder' if draft else 'model.layers.12'
            writer,moe=Writer(),Moe()
            kind='Glm5NextForConditionalGenerationNextN' if draft else 'Glm5NextForConditionalGeneration'
            model=type(kind,(),{'named_modules':lambda self:[(prefix+'.self_attn.o_proj',writer),(prefix+'.mlp',moe)]})()
            class Runner:
                device='cpu'
                @rt.owner_forward
                def forward(self,batch,x):return writer(x)[0],moe(x)
            runner=Runner();runner.is_draft_worker=draft;runner.model=model
            with patch.object(rt,'registry',return_value=(profiles,gate)),patch.dict(os.environ,{'FLASHALIGN_HTTP_GATE':'1'}):rt.install(runner)
            x=torch.ones(2,4096);attention,mlp=runner.forward(self.batch('decode',[None,salt]),x)
            self.assertEqual(attention[:,0].tolist(),[2,0])
            self.assertEqual(mlp[:,0].tolist(),[10,7])
            self.assertTrue(torch.equal(mlp[:,1:],torch.full_like(mlp[:,1:],10)))


class ProfileFiles(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        self.sites=json.loads((ROOT/'expected-sites.json').read_text())
        self.meta={'name':'test','model':rt.MODEL,'source':{'method':'synthetic CPU fixture'},'hidden_size':4096,
                   'status':'synthetic_test','transform':'row_normalized_projection_v1',
                   'requires':['row_gain','row_norm_preconditioning'],'sites':[{**s,'rank':1,'scale':1.} for s in self.sites]}
        q=torch.zeros(1,4096);q[0,0]=1
        self.tensors={f'L{s["layer"]}.{s["site"]}.{name}':value.clone() for s in self.sites for name,value in [('Q',q),('m',torch.ones(4096)),('g',torch.ones(4096))]}
        self.save()
    def save(self):
        (self.path/'profile.json').write_text(json.dumps(self.meta));save_file(self.tensors,self.path/'profile.safetensors')
    def test_loads_all_target_and_mtp_sites(self):
        p=rt.load_profile(self.path);self.assertEqual(len(p.sites),60)
        self.assertIn((45,'attn_out'),p.sites);self.assertIn((45,'shared_expert_out'),p.sites)
    def test_refuses_missing_mtp_or_extra_tensors(self):
        self.meta['sites']=[s for s in self.meta['sites'] if s['layer']!=45];self.save()
        with self.assertRaises(ValueError):rt.load_profile(self.path)
    def test_refuses_bad_norm_and_unknown_transform(self):
        key=next(k for k in self.tensors if k.endswith('.m'));self.tensors[key][0]=0;self.save()
        with self.assertRaises(ValueError):rt.load_profile(self.path)
        self.tensors[key][0]=1;self.meta['transform']='other';self.save()
        with self.assertRaises(ValueError):rt.load_profile(self.path)


class ResourceBounds(unittest.TestCase):
    def test_disk_and_vram_shortfall_are_hard_failures(self):
        from resource_preflight import bounds
        resources=json.loads((ROOT/'resources.json').read_text())
        result=bounds(resources,130*1024**3,gpu_free=[70*1024**3]*4)
        self.assertFalse(result['necessary_bounds_pass'])
        self.assertEqual(len(result['failures']),2)
        self.assertFalse(result['launch_qualified'])
    def test_passing_necessary_bounds_does_not_claim_memory_or_launch_success(self):
        from resource_preflight import bounds
        resources=json.loads((ROOT/'resources.json').read_text())
        result=bounds(resources,600*1024**3,gpu_free=[95*1024**3]*4,host_available=200*1024**3)
        self.assertTrue(result['necessary_bounds_pass'])
        self.assertFalse(result['launch_qualified'])


if __name__=='__main__':unittest.main()
