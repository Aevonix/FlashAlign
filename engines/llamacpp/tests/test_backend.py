# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import hashlib
import hmac
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.environ.get('FLASHALIGN_GGUF_PYTHON'):
    sys.path.insert(0,os.environ['FLASHALIGN_GGUF_PYTHON'])
from launcher import Gate,FieldError,speculative_arguments
from profile_gguf import read_profile,convert
import gguf


def write_profile(root,width=4,kind='embedding_out',layer=0,scale=1.):
    q=np.array([[1.,0.,0.,0.]],dtype='<f4') if width==4 else np.eye(1,width,dtype='<f4')
    key=f'L{layer}.{kind}.Q';raw=q.tobytes()
    header=json.dumps({key:{'dtype':'F32','shape':[1,width],'data_offsets':[0,len(raw)]}}).encode()
    (root/'profile.safetensors').write_bytes(len(header).to_bytes(8,'little')+header+raw)
    target={'embedding_out':'model.language_model.embed_tokens.weight',
            'attn_out':f'model.language_model.layers.{layer}.self_attn.o_proj.weight',
            'mlp_out':f'model.language_model.layers.{layer}.mlp.down_proj.weight'}[kind]
    (root/'profile.json').write_text(json.dumps({'name':'test','hidden_size':width,'model':{'id':'fixture','revision':'test'},'sites':[{'layer':layer,'site':kind,'rank':1,'scale':scale,'tensor':target}]}))


def write_model(path,architecture='gemma4',width=4,target='blk.5.attn_output.weight'):
    writer=gguf.GGUFWriter(path,architecture)
    writer.add_tensor(target,np.zeros((width,6),dtype=np.float32))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()


class Authentication(unittest.TestCase):
    def setUp(self):self.gate=Gate(b'x'*32,['test'],clock=lambda:1000)
    def sign(self,data,nonce='a'*32,time='1000'):
        mac=hmac.new(b'x'*32,(time+'\n'+nonce+'\n'+hashlib.sha256(data).hexdigest()).encode(),hashlib.sha256).hexdigest()
        return 'v1:'+time+':'+nonce+':'+mac
    def test_unsigned_and_tampering_are_off(self):
        raw=b'{"owner_mode":"test"}';body=json.loads(raw)
        self.assertEqual(self.gate.select(raw,body,[]),('off',1.))
        self.assertEqual(self.gate.select(raw+b' ',body,[self.sign(raw)]),('off',1.))
    def test_signed_and_replay(self):
        raw=b'{"owner_mode":"test"}';header=self.sign(raw)
        self.assertEqual(self.gate.select(raw,json.loads(raw),[header]),('test',1.))
        self.assertEqual(self.gate.select(raw,json.loads(raw),[header]),('off',1.))
    def test_expired_multiple_and_capacity_fail_closed(self):
        raw=b'{"owner_mode":"test"}';body=json.loads(raw)
        self.assertEqual(self.gate.select(raw,body,[self.sign(raw,time='939')]),('off',1.))
        self.assertEqual(self.gate.select(raw,body,[self.sign(raw),self.sign(raw)]),('off',1.))
        self.gate.capacity=0
        self.assertEqual(self.gate.select(raw,body,[self.sign(raw)]),('off',1.))
    def test_unknown_only_rejected_after_auth(self):
        raw=b'{"owner_mode":"unknown"}'
        self.assertEqual(self.gate.select(raw,json.loads(raw),[]),('off',1.))
        with self.assertRaisesRegex(FieldError,'no profile'):self.gate.select(raw,json.loads(raw),[self.sign(raw)])
    def test_replay_store_survives_gate_restart(self):
        raw=b'{"owner_mode":"test"}';header=self.sign(raw)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'nonces.db'
            first=Gate(b'x'*32,['test'],clock=lambda:1000,replay_path=path)
            second=Gate(b'x'*32,['test'],clock=lambda:1000,replay_path=path)
            self.assertEqual(first.select(raw,json.loads(raw),[header]),('test',1.))
            self.assertEqual(second.select(raw,json.loads(raw),[header]),('off',1.))
            path.unlink()
            fresh=self.sign(raw,nonce='b'*32)
            self.assertEqual(second.select(raw,json.loads(raw),[fresh]),('off',1.))

    def test_bad_scales_rejected(self):
        for scale in [True,0,-1,2.01,float('nan')]:
            with self.assertRaises(FieldError):self.gate.select(b'',{'owner_mode':'test','owner_mode_scale':scale},[])


class Conversion(unittest.TestCase):
    def test_roundtrip_native_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root)
            receipt=convert(root,root/'out.gguf',ROOT/'upstream/gguf-py')
            parsed=gguf.GGUFReader(root/'out.gguf')
            self.assertEqual(parsed.get_field('adapter.type').contents(),'flashalign_projection')
            self.assertEqual(receipt['sites'],1)
            tensors={x.name:x.data for x in parsed.tensors}
            np.testing.assert_array_equal(tensors['token_embd.weight.lora_b'],-tensors['token_embd.weight.lora_a'].T)
    def test_invalid_scale_and_extension_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root)
            path=root/'profile.json';meta=json.loads(path.read_text())
            for scale in [True,False,0,-1,float('nan'),float('inf'),1e300,1e-300,float(np.finfo(np.float32).max)]:
                meta['sites'][0]['scale']=scale;path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError,'finite, positive FP32'):read_profile(root)
            meta['sites'][0]['scale']=1.;meta['requires']=['matrix_input'];path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'standard output'):read_profile(root)

    def test_largest_safe_strength_remains_finite_at_maximum_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);maximum=float(np.finfo(np.float32).max)/2
            write_profile(root,kind='attn_out',scale=maximum)
            _,items,_=read_profile(root)
            self.assertTrue(np.isfinite(np.float32(items[0][2])*np.float32(2.)))
            self.assertEqual(items[0][2],maximum)

    def test_nonunit_strength_roundtrip_matches_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root,kind='attn_out',layer=5,scale=2.9979970455169678)
            receipt=convert(root,root/'out.gguf',ROOT/'upstream/gguf-py',architecture='gemma4')
            parsed=gguf.GGUFReader(root/'out.gguf')
            self.assertEqual(parsed.get_field('general.architecture').contents(),'gemma4')
            self.assertEqual(parsed.get_field('flashalign.projection.version').contents(),2)
            scale=parsed.get_field('flashalign.projection.scale.blk.5.attn_output.weight').contents()
            self.assertEqual(scale,2.9979970455169678)
            self.assertEqual(receipt['site_scales']['blk.5.attn_output.weight'],scale)
            tensors={x.name:x.data for x in parsed.tensors}
            a=tensors['blk.5.attn_output.weight.lora_a'];b=tensors['blk.5.attn_output.weight.lora_b']
            y=np.array([[2.,3.,5.,7.],[-2.,11.,13.,17.]],dtype=np.float32)
            for alpha in (0.,0.25,1.):
                actual=y+alpha*scale*((y@a.T)@b.T)
                expected=y.copy();expected[:,0]*=1-alpha*scale
                np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_gemma_architecture_inferred_from_model_and_shape_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root,kind='attn_out',layer=5,scale=3.)
            write_model(root/'model.gguf')
            receipt=convert(root,root/'out.gguf',ROOT/'upstream/gguf-py',model=root/'model.gguf')
            self.assertEqual(receipt['architecture'],'gemma4')
            with self.assertRaisesRegex(ValueError,'differs from the model'):
                convert(root,root/'wrong.gguf',ROOT/'upstream/gguf-py',model=root/'model.gguf',architecture='qwen35')
            write_model(root/'wrong-width.gguf',width=5)
            with self.assertRaisesRegex(ValueError,'shape mismatch'):
                convert(root,root/'bad.gguf',ROOT/'upstream/gguf-py',model=root/'wrong-width.gguf')

    def test_gemma_forbids_embedding_site_and_unsupported_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root)
            with self.assertRaisesRegex(ValueError,'attention and dense MLP'):
                convert(root,root/'out.gguf',ROOT/'upstream/gguf-py',architecture='gemma4')
            with self.assertRaisesRegex(ValueError,'architectures only'):
                convert(root,root/'out.gguf',ROOT/'upstream/gguf-py',architecture='llama')
    def test_mismatched_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root)
            (root/'sha256.json').write_text(json.dumps({'profile.json':'0'*64,'profile.safetensors':'0'*64}))
            with self.assertRaisesRegex(ValueError,'checksum'):read_profile(root)

    def test_bad_basis_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_profile(root)
            path=root/'profile.safetensors';data=bytearray(path.read_bytes());data[-16:]=np.zeros(4,dtype='<f4').tobytes();path.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'basis'):read_profile(root)


class Speculation(unittest.TestCase):
    metadata={'speculative_types_by_model_architecture':{'qwen35':['draft-simple'],
                                                        'gemma4':['draft-mtp','ngram-simple']},
              'draftless_speculative_types':['ngram-simple'],
              'default_speculative_type_by_model_architecture':{'gemma4':'draft-mtp'}}

    def test_explicit_method_preserved_for_supported_architecture(self):
        args=['--spec-type','draft-mtp','--spec-draft-n-max','3']
        self.assertEqual(speculative_arguments(args,'draft.gguf','gemma4',self.metadata),args)
        with self.assertRaisesRegex(ValueError,'not qualified'):
            speculative_arguments(args,'draft.gguf','qwen35',self.metadata)

    def test_qwen_default_and_gemma_draftless(self):
        self.assertEqual(speculative_arguments([], 'draft.gguf','qwen35',self.metadata),['--spec-type','draft-simple'])
        self.assertEqual(speculative_arguments([], 'draft.gguf','gemma4',self.metadata),['--spec-type','draft-mtp'])
        args=['--spec-type=ngram-simple']
        self.assertEqual(speculative_arguments(args,None,'gemma4',self.metadata),args)
        self.assertEqual(speculative_arguments([],None,'qwen35',self.metadata),[])
        metadata={**self.metadata,'default_speculative_type_by_model_architecture':{'gemma4':'ngram-simple'}}
        self.assertEqual(speculative_arguments([],None,'gemma4',metadata),['--spec-type','ngram-simple'])
        self.assertEqual(speculative_arguments([],'draft.gguf','gemma4',metadata),['--spec-type','draft-mtp'])

    def test_missing_draft_unknown_method_and_disable_rejected(self):
        with self.assertRaisesRegex(ValueError,'requires an explicit draft'):
            speculative_arguments(['--spec-type','draft-mtp'],None,'gemma4',self.metadata)
        for args in (['--no-drafts'],['--spec-type','none'],['--spec-type'],['--spec-type','--threads','8'],
                     ['--draft=8'],['--spec-ngram-size-n','8']):
            with self.assertRaises(ValueError):speculative_arguments(args,None,'gemma4',self.metadata)

if __name__=='__main__':unittest.main()
