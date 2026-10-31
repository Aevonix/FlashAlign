# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flashalign_sglang.gate import FieldError
from flashalign_sglang.runtime import registry,check_server_args,load_profile,expected_sources


def synthetic_profile(path):
    """Exercise the public format without redistributing a fitted model direction."""
    compatible=json.loads((ROOT/'backend.json').read_text())['compatible_models']
    sites=[]
    keys=[('target_and_native_mtp',0,'text_embedding_out'),('target',1,'ple_value_out')]
    keys += [(role,layer,kind) for role,count in (('target',48),('native_mtp',1))
             for layer in range(count) for kind in ('attn_out','combined_mlp_out')]
    for role,layer,kind in keys:
        site={'model_role':role,'layer':layer,'site':kind,'source_tensors':expected_sources(role,layer,kind)}
        if kind=='combined_mlp_out':
            site.update(expert_count=512,fused_expert_tensor_axes=['expert','output','input'])
        sites.append(site)
    metadata={'schema_version':1,'name':'synthetic-test-profile','model':{'id':'fixture','revision':'fixture'},
              'transform':'residual_writer_group_projection_v1','hidden_size':2560,'rank':1,'scale':1,
              'tensor':'shared.Q','requires':['writer_groups','text_embedding_before_multimodal_merge',
                  'ple_value_projection','native_mtp_writer_groups'],'sites':sites,'compatible_models':compatible}
    (path/'profile.json').write_text(json.dumps(metadata))
    header=json.dumps({'shared.Q':{'dtype':'F32','shape':[1,2560],'data_offsets':[0,10240]}}).encode()
    (path/'profile.safetensors').write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<2560f',1.,*([0.]*2559)))


class PublicContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile=Path(self.temp.name)/'profile'
        self.profile.mkdir()
        synthetic_profile(self.profile)

    def tearDown(self):registry.cache_clear()
    def test_no_profiles_preserves_stock_server_and_pinned_error(self):
        with patch.dict(os.environ,{'FLASHALIGN_PROFILES':'','FLASHALIGN_MODEL_SCOPE':''}):
            registry.cache_clear();profiles,gate=registry();self.assertEqual(profiles,{})
            check_server_args(SimpleNamespace())
            self.assertIsNone(gate.select(b'{}',{},[]))
            with self.assertRaisesRegex(FieldError,'this server was started without owner-mode profiles: owner_mode must be "off"'):
                gate.select(b'{}',{'owner_mode':'test'},[])
    def test_profile_binds_to_fingerprinted_model(self):
        metadata=json.loads((ROOT/'backend.json').read_text())['allowed_model_metadata'][0]
        scope=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            key=Path(directory)/'key';key.write_bytes(b'x'*32);key.chmod(0o600)
            with patch.dict(os.environ,{'FLASHALIGN_PROFILES':str(self.profile),'FLASHALIGN_KEY_FILE':str(key),'FLASHALIGN_MODEL_SCOPE':scope,
                                       'FLASHALIGN_REPLAY_STORE':str(Path(directory)/'replay'/'nonces.sqlite3')}):
                registry.cache_clear();profiles,gate=registry();self.assertEqual(len(profiles),1)
                self.assertEqual(gate.model_scope,scope)
    def test_profile_rejects_another_model_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            key=Path(directory)/'key';key.write_bytes(b'x'*32);key.chmod(0o600)
            with patch.dict(os.environ,{'FLASHALIGN_PROFILES':str(self.profile),'FLASHALIGN_KEY_FILE':str(key),'FLASHALIGN_MODEL_SCOPE':'different'}):
                registry.cache_clear()
                with self.assertRaisesRegex(ValueError,'does not match'):registry()
    def test_invalid_schema_and_boolean_geometry_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);shutil.copyfile(self.profile/'profile.safetensors',path/'profile.safetensors')
            original=json.loads((self.profile/'profile.json').read_text())
            for key,value in (('schema_version',2),('schema_version',True),('rank',True),('scale',True)):
                meta={**original,key:value};(path/'profile.json').write_text(json.dumps(meta))
                with self.assertRaises(ValueError):load_profile(path)


if __name__=='__main__':unittest.main()
