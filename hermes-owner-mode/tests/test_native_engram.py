# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Auxiliary Engram compatibility is checked without reading tensor payloads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock

import test_owner_mode  # loads owner_mode
from owner_mode.native_remote import inspect_engram


class NativeEngram(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.source = self.root/'source'; self.source.mkdir()
        self.config = {'model_type':'deepseek_v41','text_config':{'engram_layer_ids':[1], 'engram_num_embeddings':[4]}}
        (self.source/'config.json').write_text(json.dumps(self.config))
        index={'weight_map':{'layers.1.engram.embed.weight':'engram.safetensors','layers.1.engram.embed.scale':'engram.safetensors'}}
        (self.source/'model.safetensors.index.json').write_text(json.dumps(index))
        header={'layers.1.engram.embed.weight':{'dtype':'F8_E4M3','shape':[4,256],'data_offsets':[0,1024]},
                'layers.1.engram.embed.scale':{'dtype':'F8_E8M0','shape':[4,8],'data_offsets':[1024,1056]}}
        raw=json.dumps(header).encode();self.shard=self.source/'engram.safetensors';self.shard.write_bytes(struct.pack('<Q',len(raw))+raw+bytes(1056))
        self.contract={'source':{'id':'Example/Original','revision':'a'*40,
            'config_sha256':hashlib.sha256((self.source/'config.json').read_bytes()).hexdigest(),
            'index_sha256':hashlib.sha256((self.source/'model.safetensors.index.json').read_bytes()).hexdigest()}}
        self.request={'engram_contract':self.contract,'engram_source':str(self.source),'tp_rank':0,'tp_world':4}

    def test_original_partial_checkpoint_checks_headers_and_sizes(self):
        result=inspect_engram(self.request,self.config)
        self.assertEqual(result['kind'],'safetensors');self.assertEqual(len(result['files']),1)
        self.assertEqual(result['files'][0]['file'],'engram.safetensors')
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'file size'):
            inspect_engram(self.request,self.config)

    def test_wrong_original_metadata_is_rejected(self):
        (self.source/'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'pinned public source'):
            inspect_engram(self.request,self.config)

    def test_missing_and_ambiguous_sources_fail(self):
        request={k:v for k,v in self.request.items() if k!='engram_source'}
        with mock.patch.dict(os.environ,{'HF_HOME':str(self.root/'empty')}):
            with self.assertRaisesRegex(ValueError,'requires original Engram'):
                inspect_engram(request,self.config)
        with self.assertRaisesRegex(ValueError,'Choose one'):
            inspect_engram({**self.request,'engram_dir':str(self.source)},self.config)

    def test_pinned_huggingface_snapshot_is_detected(self):
        cache=self.root/'hf';snapshot=cache/'hub/models--Example--Original/snapshots'/('a'*40)
        shutil.copytree(self.source,snapshot)
        request={k:v for k,v in self.request.items() if k!='engram_source'}
        with mock.patch.dict(os.environ,{'HF_HOME':str(cache)}):
            self.assertEqual(inspect_engram(request,self.config)['path'],str(snapshot))

    def test_packed_tables_require_matching_rank_geometry_and_complete_payload(self):
        packed=self.root/'packed';packed.mkdir();path=packed/'engram-l1-r0of4.bin'
        path.write_bytes(struct.pack('<6Q',0x31344E4531565344,1,0,4,4,264)+bytes(4096-48+4*264))
        request={**self.request,'engram_source':None,'engram_dir':str(packed)}
        self.assertEqual(inspect_engram(request,self.config)['kind'],'packed')
        path.write_bytes(struct.pack('<6Q',0x31344E4531565344,1,0,4,5,264)+bytes(4096-48+4*264))
        with self.assertRaisesRegex(ValueError,'model and rank'):
            inspect_engram(request,self.config)


if __name__=='__main__':unittest.main()
