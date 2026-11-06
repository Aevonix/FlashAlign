# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class NativeControlBundleBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir()
        (self.source/'tools').mkdir()
        self.bridge=b"SCHEMA = 'flashalign.tensorfold-control.v2'\n"
        (self.source/'tools/flashalign_admin.py').write_bytes(self.bridge)
        (self.source/'LICENSE').write_text('Source fixture license\n')
        for command in (['init','-q'],['add','.'],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']):
            subprocess.run(['git','-C',str(self.source),*command],check=True,capture_output=True)
        self.revision=subprocess.check_output(['git','-C',str(self.source),'rev-parse','HEAD'],text=True).strip()
        self.binary=self.root/'native';self.binary.write_bytes(b'fixture native binary')
        self.grammar=self.root/'grammar';self.grammar.write_bytes(b'fixture grammar library')
        self.notice=self.root/'notice';self.notice.write_text('Fixture library license\n')
        self.caps={'durable_owner_replay':True,'model_types':['qwen3_5'],'transforms':['projection'],
                   'tensor_parallel_sizes':[1],'rank_transport':'cli','speculative_methods':['dflash'],
                   'profile_control':{'schema':'flashalign.native-profile-control.v1',
                    'protocol':'flashalign.tensorfold-control.v2','bridge':'bin/flashalign_admin.py'}}

    def build(self):
        caps=self.root/'caps.json';caps.write_text(json.dumps(self.caps))
        script=Path(__file__).resolve().parents[2]/'tools/build-native-bundle.py'
        return subprocess.run([sys.executable,str(script),'--source-tree',str(self.source),
            '--source-revision',self.revision,'--binary',str(self.binary),'--capabilities',str(caps),
            '--platform','linux-x86_64','--cuda-architecture','sm_120',
            '--library','libtfgrammar.so='+str(self.grammar),'--library-notice','libtfgrammar.so='+str(self.notice),
            '--output',str(self.root/'bundle')],capture_output=True,text=True)

    def test_bridge_is_copied_from_commit_and_bound_to_inventory(self):
        (self.source/'tools/flashalign_admin.py').write_text('raise RuntimeError("working tree is not a release input")\n')
        result=self.build();self.assertEqual(result.returncode,0,result.stderr)
        path=self.root/'bundle/bin/flashalign_admin.py';self.assertEqual(path.read_bytes(),self.bridge)
        manifest=json.loads((self.root/'bundle/manifest.json').read_text())
        self.assertEqual(manifest['files']['bin/flashalign_admin.py']['sha256'],hashlib.sha256(self.bridge).hexdigest())
        self.assertEqual(manifest['source_revision'],self.revision)

    def test_bridge_protocol_mismatch_prevents_a_finished_bundle(self):
        self.caps['profile_control']['protocol']='flashalign.tensorfold-control.v1'
        result=self.build();self.assertNotEqual(result.returncode,0)
        self.assertIn('protocol differs',result.stderr)
        self.assertFalse((self.root/'bundle/manifest.json').exists())
