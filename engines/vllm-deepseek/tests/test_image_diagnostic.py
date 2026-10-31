# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import os
import hashlib
from unittest.mock import patch
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'flashalign_deepseek/plugin.py'

class Value:
    def __init__(self,label):self.label=label;self.shape=(3,2);self.device='cpu'
    def numel(self):return 3
    def __getitem__(self,span):return self
    def to(self,device):return self

class Counts(list):
    def __add__(self, value):return Counts([item+value for item in self])

class NoSnapshot(dict):
    def keys(self):raise AssertionError('diagnostics-off counter snapshot')
    def __iter__(self):raise AssertionError('diagnostics-off counter snapshot')

class Diagnostic(unittest.TestCase):
    def fixture(self,selected=True,recording=True,counts=None):
        node=next(n for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='gather')
        node.decorator_list=[]
        tree=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(tree)
        cached=Value('cache');out=Value('view');changed=Value('changed');profile=NS(applied_rows={} if counts is None else counts)
        selection=NS(name='profile',scale=1.) if selected else None
        runner=NS(_flashalign_selections={'request':selection} if selected else {},_flashalign_profiles={'profile':profile},
            _flashalign_mm_types={'feature':Value('types')},model=NS(aligner=NS(w2=NS(bias=None))))
        pos=NS(offset=0,length=3,is_embed=None,get_embeds_indices_in_range=lambda a,b:(a,b))
        encoder=NS(_flashalign_runner=runner,encoder_cache=NS(mm_features={'request':[NS(mm_position=pos,identifier='feature')]},encoder_outputs={'feature':cached}))
        rows=[];calls=[]
        def project(output,types,p,sel,bias):
            calls.append('project')
            for key,value in [('0.vision_aligner_out',2),('0.image_marker_out',1)]:p.applied_rows[key]=p.applied_rows.get(key,0)+value
            return changed
        def digest(value):
            if not recording:raise AssertionError('diagnostics-off tensor digest')
            return value.label
        scope={'original_gather':lambda *a:([out],False),'get_mm_features_in_window':lambda *a,**k:(0,1),
            'os':os,'tensor_digest':digest,'image_project':project,'diagnostic':rows.append,'hashlib':hashlib}
        exec(compile(tree,str(SOURCE),'exec'),scope)
        with patch.dict(os.environ,{'FLASHALIGN_DEEPSEEK_DIAGNOSTIC':'record' if recording else ''}):
            result=scope['gather'](encoder,['request'],3,[3],[0],[3],Counts([0]))
        return rows,calls,profile,result

    def test_selected_site_deltas_are_per_gather_not_cumulative(self):
        rows,calls,profile,_=self.fixture(counts={'0.vision_aligner_out':11,'0.image_marker_out':7,'5.attn_out':900})
        self.assertEqual(calls,['project'])
        self.assertEqual(rows[0]['applied_rows'],{'0.vision_aligner_out':2,'0.image_marker_out':1})
        self.assertTrue(rows[0]['cache_unchanged'])

    def test_off_image_reports_no_applied_sites_or_transform(self):
        rows,calls,_,_=self.fixture(selected=False)
        self.assertEqual(calls,[]);self.assertEqual(rows[0]['applied_rows'],{})
        self.assertEqual(rows[0]['profile'],'off')
        self.assertEqual(rows[0]['input_view_sha256'],rows[0]['output_view_sha256'])

    def test_diagnostic_off_does_not_snapshot_or_hash_and_keeps_transform(self):
        rows,calls,profile,_=self.fixture(recording=False,counts=NoSnapshot())
        self.assertEqual(rows,[]);self.assertEqual(calls,['project'])
        self.assertEqual(profile.applied_rows['0.vision_aligner_out'],2)

    def test_off_without_diagnostics_uses_original_fast_path(self):
        rows,calls,_,_=self.fixture(selected=False,recording=False)
        self.assertEqual(rows,[]);self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
