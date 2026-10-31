# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Generic weighted-profile and bundled-draft setup rules; no model or GPU."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_setup import make_profile, write_tensors
from owner_mode.backend_setup import _bundled_draft_path, _select_kit, prepare_backend
from owner_mode.profiles import validate_profile, ProfileError
from owner_mode.quickstart import make_default_plan
from owner_mode.setup_runtime import Console, SetupError


def weighted(root):
    path=make_profile(root,model='example/model',transform='weighted_output_projection_v1')
    data=json.loads((path/'profile.json').read_text())
    data['requires']=['per_direction_strength']
    data['sites'][0].update(rank=2,scale=1.,tensor='layers.0.out')
    (path/'profile.json').write_text(json.dumps(data))
    write_tensors(path,{'L0.attn_out.Q':([2,2],[1.,0.,0.,1.]),
                        'L0.attn_out.strengths':([2],[1.,2.5])})
    return path


class WeightedSchema(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.profile=weighted(self.root/'profile')
    def test_weighted_profile_preserves_distinct_strengths(self):
        result=validate_profile(self.profile)
        self.assertEqual(result['requires'],['per_direction_strength'])
    def test_weighted_profile_requires_its_capability(self):
        p=self.profile/'profile.json';d=json.loads(p.read_text());d['requires']=[];p.write_text(json.dumps(d))
        with self.assertRaises(ProfileError):validate_profile(self.profile)
    def test_weighted_profile_requires_positive_direction_strengths(self):
        write_tensors(self.profile,{'L0.attn_out.Q':([2,2],[1.,0.,0.,1.]),
                        'L0.attn_out.strengths':([2],[1.,0.])})
        with self.assertRaises(ProfileError):validate_profile(self.profile)
    def test_weighted_profile_requires_unit_site_scale(self):
        p=self.profile/'profile.json';d=json.loads(p.read_text());d['sites'][0]['scale']=2;p.write_text(json.dumps(d))
        with self.assertRaises(ProfileError):validate_profile(self.profile)
    def test_valid_schema_does_not_enable_an_unsupported_backend(self):
        with self.assertRaises(SetupError):
            _select_kit('vllm',{'model_type':'qwen3_5'},validate_profile(self.profile))


class BundledDraft(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.model=self.root/'model';self.model.mkdir();(self.model/'draft').mkdir()
    def test_existing_internal_subdirectory_is_resolved(self):
        self.assertEqual(_bundled_draft_path(self.model,{'default_draft_subdirectory':'draft'}),self.model/'draft')
    def test_dot_traversal_and_both_separators_are_rejected(self):
        for name in ('.','..','a/b','a'+chr(92)+'b',chr(0)):
            with self.subTest(name=repr(name)),self.assertRaises(SetupError):
                _bundled_draft_path(self.model,{'default_draft_subdirectory':name})
    def test_symlink_escape_is_rejected(self):
        other=self.root/'other';other.mkdir();(self.model/'escape').symlink_to(other,target_is_directory=True)
        with self.assertRaises(SetupError):_bundled_draft_path(self.model,{'default_draft_subdirectory':'escape'})
    def test_default_setup_cannot_supply_private_opt_in(self):
        (self.model/'config.json').write_text(json.dumps({'model_type':'qwen3_5','hidden_size':2}))
        profile=weighted(self.root/'profile')
        args=SimpleNamespace(home=self.root/'home',engine_only=True,model_path=self.model,profile=profile,
                             engine_bundle=None,backend='sglang',devices='0',profiles=[],engine_python=None,
                             tools_default='off',drafter=None)
        console=Console(output=io.StringIO(), color=False)
        with patch('owner_mode.backend_setup._select_kit',return_value=(self.root,{'requires_private_opt_in':True})), \
             patch('owner_mode.backend_setup.active_speculation',return_value=None), \
             patch('builtins.input',side_effect=AssertionError('Must not prompt')):
            with self.assertRaisesRegex(SetupError,'explicitly supplied setup plan'):
                make_default_plan(args,{},console)
        self.assertFalse(args.home.exists())
    def test_stateful_launcher_flag_obeys_explicit_private_policy(self):
        (self.model/'config.json').write_text(json.dumps({'model_type':'qwen3_5','hidden_size':2}))
        profile=weighted(self.root/'profile');kit=self.root/'kit';kit.mkdir()
        (kit/'launch.py').write_text('# fixture only\n');(kit/'requirements.txt').write_text('')
        base={'model_types':['qwen3_5'],'transforms':['weighted_output_projection_v1'],'sites':['attn_out'],
              'site_tensor_patterns':{'attn_out':r'layers\.{layer}\.out'},'tensor_parallel_sizes':[1],
              'speculative_algorithms':['EAGLE'],'native_speculative_algorithms':['EAGLE'],
              'managed_stateful_launcher':True,'default_arguments':['--context-length','65536'],
              'status':'candidate'}
        for private in (False,True):
            metadata={**base,'requires_private_opt_in':private};(kit/'backend.json').write_text(json.dumps(metadata))
            spec={'kind':'sglang','model_path':str(self.model),'devices':['0'],'python':sys.executable,
                  'draft_algorithm':'EAGLE','allow_research':private}
            with patch('owner_mode.backend_setup._select_kit',return_value=(kit,metadata)), \
                 patch('owner_mode.backend_setup._prepare_runtime',return_value={'python':sys.executable,'source':str(self.model)}):
                result=prepare_backend(self.root/str(private),spec,profile,model_id='example/model')
            self.assertEqual('--allow-unqualified-candidate' in result['commands'][0]['argv'],private)

    def default_plan(self, metadata, active=None):
        (self.model/'config.json').write_text(json.dumps({'model_type':'example','hidden_size':2,
                                                        'architectures':['NativeTarget']}))
        (self.model/'draft/config.json').write_text(json.dumps({'architectures':['DFlashDraftModel']}))
        profile=weighted(self.root/'profile')
        args=SimpleNamespace(home=self.root/'home',engine_only=True,model_path=self.model,profile=profile,
                             engine_bundle=None,backend='sglang',devices='0',profiles=[],engine_python=None,
                             tools_default='off',drafter=None)
        console=Console(output=io.StringIO(), color=False)
        # This metadata describes a synthetic qualified lane. It does not bypass
        # the private opt-in of any real engine or contact a model.
        with patch('owner_mode.backend_setup._select_kit',return_value=(self.root,metadata)), \
             patch('owner_mode.backend_setup.active_speculation',return_value=active), \
             patch('builtins.input',side_effect=AssertionError('Must not prompt')):
            return json.loads(make_default_plan(args,{},console).read_text())

    def test_declared_active_native_algorithms_keep_same_model(self):
        metadata={'speculative_algorithms':['DFLASH','EAGLE','NEXTN'],
                  'native_speculative_algorithms':['EAGLE','NEXTN'],
                  'default_draft_subdirectory':'draft','default_arguments':['--context-length','65536']}
        for method in ('EAGLE','NEXTN'):
            with self.subTest(method=method):
                active={'kind':'sglang','draft_model':str(self.model),'draft_algorithm':method,
                        'speculative_arguments':['--speculative-num-draft-tokens','4']}
                plan=self.default_plan(metadata,active)
                self.assertEqual(plan['engine']['draft_model'],str(self.model))
                self.assertEqual(plan['engine']['draft_algorithm'],method)
                self.assertEqual(plan['engine']['speculative_arguments'],active['speculative_arguments'])

    def test_declared_native_default_precedes_bundled_draft_assignment(self):
        metadata={'speculative_algorithms':['EAGLE','DFLASH'],
                  'native_speculative_algorithms':['EAGLE'],
                  'default_draft_subdirectory':'draft','default_arguments':['--context-length','65536']}
        plan=self.default_plan(metadata)
        self.assertEqual(plan['engine']['draft_model'],str(self.model))
        self.assertEqual(plan['engine']['draft_algorithm'],'EAGLE')

    def test_external_default_still_uses_bundled_draft(self):
        metadata={'speculative_algorithms':['DFLASH','EAGLE'],
                  'native_speculative_algorithms':['EAGLE'],
                  'default_draft_subdirectory':'draft','default_arguments':['--context-length','65536']}
        plan=self.default_plan(metadata)
        self.assertEqual(plan['engine']['draft_model'],str(self.model/'draft'))


if __name__=='__main__':unittest.main()
