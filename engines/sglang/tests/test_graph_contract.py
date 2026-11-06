# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CPU graph-generation contracts; real CUDA execution is a separate gate."""
import importlib
import sys
from contextvars import ContextVar
from types import ModuleType, SimpleNamespace
import unittest
import torch
from unittest.mock import patch

from flashalign_sglang.row_contract import (ProfileIdentity, Selection, SlotRegistry,
    compile_plan, graph_batch_reason, graph_memory_bytes, graph_profile_reason)
from flashalign_sglang.graph_capability import check_graph_capabilities


class GraphContractTests(unittest.TestCase):
    def test_raw_graph_flags_are_accepted_before_normalization(self):
        args=SimpleNamespace(cuda_graph_config=None,cuda_graph_backend_decode='full',cuda_graph_backend_prefill='breakable')
        self.assertEqual(check_graph_capabilities(args,{}),{'decode':'full','prefill':'breakable'})
        args.cuda_graph_config={'decode':{'backend':'disabled'},'prefill':{'backend':'disabled'}}
        with self.assertRaises(ValueError):check_graph_capabilities(args,{})

    def test_mixed_generations_force_eager_before_any_writer(self):
        fast = ProfileIdentity('g-first', 'first-digest')
        selectors = [None, Selection(fast.name, fast.digest, 1), Selection('g-rank128', 'second', 1)]
        self.assertEqual(graph_batch_reason(selectors, {fast}, rows=3, capacity=64), 'generation_requires_eager')
        self.assertIsNone(graph_batch_reason(selectors[:2], {fast}, rows=2, capacity=64))

    def test_new_sites_and_higher_rank_keep_eager_support(self):
        profile = lambda rank, sites: SimpleNamespace(hidden=4, sites={site:{'q':SimpleNamespace(shape=(rank,4))} for site in sites})
        self.assertIsNone(graph_profile_reason(profile(1, ['attn']), {'attn'}))
        self.assertEqual(graph_profile_reason(profile(128, ['attn']), {'attn'}), 'rank_not_supported_by_fused_kernel')
        self.assertEqual(graph_profile_reason(profile(1, ['attn','new']), {'attn'}), 'uncaptured_sites')

    def test_slot_reuse_requires_zero_refs_and_completion(self):
        old, new = ProfileIdentity('g-old','digest-old'), ProfileIdentity('g-new','digest-new')
        slots = SlotRegistry([old], capacity=1)
        self.assertIsNone(slots.reserve(new))
        for count, completed in ((1,True),(0,False)):
            with self.assertRaises(ValueError): slots.release(old,references=count,completed=completed)
        self.assertEqual(slots.slot_for(Selection(old.name,old.digest,1))[0],1)
        slots.release(old,references=0,completed=True)
        self.assertEqual(slots.reserve(new),1)
        with self.assertRaises(ValueError): slots.slot_for(Selection(old.name,old.digest,1))

    def test_interleaved_generation_handles_never_rebind(self):
        a,b,c = (ProfileIdentity('g-'+name,name*64) for name in ('a','b','c'))
        slots=SlotRegistry([a,b],capacity=3)
        selected=[Selection(a.name,a.digest,0.5),Selection(b.name,b.digest,1),None]
        before=compile_plan(slots,selections=selected,lengths=[2,1,1],rows=4,capacity=8)
        slots.reserve(c)
        after=compile_plan(slots,selections=selected,lengths=[2,1,1],rows=4,capacity=8)
        self.assertEqual(before,after)
        with self.assertRaises(ValueError): slots.reserve(ProfileIdentity(a.name,'mutated'))

    def test_row_and_memory_budgets_fail_before_allocation(self):
        expected=(8+1)*(4096+1)*4+65536*8
        self.assertEqual(graph_memory_bytes([4096],slots=8,rows=65536,budget=expected),expected)
        with self.assertRaises(ValueError): graph_memory_bytes([4096],slots=8,rows=65536,budget=expected-1)
        with self.assertRaises(ValueError): graph_memory_bytes([4096],slots=257,rows=8,budget=10**9)
        self.assertEqual(graph_batch_reason([None],set(),rows=65,capacity=64),'row_capacity')

    def test_actual_submission_marks_mixed_batch_before_projection(self):
        # Import the real adapter without compiling its CUDA-only kernel.
        stub=ModuleType('flashalign_sglang.fused_projection')
        stub.project_inplace=lambda *args: (_ for _ in ()).throw(AssertionError('Fused path reached'))
        with patch.dict(sys.modules,{'flashalign_sglang.fused_projection':stub}):
            module=importlib.import_module('flashalign_sglang.graph_adapter')
        graph=module.GraphAdapter.__new__(module.GraphAdapter)
        graph.eligible={ProfileIdentity('g-fast','one')}
        graph.gate=SimpleNamespace(from_salt=lambda x:x)
        graph.state=SimpleNamespace(capacity=64)
        graph._submission=ContextVar('test_submission',default=None)
        graph.graph_batches=graph.eager_batches=0
        graph.last_mode=None
        graph.scope=SimpleNamespace(active=lambda:True)
        graph.profiles={}
        batch=SimpleNamespace(input_ids=SimpleNamespace(numel=lambda:2),
            flashalign_cache_salts=[Selection('g-fast','one',1),Selection('g-unsupported','two',1)])
        with patch('flashalign_sglang.runtime.project',return_value='eager-output') as project:
            with graph.submission(batch) as observed:
                self.assertTrue(observed.force_eager)
                self.assertEqual(graph.project('input','new-site'),'eager-output')
                project.assert_called_once_with('input',batch,'new-site',graph.profiles,graph.gate)
        self.assertEqual(graph.last_mode,'eager')
        self.assertEqual(graph.eager_batches,1)

    def test_new_uncaptured_hook_is_safe_for_old_new_and_rollback_eager_batches(self):
        stub=ModuleType('flashalign_sglang.fused_projection')
        stub.project_inplace=lambda *args: (_ for _ in ()).throw(AssertionError('Unknown site reached fused kernel'))
        with patch.dict(sys.modules,{'flashalign_sglang.fused_projection':stub}):
            module=importlib.import_module('flashalign_sglang.graph_adapter')
        from flashalign_sglang.gate import Gate, Selection as GateSelection
        from flashalign_sglang.runtime import Profile
        q=torch.tensor([[1.,0.,0.,0.]])
        site=(9,'mlp_out')
        profiles={'g-old':Profile('old','a'*64,{},4,{(0,'attn_out'):{'q':q,'scale':1.0}}),
                  'g-new':Profile('new','b'*64,{},4,{site:{'q':q,'scale':1.0}})}
        gate=Gate(profiles,b'k'*32)
        graph=module.GraphAdapter.__new__(module.GraphAdapter)
        graph.scope=SimpleNamespace(active=lambda:True)
        graph.profiles,graph.gate,graph.banks=profiles,gate,{}
        graph._submission=ContextVar('uncaptured-site',default=None)
        mode=SimpleNamespace(is_target_verify=lambda:False,is_decode=lambda:True,is_extend=lambda:False,is_idle=lambda:False)
        old=gate.cache_salt(GateSelection('g-old','a'*64,1.0),None)
        new=gate.cache_salt(GateSelection('g-new','b'*64,1.0),None)
        values=torch.ones(2,4)
        for salts,expected in (([old,old],values),([old,new],torch.tensor([[1.,1.,1.,1.],[0.,1.,1.,1.]])),([old,old],values)):
            batch=SimpleNamespace(flashalign_cache_salts=salts,forward_mode=mode)
            token=graph._submission.set((batch,False))
            try:
                torch.testing.assert_close(graph.project(values,site),expected)
            finally:graph._submission.reset(token)

    def test_model_config_dtype_is_required_without_late_runner_dtype(self):
        stub=ModuleType('flashalign_sglang.fused_projection')
        stub.project_inplace=lambda *args:None
        with patch.dict(sys.modules,{'flashalign_sglang.fused_projection':stub}):
            module=importlib.import_module('flashalign_sglang.graph_adapter')
        runner=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.int8))
        with self.assertRaisesRegex(ValueError,'activation dtype'):
            module.GraphAdapter(runner,{},None,{})


if __name__=='__main__':
    unittest.main()
