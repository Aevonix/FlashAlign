# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Pinned SGLang cache-layout regression; requires the engine dependencies.

Run with FLASHALIGN_REAL_SGLANG_TESTS=1 in the prepared SGLang environment.
The real cache/node classes are used without allocating model or KV tensors.
"""
from array import array
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get('FLASHALIGN_REAL_SGLANG_TESTS') == '1',
                     'requires the pinned SGLang runtime environment')
class RealCacheLifecycleTests(unittest.TestCase):
    def setUp(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore, UnifiedTreeNode
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
        from flashalign_sglang.gate import Gate, Selection
        from flashalign_sglang import lifecycle_worker
        self.worker = lifecycle_worker
        self.old, self.new = 'g-old', 'g-new'
        profiles = {self.old:SimpleNamespace(digest='a'*64), self.new:SimpleNamespace(digest='b'*64)}
        gate = Gate(profiles, b'k'*32)
        self.old_salt = gate.cache_salt(Selection(self.old, 'a'*64, 1.0), None)
        self.new_salt = gate.cache_salt(Selection(self.new, 'b'*64, 1.0), None)
        self.decode = patch.object(lifecycle_worker, '_selection', side_effect=gate.from_salt)
        self.decode.start()
        self.addCleanup(self.decode.stop)
        self.Key, self.TreeNode, self.RadixCache = RadixKey, TreeNode, RadixCache
        # Construction without allocators isolates the exact facade/core/node
        # layout responsible for reference tracking from model execution.
        self.unified = UnifiedRadixCache.__new__(UnifiedRadixCache)
        self.unified.tree_core = UnifiedTreeCore.__new__(UnifiedTreeCore)
        self.unified.tree_core.root_node = UnifiedTreeNode((ComponentType.FULL,))
        self.make_unified_node = lambda: UnifiedTreeNode((ComponentType.FULL,))

    def test_unified_cache_retains_each_generation_namespace(self):
        old, new = self.make_unified_node(), self.make_unified_node()
        old.key = self.Key(array('q',[1,2,3]), cache_salt=self.old_salt)
        new.key = self.Key(array('q',[4,5,6]), cache_salt=self.new_salt)
        self.unified.tree_core.root_node.children['old'] = old
        self.unified.tree_core.root_node.children['new'] = new
        self.assertEqual(self.worker.cache_references(self.unified,self.old),1)
        self.assertEqual(self.worker.cache_references(self.unified,self.new),1)
        self.unified.tree_core.root_node.children.clear()
        self.assertEqual(self.worker.cache_references(self.unified,self.old),0)

    def test_legacy_cache_layout_still_tracks_namespaces(self):
        cache = self.RadixCache.__new__(self.RadixCache)
        cache.root_node = self.TreeNode()
        child = self.TreeNode()
        child.key = self.Key(array('q',[1,2,3]), cache_salt=self.old_salt)
        cache.root_node.children['old'] = child
        self.assertEqual(self.worker.cache_references(cache,self.old),1)


@unittest.skipUnless(os.environ.get('FLASHALIGN_REAL_SGLANG_TESTS') == '1',
                     'requires the pinned SGLang runtime environment')
class RealServerArgsGraphTests(unittest.TestCase):
    def test_real_raw_server_args_and_graph_normalization(self):
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.arg_groups.cuda_graph_hook import parse_cuda_graph_config
        from sglang.srt.arg_groups.overrides import resolution_result
        from flashalign_sglang.graph_capability import check_graph_capabilities
        args=prepare_server_args(['--model-path','fixture-local-model','--cuda-graph-backend-decode','full',
                                 '--cuda-graph-backend-prefill','breakable'])
        self.assertIsNone(args.cuda_graph_config)
        self.assertEqual(check_graph_capabilities(args,{}),{'decode':'full','prefill':'breakable'})
        parse_cuda_graph_config(args)
        normalized=resolution_result(args,'cuda_graph_config')
        self.assertEqual(normalized.decode.backend,'full')
        self.assertEqual(normalized.prefill.backend,'breakable')
        self.assertEqual(check_graph_capabilities(args,{},config=normalized),{'decode':'full','prefill':'breakable'})


@unittest.skipUnless(os.environ.get('FLASHALIGN_REAL_SGLANG_TESTS') == '1',
                     'requires the pinned prepared SGLang runtime environment')
class RealSchedulerDispatchTests(unittest.TestCase):
    def test_enabled_cache_reporting_keeps_measured_zero_distinct_from_disabled(self):
        from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
        response={'meta_info':{'prompt_tokens':10,'completion_tokens':2,'cached_tokens':0}}
        zero=UsageProcessor.calculate_response_usage([response],enable_cache_report=True)
        self.assertEqual(zero.model_dump()['prompt_tokens_details']['cached_tokens'],0)
        disabled=UsageProcessor.calculate_response_usage([response],enable_cache_report=False)
        self.assertIsNone(disabled.prompt_tokens_details)
        response['meta_info']['cached_tokens']=7
        hit=UsageProcessor.calculate_response_usage([response],enable_cache_report=True)
        self.assertEqual(hit.prompt_tokens_details.cached_tokens,7)
        streaming=UsageProcessor.calculate_streaming_usage({0:10},{0:0},{0:2},{0:0},1,enable_cache_report=True)
        self.assertEqual(streaming.prompt_tokens_details.cached_tokens,0)

    def test_actual_single_and_batch_dispatcher_share_the_admission_fence(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput, TokenizedGenerateReqInput
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.flashalign import lifecycle_worker
        class Unused:
            def __getattr__(self,name):return self
            def __call__(self,*args,**kwargs):raise AssertionError('Unrelated handler was invoked')
        class Harness(Scheduler):
            def __getattr__(self,name):return Unused()
        scheduler=Harness.__new__(Harness)
        accepted=[]
        scheduler.handle_generate_request=accepted.append
        scheduler.handle_batch_generate_request=Unused()
        scheduler._flashalign_fenced=True
        scheduler._flashalign_deferred=[]
        scheduler._flashalign_deferred_aborts=[]
        scheduler.init_request_dispatcher()
        def request(rid):
            return TokenizedGenerateReqInput(rid=rid,input_text='',input_ids=array('q',[1]),input_embeds=None,
                mm_inputs=None,token_type_ids=None,sampling_params=SamplingParams(),return_logprob=False,
                logprob_start_len=0,top_logprobs_num=0,token_ids_logprob=None,stream=False,http_worker_ipc='route-'+rid)
        first,second,third=request('single'),request('batch-1'),request('batch-2')
        with patch.object(lifecycle_worker,'_selection',return_value=None):
            scheduler._request_dispatcher(first)
            scheduler._request_dispatcher(BatchTokenizedGenerateReqInput(batch=[second,third]))
            self.assertEqual(scheduler._flashalign_deferred,[first,second,third])
            self.assertFalse(accepted)
            scheduler._flashalign_fenced=False
            for item in scheduler._flashalign_deferred:lifecycle_worker.dispatch_generate(scheduler,item)
        self.assertEqual(accepted,[first,second,third])
        self.assertEqual([item.rid for item in accepted],['single','batch-1','batch-2'])
        self.assertEqual([item.http_worker_ipc for item in accepted],['route-single','route-batch-1','route-batch-2'])

    def test_graph_buffers_use_early_model_config_dtype(self):
        import tempfile
        import torch
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.configs.model_config import ModelConfig
        from flashalign_sglang.graph_adapter import GraphAdapter
        from flashalign_sglang.runtime import Profile
        from flashalign_sglang.gate import Gate
        runner=ModelRunner.__new__(ModelRunner)
        runner.device='cpu'
        runner.model_config=ModelConfig.__new__(ModelConfig)
        runner.model_config.dtype=torch.bfloat16
        runner.model=torch.nn.Module()
        runner.model.model=torch.nn.Module()
        runner.model.model.layers=torch.nn.ModuleList([torch.nn.Identity()])
        runner.model.model.embed_tokens=torch.nn.Embedding(2,4,dtype=torch.bfloat16)
        self.assertFalse(hasattr(runner,'dtype'))
        site=(0,'attn_out')
        profiles={'fixture':Profile('fixture','a'*64,{},4,{site:{'q':torch.tensor([[1.,0.,0.,0.]]),'scale':1.}})}
        gate=Gate(profiles,b'k'*32)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'FLASHALIGN_GRAPH_ATTEST':'1',
                'FLASHALIGN_GRAPH_ATTEST_FILE':tmp+'/evidence.jsonl','FLASHALIGN_GRAPH_MAX_ROWS':'16'}):
            graph=GraphAdapter(runner,profiles,gate,{'writer':site})
        self.assertEqual(graph.before[site].dtype,torch.bfloat16)
        self.assertEqual(graph.after[site].dtype,torch.bfloat16)
        self.assertEqual(graph.banks[site][0].dtype,torch.float32)


if __name__ == '__main__':
    unittest.main()
