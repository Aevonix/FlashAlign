# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from kernel_schedule import apply_schedule_patch

def original_bytes():
    return (ROOT/'fixtures/tilelang_kernel.pinned.txt').read_bytes()

def patched_text():
    return apply_schedule_patch(original_bytes()).decode()


class Tensor:
    def __init__(self, shape, dtype='bf16'):
        self.shape = shape
        self.dtype = dtype
        self.device = 'cuda:0'
    def dim(self):
        return len(self.shape)
    def unsqueeze(self, _):
        return self


class Schedule(unittest.TestCase):
    def invoke(self, capability=(12,1), heads=16, tail=0, dtype='bf16'):
        tree = ast.parse(patched_text())
        wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == 'tilelang_sparse_fwd')
        factory1 = Mock(return_value=lambda *args: 'result')
        factory2 = Mock(return_value=lambda *args: 'result')
        namespace = {'torch': SimpleNamespace(Tensor=Tensor,bfloat16='bf16',
                       cuda=SimpleNamespace(get_device_capability=lambda _: capability)),
                     '_is_hip':False, 'sparse_attention_fwd_kernel_v1':factory1,
                     'sparse_attention_fwd_kernel_v2':factory2}
        exec(compile(ast.Module(body=[wrapper],type_ignores=[]), 'wrapper', 'exec',
                     dont_inherit=True), namespace)
        result = namespace['tilelang_sparse_fwd'](Tensor((4,heads,512+tail),dtype),
            Tensor((2176,1,512+tail),dtype),Tensor((4,1,2112),'int32'),0.125)
        self.assertEqual(result,'result')
        return factory1,factory2

    def test_sm121_glm_lane_selects_measured_schedule(self):
        first, second = self.invoke()
        first.assert_called_once_with(16,512,0,2112,sm_scale=0.125,block_I=64,num_stages=1)
        second.assert_not_called()

    def test_sm120_schedule_matches_measured(self):
        first,_ = self.invoke(capability=(12,0))
        self.assertEqual(first.call_args.kwargs,{'sm_scale':0.125,'block_I':64,'num_stages':1})

    def test_sm90_schedule_is_unchanged(self):
        first,_ = self.invoke(capability=(9,0))
        self.assertEqual(first.call_args.kwargs,{'sm_scale':0.125})

    def test_other_head_counts_are_unchanged(self):
        first,_ = self.invoke(heads=32)
        self.assertEqual(first.call_args.kwargs,{'sm_scale':0.125})

    def test_non_bf16_schedule_is_unchanged(self):
        first,_ = self.invoke(dtype='fp8')
        self.assertEqual(first.call_args.kwargs,{'sm_scale':0.125})

    def test_rope_factory_is_unchanged(self):
        first,second = self.invoke(tail=64)
        first.assert_not_called()
        second.assert_called_once_with(16,512,64,2112,sm_scale=0.125)

    def test_changed_upstream_preimage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'preimage'):
            apply_schedule_patch(original_bytes() + b'\n')

    def test_all_kernel_formulas_are_byte_identical_ast(self):
        original=ast.parse(original_bytes().decode())
        patched=ast.parse(patched_text())
        select=lambda tree: {n.name:ast.dump(n) for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='tilelang_sparse_fwd'}
        self.assertEqual(select(original),select(patched))


if __name__ == '__main__':
    unittest.main()
