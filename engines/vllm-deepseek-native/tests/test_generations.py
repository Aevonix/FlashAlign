# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Native generation ownership checks; no CUDA qualification is implied."""
from dataclasses import dataclass
from types import SimpleNamespace
from types import ModuleType
from pathlib import Path
import ast
import hashlib
import json
import sys

import pytest

from flashalign_native.generations import GenerationRuntime, Snapshot
from flashalign_native.protocol import Selection
from flashalign_native_wo.execution import install_native_weights, run_native_generations


def test_snapshot_identity_covers_generations_and_different_same_alias_profiles():
    a = Snapshot((('g-one', 'a'*64),))
    b = Snapshot((('g-two', 'a'*64),))
    c = Snapshot((('g-one', 'b'*64),))
    assert len({a.digest, b.digest, c.digest}) == 3
    with pytest.raises(Exception): a.slot('g-two')


def test_native_runtime_keeps_old_and_new_selections_and_layer_bindings():
    old = SimpleNamespace(name='owner', digest='a'*64, is_native_wo=True,
                          spec=SimpleNamespace(layers=(0,)))
    new = SimpleNamespace(name='owner', digest='b'*64, is_native_wo=True,
                          spec=SimpleNamespace(layers=(0, 1)))
    handles = {'g-one': SimpleNamespace(profile=old), 'g-two': SimpleNamespace(profile=new)}
    def resolve(gid, digest):
        selected = handles[gid]
        if selected.profile.digest != digest: raise ValueError('wrong digest')
        return selected
    registry = SimpleNamespace(resolve=resolve)
    original = SimpleNamespace(profile=old, selections={
        'old': Selection('owner', old.digest, 1., 'g-one'),
        'new': Selection('owner', new.digest, 1., 'g-two')})
    runtime = GenerationRuntime(original, registry)
    runtime.handles = handles
    runtime.active_snapshot = Snapshot((('g-one', old.digest), ('g-two', new.digest)))
    assert runtime.classify_scheduled({'old':2,'new':3}) == 1
    assert runtime.classify_scheduled({'old':2,'new':3,'off':1}) == 2
    assert runtime.classify_scheduled({'off':1}) == 0
    assert runtime._slot(original.selections['old']) == 1
    assert runtime._slot(original.selections['new']) == 2
    layer = SimpleNamespace(layer_id=0, _flashalign_native_generations={
        'g-one':(None,None,'old-matrix',None),'g-two':(None,None,'new-matrix',None)})
    assert runtime.bindings(layer) == ((1,'old-matrix'),(2,'new-matrix'))
    layer.layer_id = 1
    assert runtime.bindings(layer) == ((2,'new-matrix'),)
    with pytest.raises(ValueError): runtime._slot(Selection('owner',new.digest,1.1,'g-two'))
    with pytest.raises(ValueError): runtime._slot(Selection('owner',old.digest,1.,'g-two'))


@dataclass(frozen=True)
class Weights:
    wo_a: object
    wo_b: object


@dataclass(frozen=True)
class Binding:
    weights: Weights
    tmp: object
    output: list


def test_multiple_native_matrices_modify_only_their_own_rows_and_preserve_stock():
    shared = object()
    stock = Weights(shared,'stock');old = Weights(shared,'old');new = Weights(shared,'new')
    binding = Binding(stock, 'shared-A', [b'off-nan-bits',b'stock-old',b'stock-new',b'off-minus-zero'])
    calls=[]
    class State:
        def run_inv_rope(self, value, *, stream):
            calls.append(('stock',value.weights.wo_b))
            return value.output
    class Operations:
        def mask_tmp(self, source, owners, indices, *, slot):
            calls.append(('mask',slot))
            return tuple(value == slot for value in owners)
        def run_b(self, state, bound, alternate, mask, stream):
            calls.append(('matrix',alternate.wo_b))
            return [alternate.wo_b.encode() if value else b'zero' for value in mask]
        def scatter_owner(self, output, edited, owners, indices, *, slot):
            for index,value in enumerate(owners):
                if value == slot: output[index]=edited[index]
    result=run_native_generations(binding,State(),((1,old),(2,new)),2,(0,1,2,0),None,
                                  None,operations=Operations())
    assert result == [b'off-nan-bits',b'old',b'new',b'off-minus-zero']
    assert calls == [('stock','stock'),('mask',1),('matrix','old'),('mask',2),('matrix','new')]
    assert binding.weights is stock


def test_single_target_layer_does_not_treat_other_generation_as_owner():
    shared=object();binding=Binding(Weights(shared,'stock'),None,[b'old-stock',b'new-stock'])
    class State:
        def run_inv_rope(self,value,*,stream):return value.output
    class Operations:
        def mask_tmp(self,source,owners,indices,*,slot):return slot
        def run_b(self,state,bound,alternate,mask,stream):return [b'new',b'new']
        def scatter_owner(self,output,edited,owners,indices,*,slot):
            for i,owner in enumerate(owners):
                if owner==slot:output[i]=edited[i]
    result=run_native_generations(binding,State(),((2,Weights(shared,'new')),),1,(1,2),None,
                                 None,operations=Operations())
    assert result == [b'old-stock',b'new']


def test_capture_constructor_matches_pinned_runner_and_overlay_hashes():
    package = Path(__file__).resolve().parents[1]/'flashalign_native'
    tree = ast.parse((package/'overlays/v1/worker/gpu/cudagraph_utils.py').read_text())
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ModelCudaGraphManager')
    constructor = next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
    parameters = {arg.arg for arg in constructor.args.args + constructor.args.kwonlyargs}
    module = ast.parse((package/'generations.py').read_text())
    capture = next(node for node in ast.walk(module) if isinstance(node,ast.FunctionDef) and node.name=='_capture')
    invocation = next(node.value for node in capture.body if isinstance(node,ast.Assign) and
                      any(isinstance(target,ast.Name) and target.id=='manager' for target in node.targets))
    assert {kw.arg for kw in invocation.keywords} <= parameters
    pins=json.loads((package/'overlay-pins.json').read_text())
    for name,record in pins.items():
        assert hashlib.sha256((package/'overlays'/name).read_bytes()).hexdigest()==record['overlay_sha256']


def test_real_decoder_forward_uses_generation_graph_identity():
    """Execute the actual overlay method without importing a GPU engine."""
    package=Path(__file__).resolve().parents[1]/'flashalign_native'
    path=package/'overlays/models/deepseek_v4_1/nvidia/model.py'
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='DeepseekV4DecoderLayer')
    forward=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    code=compile(ast.fix_missing_locations(ast.Module(body=[future,forward],type_ignores=[])),str(path),'exec')
    descriptor=SimpleNamespace(alignment_case=1,alignment_profile_digest='')
    namespace={'get_forward_context':lambda:SimpleNamespace(batch_descriptor=descriptor)}
    exec(code,namespace)
    calls=[];value=SimpleNamespace(ndim=2)
    def attention(positions,x,_unused,**kwargs):
        calls.append(kwargs)
        return x
    native=GenerationRuntime(SimpleNamespace(profile=SimpleNamespace(digest='a'*64),
        preparing=False,owner_rows=object()),None)
    native.active_snapshot=Snapshot((('g-new','b'*64),))
    layer=SimpleNamespace(_flashalign_native_runtime=native,engram=None,
        hc_attn_fn_broadcast=None,hc_attn_fn=None,hc_attn_scale=None,hc_attn_base=None,
        hc_ffn_fn=None,hc_ffn_scale=None,hc_ffn_base=None,attn_norm=SimpleNamespace(weight=None),
        ffn_norm=SimpleNamespace(weight=None),attn=attention,ffn=lambda x,_ids:x,
        _b12x_mhc=SimpleNamespace(pre=lambda *args,**kwargs:(value,None,None,value,None),
            post_pre=lambda *args,**kwargs:(value,None,None,value,None)))
    descriptor.alignment_profile_digest=native.graph_digest
    assert descriptor.alignment_profile_digest != native.profile.digest
    result=namespace['forward'](layer,value,None,None)
    assert result[0] is value and calls[-1]['native_mode']==1
    assert calls[-1]['owner_rows'] is native.owner_rows
    descriptor.alignment_profile_digest=native.profile.digest
    with pytest.raises(ValueError,match='descriptor profile digest mismatch'):
        namespace['forward'](layer,value,None,None)
    native.active_snapshot=None
    namespace['forward'](layer,value,None,None)


def test_real_native_installer_keeps_two_generation_packed_bindings(monkeypatch):
    """Use CPU tensors and a packing stub to exercise installation ownership."""
    torch = pytest.importorskip('torch', reason='The Python 3.12 CI arm runs the CPU tensor checks')
    module=ModuleType('b12x.gemm._shared.wo_mxfp8')
    def pack(weight,scale,**kwargs):
        expected=scale.view(torch.uint8).repeat_interleave(32,dim=0).unsqueeze(0)
        return SimpleNamespace(values=weight,scale_rows=expected,
            scale_mma=expected.reshape(1,1,1,1,1,32))
    module.pack_fp8_block_scaled_weight_mxfp8=pack
    monkeypatch.setitem(sys.modules,module.__name__,module)
    @dataclass(frozen=True)
    class Packed:
        wo_a: object
        wo_b: object
    weight=torch.tensor([[1.,1.],[1.,1.]]).to(torch.float8_e4m3fn)
    scale=torch.tensor([[127]],dtype=torch.uint8).view(torch.float8_e8m0fnu)
    stock=Packed(object(),object())
    attention=SimpleNamespace(layer_id=0,hidden_size=2,n_local_groups=1,o_lora_rank=2,
        _wo_projection_weights=stock,wo_b=SimpleNamespace(weight=weight,weight_scale_inv=scale))
    before=weight.view(torch.uint8).clone();pointer=weight.data_ptr()
    class Payload:
        def __init__(self,value):
            self.edited_weight=torch.full((2,2),value).to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
            self.edited_scale=scale.view(torch.uint8).numpy().tobytes()
        def partition(self,name,rank):
            return self.edited_weight if name.endswith('.weight') else self.edited_scale
        def verify_stock_partition(self,name,rank,raw):
            expected=weight if name.endswith('.weight') else scale
            assert raw==expected.view(torch.uint8).numpy().tobytes()
    def spec(digest,payload):
        def tensor(name):
            raw=payload.partition(name,0)
            return SimpleNamespace(shape=(2,2) if name.endswith('.weight') else (1,1),
                edited_partitions={0:hashlib.sha256(raw).hexdigest()})
        return SimpleNamespace(digest=digest,layers=(0,),hidden_size=2,input_size=2,tp_world_size=1,tensor=tensor)
    payload_a,payload_b=Payload(2.),Payload(3.)
    recipe_a,recipe_b=spec('a'*64,payload_a),spec('b'*64,payload_b)
    first=install_native_weights(attention,recipe_a,payload_a,0,generation_id='g-old')
    old=attention._flashalign_native_generations['g-old'][2]
    install_native_weights(attention,recipe_b,payload_b,0,generation_id='g-new')
    new=attention._flashalign_native_generations['g-new'][2]
    assert old is attention._flashalign_native_generations['g-old'][2]
    assert old.wo_a is new.wo_a is stock.wo_a
    assert old.wo_b.values.data_ptr()!=new.wo_b.values.data_ptr()
    assert old.wo_b.values.float().tolist()==[[2.,2.],[2.,2.]]
    assert new.wo_b.values.float().tolist()==[[3.,3.],[3.,3.]]
    assert install_native_weights(attention,recipe_a,payload_a,0,generation_id='g-old') is first
    assert attention._wo_projection_weights is stock and weight.data_ptr()==pointer
    assert torch.equal(weight.view(torch.uint8),before)
