# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for real worker IPC adapters; these do not qualify GPU kernels."""
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from owner_mode.profile_lifecycle import GenerationBusy, LifecycleError
from owner_mode.profile_staging import stage_validated_profile
from flashalign_vllm.core import Profile, Selection, project, selected_rows, spans_for_runner
from flashalign_vllm.generations import ProjectionAdapter
from flashalign_vllm.lifecycle import SCHEMA, WorkerRegistry, artifact_record, VllmBackend
from flashalign_vllm.lifecycle_service import LifecycleService


def profile(path, vector=(1., 0.), site='attn_out'):
    path.mkdir()
    part = 'self_attn.o_proj' if site == 'attn_out' else 'mlp.down_proj'
    data = {'name':'owner','model':{'id':'fixture/model','revision':'fixture-revision'},
            'hidden_size':2,'sites':[{'layer':0,'site':site,'rank':1,'scale':1.,
            'tensor':f'model.language_model.layers.0.{part}.weight'}]}
    (path/'profile.json').write_text(json.dumps(data))
    save_file({f'L0.{site}.Q':torch.tensor([vector],dtype=torch.float32)},path/'profile.safetensors')
    return path


def runner():
    model=torch.nn.Module();model.model=torch.nn.Module()
    layer=torch.nn.Module();layer.self_attn=torch.nn.Module();layer.self_attn.o_proj=torch.nn.Identity()
    layer.mlp=torch.nn.Module();layer.mlp.down_proj=torch.nn.Identity()
    model.model.layers=torch.nn.ModuleList([layer])
    config=SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(
        hidden_size=2,num_hidden_layers=1,layer_types=['full_attention'])))
    return SimpleNamespace(get_model=lambda:model,vllm_config=config,device='cpu',requests={})


def registry(path):
    run=runner();adapter=ProjectionAdapter(run,{'owner':Profile(path)})
    reg=WorkerRegistry(adapter,'0');adapter.registry=reg;run._flashalign_lifecycle=reg
    return reg,run


def envelope(op, generation=None, artifact=None):
    tx={'id':'t-fixture','generation':generation,'was_allocated':False}
    data={'schema':SCHEMA,'operation':op,'transaction':tx}
    if artifact is not None:data.update(generation=generation,artifact=artifact_record(artifact))
    return data


def test_different_generations_same_alias_use_different_tensors_and_new_sites(tmp_path):
    first=profile(tmp_path/'a');second=profile(tmp_path/'b',(0.,1.),site='mlp_out')
    (tmp_path/'a-stage').mkdir();(tmp_path/'b-stage').mkdir()
    a=stage_validated_profile(first,tmp_path/'a-stage');b=stage_validated_profile(second,tmp_path/'b-stage')
    reg,run=registry(first)
    for gid,artifact in [('g-first',a),('g-second',b)]:
        reg.dispatch(envelope('prepare',gid,artifact));reg.dispatch(envelope('fence'))
        reg.dispatch(envelope('commit',gid,artifact));reg.dispatch(envelope('resume'))
    old=Selection('owner',a.digest,1.,'g-first');new=Selection('owner',b.digest,1.,'g-second')
    x=torch.tensor([[3.,4.],[5.,6.],[7.,8.]])
    with selected_rows([(0,1,old),(1,2,new)]):
        got=run.get_model().model.layers[0].self_attn.o_proj(x)
        torch.testing.assert_close(got,torch.tensor([[0.,4.],[5.,6.],[7.,8.]]))
        got=run.get_model().model.layers[0].mlp.down_proj(x)
        torch.testing.assert_close(got,torch.tensor([[3.,4.],[5.,0.],[7.,8.]]))
    assert reg.generations['g-first'].handle.digest == a.digest
    assert reg.generations['g-second'].handle.digest == b.digest
    with pytest.raises(LifecycleError):reg.resolve('g-first',b.digest)


def test_retirement_blocks_cached_and_live_generations(tmp_path):
    source=profile(tmp_path/'a');(tmp_path/'staged').mkdir()
    artifact=stage_validated_profile(source,tmp_path/'staged');reg,run=registry(source)
    reg.dispatch(envelope('prepare','g-one',artifact));reg.dispatch(envelope('fence'))
    reg.dispatch(envelope('commit','g-one',artifact));reg.resolve('g-one',artifact.digest)
    with pytest.raises(GenerationBusy):reg.dispatch(envelope('retire','g-one',artifact))
    run.requests['active']=SimpleNamespace(sampling_params=SimpleNamespace(extra_args={'_flashalign_generation':'g-one'}))
    with pytest.raises(GenerationBusy):reg.dispatch(envelope('evict_cache'))
    run.requests.clear();reg.dispatch(envelope('evict_cache'))
    reg.dispatch(envelope('retire','g-one',artifact));assert not reg.generations


def test_fixed_rpc_rejects_unknown_operations_and_fields(tmp_path):
    source=profile(tmp_path/'a');reg,_=registry(source)
    with pytest.raises(LifecycleError):reg.dispatch({'schema':SCHEMA,'operation':'__import__'})
    with pytest.raises(LifecycleError):reg.dispatch({'schema':SCHEMA,'operation':'inventory','method':'arbitrary'})


@pytest.mark.parametrize('state,phase', [('staged','aborting'),('retiring','retiring')])
def test_recovery_does_not_resurrect_aborted_or_retiring_generations(tmp_path,state,phase):
    source=profile(tmp_path/'a');root=tmp_path/'artifacts';root.mkdir()
    (root/'g-one').mkdir()
    artifact=stage_validated_profile(source,root/'g-one');reg,_=registry(source)
    reg.dispatch(envelope('prepare','g-one',artifact))
    recovery={'schema':SCHEMA,'operation':'reconcile','transaction':{'id':'t-recover'},
              'snapshot':{'artifact_root':str(root),'aliases':{},'generations':{
                  'g-one':{'allocated':True,'state':state,'artifact':artifact.record()}},
                  'pending':{'operation':'activate' if phase=='aborting' else 'retire',
                             'phase':phase,'generation':'g-one','was_allocated':False}}}
    assert reg.dispatch(recovery)['phase']=='reconciled'
    assert reg.generations == {}


def test_api_backend_real_manager_and_async_collective_roundtrip(tmp_path,monkeypatch):
    async def run():
        source=profile(tmp_path/'a');updated=profile(tmp_path/'b',(0.,1.))
        reg,model=registry(source);calls=[]
        class Client:
            async def collective_rpc(self,method,*,timeout,args):
                assert method=='flashalign_lifecycle';calls.append(args[0]['operation'])
                await asyncio.sleep(0)
                return [reg.dispatch(args[0])]
            async def reset_prefix_cache(self,*,reset_running_requests):
                assert reset_running_requests is False
                return True
        service=await asyncio.to_thread(LifecycleService,Client(),asyncio.get_running_loop(),
                                        [source],tmp_path/'state',tmp_path/'state/admin.sock')
        try:
            assert service.manager.status()['aliases']['owner']['revision']==1
            lease=service.manager.pin('owner');old=lease.generation
            new=await asyncio.to_thread(service.manager.stage,service.permit,updated)
            await asyncio.to_thread(service.manager.activate,service.permit,'owner',new.id,expected_revision=1)
            assert lease.generation.id==old.id
            assert service.manager.pin('owner').generation.id==new.id
            assert reg.resolve(old.id,old.artifact.digest) is not reg.resolve(new.id,new.artifact.digest)
            status = await asyncio.to_thread(service.public_status)
            assert status['schema'] == 'flashalign.runtime-status.v1'
            assert status['generation'] == new.id
            assert status['profile_digest'] == new.artifact.digest
            assert status['graph_mode'] == 'eager'
            assert status['expected_sites'] == 1
            assert 'model_digest' not in status
            assert str(tmp_path) not in json.dumps(status)
            # A control thread can hold admission while awaiting this event
            # loop's RPC. New API admissions must fail promptly, never wait.
            entered = threading.Event(); release = threading.Event()
            def control():
                with service._admission:
                    entered.set(); release.wait(5)
            task = asyncio.create_task(asyncio.to_thread(control))
            while not entered.is_set(): await asyncio.sleep(.001)
            try:
                with pytest.raises(GenerationBusy): service.pin('owner')
                assert service.aliases['owner'].digest == new.artifact.digest
            finally:
                release.set(); await task
            lease.close()
            assert calls.count('commit')==2
            with pytest.raises(RuntimeError):service.backend._call('inventory')
        finally:service.manager.close()
    asyncio.run(run())


def test_packaged_shared_backend_copies_match():
    root = Path(__file__).resolve().parents[3]
    expected = (root/'engines/common/vllm_lifecycle.py').read_bytes()
    for package in ('vllm/flashalign_vllm','vllm-gemma4/flashalign_vllm','vllm-deepseek-native/flashalign_native'):
        assert (root/'engines'/package/'lifecycle.py').read_bytes() == expected


def test_restart_preserves_deactivated_alias_after_all_generations_retire(tmp_path):
    async def run():
        source=profile(tmp_path/'a');reg,_=registry(source)
        class Client:
            async def collective_rpc(self,method,*,timeout,args):
                return [reg.dispatch(args[0])]
        loop=asyncio.get_running_loop()
        service=await asyncio.to_thread(LifecycleService,Client(),loop,[source],
                                       tmp_path/'state',tmp_path/'state/admin.sock')
        gid=service.manager.status()['aliases']['owner']['generation']
        await asyncio.to_thread(service.manager.deactivate,service.permit,'owner',expected_revision=1)
        await asyncio.to_thread(service.manager.retire,service.permit,gid)
        assert service.manager.status()['generations']==[]
        service.manager.close()
        restarted=await asyncio.to_thread(LifecycleService,Client(),loop,[source],
                                         tmp_path/'state',tmp_path/'state/admin.sock')
        try:
            status=restarted.manager.status()
            assert status['generations']==[]
            assert status['aliases']['owner']=={'generation':None,'revision':2}
            assert list(restarted.aliases)==[]
            assert not reg.generations
        finally:
            restarted.manager.close()
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['registration', 'publication'])
def test_later_hook_registration_failure_removes_only_new_hooks(tmp_path, monkeypatch, failure):
    import copy

    first = profile(tmp_path/'first')
    updated = profile(tmp_path/'updated', (0., 1.))
    metadata = json.loads((updated/'profile.json').read_text())
    metadata['sites'].extend([
        {'layer':1,'site':'attn_out','rank':1,'scale':1.,
         'tensor':'model.language_model.layers.1.self_attn.o_proj.weight'},
        {'layer':1,'site':'mlp_out','rank':1,'scale':1.,
         'tensor':'model.language_model.layers.1.mlp.down_proj.weight'},
    ])
    (updated/'profile.json').write_text(json.dumps(metadata))
    save_file({name:torch.tensor([[0., 1.]],dtype=torch.float32) for name in
               ('L0.attn_out.Q','L1.attn_out.Q','L1.mlp_out.Q')}, updated/'profile.safetensors')
    (tmp_path/'old-stage').mkdir();(tmp_path/'new-stage').mkdir()
    old_artifact = stage_validated_profile(first, tmp_path/'old-stage')
    new_artifact = stage_validated_profile(updated, tmp_path/'new-stage')
    run = runner()
    model = run.get_model()
    model.model.layers.append(copy.deepcopy(model.model.layers[0]))
    run.vllm_config.model_config.hf_text_config.num_hidden_layers = 2
    run.vllm_config.model_config.hf_text_config.layer_types.append('full_attention')
    adapter = ProjectionAdapter(run, {'owner':Profile(first)})
    reg = WorkerRegistry(adapter, '0');adapter.registry = reg
    reg.dispatch(envelope('prepare', 'g-old', old_artifact))
    reg.dispatch(envelope('fence'))
    commit = envelope('commit', 'g-old', old_artifact)
    commit['transaction']['alias'] = 'owner'
    reg.dispatch(commit);reg.dispatch(envelope('resume'))
    old_handle = reg.resolve('g-old', old_artifact.digest)
    old_hooks = dict(adapter.hooks)
    added_before_failure = model.model.layers[1].self_attn.o_proj
    fails_later = model.model.layers[1].mlp.down_proj
    original = fails_later.register_forward_hook

    def fail(_hook):
        assert len(added_before_failure._forward_hooks) == 1
        raise MemoryError('injected later hook registration failure')

    if failure == 'registration':
        monkeypatch.setattr(fails_later, 'register_forward_hook', fail)
    else:
        class PartialPublication(dict):
            def copy(self):
                return type(self)(self)
            def update(self, entries):
                for name, handle in entries.items():
                    self[name] = handle
                    raise MemoryError('injected partial dictionary publication')
        adapter.hooks = PartialPublication(adapter.hooks)
    existing_map = adapter.hooks
    with pytest.raises(MemoryError):
        reg.dispatch(envelope('prepare', 'g-new', new_artifact))
    assert not added_before_failure._forward_hooks
    assert not fails_later._forward_hooks
    assert adapter.hooks is existing_map
    assert adapter.hooks == old_hooks
    assert reg.aliases == {'owner':'g-old'} and not reg.fenced
    assert set(reg.generations) == {'g-old'}
    assert reg.resolve('g-old', old_artifact.digest) is old_handle
    assert adapter.statistics()['installed_sites'] == len(old_hooks)
    with selected_rows([(0,1,Selection('owner',old_artifact.digest,1.,'g-old'))]):
        torch.testing.assert_close(model.model.layers[0].self_attn.o_proj(torch.tensor([[3.,4.]])),
                                   torch.tensor([[0.,4.]]))
    # A later successful prepare installs each new site once, without a leaked
    # callback from the failed candidate.
    monkeypatch.setattr(fails_later, 'register_forward_hook', original)
    adapter.hooks = dict(adapter.hooks)
    reg.dispatch(envelope('prepare', 'g-new', new_artifact))
    assert len(added_before_failure._forward_hooks) == len(fails_later._forward_hooks) == 1
    assert len(adapter.hooks) == 3


def test_projection_generation_copies_match_canonical():
    root = Path(__file__).resolve().parents[3]
    expected = (root/'engines/common/vllm_generations.py').read_bytes()
    for package in ('vllm', 'vllm-gemma4'):
        assert (root/'engines'/package/'flashalign_vllm/generations.py').read_bytes() == expected
