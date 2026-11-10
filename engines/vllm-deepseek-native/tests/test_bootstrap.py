# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Real service/coordinator bootstrap regressions; backend I/O is CPU-only."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from owner_mode.profile_lifecycle import Artifact, JournalError, LifecycleManager, WorkerAck

ROOT=Path(__file__).resolve().parents[3]
KEY=b'k'*32


def module_for(engine):
    package='bootstrap_'+engine.replace('-','_')
    if engine=='llamacpp':
        sys.path.insert(0,str(ROOT/'engines/llamacpp'))
        spec=importlib.util.spec_from_file_location(package,ROOT/'engines/llamacpp/lifecycle.py')
        module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
        return module
    name=('flashalign_sglang' if engine=='sglang' else
          'flashalign_native' if engine=='vllm-deepseek-native' else 'flashalign_vllm')
    spec=importlib.util.spec_from_file_location(package,ROOT/'engines'/engine/name/'__init__.py',
                   submodule_search_locations=[str(ROOT/'engines'/engine/name)])
    module=importlib.util.module_from_spec(spec);sys.modules[package]=module;spec.loader.exec_module(module)
    return importlib.import_module(package+('.lifecycle_api' if engine=='sglang' else '.lifecycle_service'))


class Backend:
    identity='fixture.lifecycle'
    worker_ids=('worker-0',)
    instances=[]

    def __init__(self,*_,validate,estimate):
        self.validate,self.estimate=validate,estimate
        self.inventories=[{'model_id':'fixture/model','model_revision':'r1','available_device_bytes':1024*1024}]
        self.events=[];self.aliases={}
        self.instances.append(self)

    def ack(self,phase,tx,generation=None):
        self.events.append(phase)
        return [getattr(self,'ack_type',WorkerAck)('worker-0',tx['id'],generation.id if generation else None,
                          generation.artifact.digest if generation else None,phase)]

    def reserve(self,*args):pass
    def prepare(self,tx,g):return self.ack('prepared',tx,g)
    def fence(self,tx):return self.ack('fenced',tx)
    def commit(self,tx,g):return self.ack('committed',tx,g)
    def resume(self,tx):return self.ack('resumed',tx)
    def abort(self,tx,g):return self.ack('aborted',tx,g)
    def references(self,g):return 0
    def retire(self,tx,g):return self.ack('retired',tx,g)
    def reconcile(self,tx,snapshot):
        self.aliases=snapshot['aliases']
        return self.ack('reconciled',tx)
    def call(self,operation):
        self.events.append(operation)


class FakeProfile:
    def __init__(self,path):
        self.directory=Path(path)
        value=json.loads((self.directory/'profile.json').read_bytes())
        self.name=value['name'];self.meta={'model':value['model']}
        self.spec=SimpleNamespace(payload_file='payload.bin',tp_world_size=4)
        q=SimpleNamespace(numel=lambda:16,element_size=lambda:4)
        self.sites={0:(q,None,None)}


@pytest.fixture(params=['vllm-deepseek-native','vllm','vllm-gemma4','sglang','llamacpp'])
def setup_service(request,tmp_path,monkeypatch):
    module=module_for(request.param)
    operator='native_matrix_override_v1' if request.param=='vllm-deepseek-native' else 'projection'
    source=tmp_path/'source';source.mkdir()
    payload={'name':'owner','model':{'id':'fixture/model','revision':'r1'},'operator':operator}
    raw=json.dumps(payload).encode();(source/'profile.json').write_bytes(raw);(source/'payload.bin').write_bytes(b'1234')
    artifact_type=(module.Artifact if request.param=='llamacpp' else
                   importlib.import_module(module.__package__+'.flashalign_lifecycle').Artifact
                   if request.param=='sglang' else Artifact)
    def stage(src,dest):
        raw=(Path(src)/'profile.json').read_bytes();value=json.loads(raw)
        (dest/'profile.json').write_bytes(raw);(dest/'payload.bin').write_bytes(b'1234')
        return artifact_type(value['name'],hashlib.sha256(raw).hexdigest(),'fixture/model','r1',value['operator'],dest)
    base=tmp_path/'state'
    state=base/'lifecycle' if request.param=='llamacpp' else base
    active=[]
    if request.param=='sglang':
        monkeypatch.setattr(module,'stage_profile',stage)
        monkeypatch.setenv('FLASHALIGN_LIFECYCLE_STATE',str(state))
        monkeypatch.setenv('FLASHALIGN_PROFILES',str(source))
        def create():
            backend=Backend(validate=lambda a:None,estimate=lambda a:{'host':64,'worker-0':64})
            backend.ack_type=module.WorkerAck
            service=module.Controller(backend);service.initialize()
            service.permit=service.manager.admin(service._secret)
            return service
    elif request.param=='llamacpp':
        monkeypatch.setattr(module,'stage_projection',stage)
        monkeypatch.setattr(module,'read_profile',lambda p:(payload,None,None))
        class LlamaBackend(Backend):
            def __init__(self,*args,**kwargs):
                super().__init__(validate=lambda a:None,estimate=lambda a:{'profile_bytes':64})
                self.ack_type=module.WorkerAck
                self.profile_model=payload['model']
        monkeypatch.setattr(module,'LlamaCppBackend',LlamaBackend)
        def create():
            service=module.NativeLifecycle(base,control=None,model=tmp_path/'model',profiles=[source],gguf_python=tmp_path)
            active.append(service)
            return service
    else:
        monkeypatch.setattr(module,'VllmBackend',Backend)
        monkeypatch.setattr(module,'stage_validated_profile',stage)
        def create():
            return module.LifecycleService(None,None,[source],state,tmp_path/'admin.sock',loader=FakeProfile)
    yield module,source,state,stage,create
    for service in active:
        if not service.loop.is_closed():service.close()


def close(service):
    if hasattr(service,'loop') and not service.loop.is_closed():service.close()
    else:service.manager.close()


def test_fresh_service_initializes_once(setup_service):
    _,_,_,_,create=setup_service
    service=create()
    try:
        assert service.manager.journal_was_present is False
        assert service.manager.status()['aliases']['owner']['generation'] is not None
        assert 'prepared' in service.backend.events
        with pytest.raises(AttributeError):service.manager.journal_was_present=True
    finally:close(service)


def test_existing_empty_journal_stays_empty(setup_service):
    _,_,state,stage,create=setup_service
    backend=Backend(validate=lambda a:None,estimate=lambda a:{'device_bytes':4,'host_bytes':4})
    manager=LifecycleManager(state,backend=backend,stager=stage,admin_key=KEY,
                             resource_limits={'device_bytes':100,'host_bytes':100})
    assert manager.journal_was_present is False
    manager.close()
    service=create()
    try:
        assert service.manager.journal_was_present is True
        assert service.manager.status()['aliases']['owner']['generation'] is not None
        assert len(service.manager.status()['generations'])==1
        assert 'reconciled' in service.backend.events and 'resumed' in service.backend.events
        assert 'prepared' in service.backend.events
    finally:close(service)


def test_corrupt_existing_journal_fails_closed(setup_service):
    _,_,state,stage,create=setup_service
    manager=LifecycleManager(state,backend=Backend(validate=lambda a:None,estimate=lambda a:{}),stager=stage,
                             admin_key=KEY,resource_limits={'device_bytes':100,'host_bytes':100})
    journal=manager.journal;manager.close();journal.write_bytes(b'{broken')
    with pytest.raises(Exception) as raised:create()
    assert type(raised.value).__name__=='JournalError'
    assert 'prepared' not in Backend.instances[-1].events


def test_tombstone_is_not_reactivated(setup_service):
    _,_,_,_,create=setup_service
    service=create()
    service.manager.deactivate(service.permit,'owner',expected_revision=1)
    close(service)
    reopened=create()
    try:
        assert reopened.manager.status()['aliases']['owner']['generation'] is not None
        assert reopened.manager.status()['aliases']['owner']['revision']==3
        assert 'prepared' in reopened.backend.events
    finally:close(reopened)
