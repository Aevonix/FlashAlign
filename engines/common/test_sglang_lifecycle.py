# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CPU tests of the real control serialization with explicit fake model runners.

These tests establish lifecycle protocol behavior, not SGLang/GPU numerics.
"""
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'engines/sglang/flashalign_sglang'


class Tensor:
    def numel(self): return 4
    def element_size(self): return 4


class Dispatcher:
    def __init__(self, items=()): self.items = dict(items)
    def __iadd__(self, other): self.items.update(other.items); return self
    def __call__(self, value): self.items[type(value)](value)


class Wire:
    def __init__(self, **kw): self.__dict__.update(kw)


class Input(Wire): pass
class Output(Wire): pass


def package(name, source, directory=PACKAGE):
    mod = ModuleType(name)
    mod.__path__ = [str(directory)]
    sys.modules[name] = mod
    gate_module = importlib.import_module(name + '.gate')
    profiles = {}
    def load(path):
        raw = (Path(path) / 'profile.json').read_bytes()
        meta = json.loads(raw)
        if meta.get('transform') not in ('projection', 'row_normalized_projection_v1'):
            raise ValueError('Unsupported fake operator')
        hashes = {'profile.json':hashlib.sha256(raw).hexdigest(),
                  'profile.safetensors':hashlib.sha256((Path(path)/'profile.safetensors').read_bytes()).hexdigest()}
        digest = hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return SimpleNamespace(name=meta['name'], digest=digest, model=meta['model'],
                               sites={tuple(site): {'q': Tensor()} for site in meta['sites']})
    profile = load(source)
    profiles[profile.name] = profile
    gate = gate_module.Gate(profiles, b'k' * 32)
    runtime = ModuleType(name + '.runtime')
    runtime.load_profile = load
    runtime.registry = lambda: (profiles, gate)
    def install(runner):
        runner.installs += 1
        if runner.fail:
            raise ValueError('Fake runner preparation failed')
        runner.sites = set().union(*(p.sites for p in profiles.values()))
    runtime.install = install
    sys.modules[runtime.__name__] = runtime
    return runtime, gate_module


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.write_profile('initial', b'first')
        self.api_name = 'fa_api_' + secrets.token_hex(8)
        self.worker_name = 'fa_worker_' + secrets.token_hex(8)
        self.api_runtime, self.api_gate = package(self.api_name, self.source)
        self.worker_runtime, _ = package(self.worker_name, self.source)
        self.api = importlib.import_module(self.api_name + '.lifecycle_api')
        self.worker = importlib.import_module(self.worker_name + '.lifecycle_worker')
        io = ModuleType('sglang.srt.managers.io_struct')
        io.FlashAlignControlReqInput = Input
        io.FlashAlignControlReqOutput = Output
        utils = ModuleType('sglang.utils')
        utils.TypeBasedDispatcher = Dispatcher
        torch_stub = ModuleType('torch')
        torch_stub.cuda = SimpleNamespace(synchronize=lambda device: None)
        self.modules = patch.dict(sys.modules, {'sglang.srt.managers.io_struct': io, 'sglang.utils': utils, 'torch': torch_stub})
        self.modules.start()
        self.env = patch.dict(os.environ, {'FLASHALIGN_LIFECYCLE_STATE': str(self.root/'state'),
                                          'FLASHALIGN_PROFILES': str(self.source)})
        self.env.start()
        target = SimpleNamespace(device='cpu', is_draft_worker=False, installs=0, sites=set(), fail=False)
        native = SimpleNamespace(device='cpu', is_draft_worker=True, installs=0, sites=set(), fail=False)
        self.worker.register_runner(target)
        self.worker.register_runner(native)
        self.runners = [target, native]
        self.scheduler = SimpleNamespace(ps=SimpleNamespace(tp_rank=0, tp_size=1), waiting_queue=[],
            running_batch=SimpleNamespace(reqs=[]), tree_cache=SimpleNamespace(root_node=SimpleNamespace(children={})),
            _flashalign_fenced=False, accepted=[])
        self.scheduler.handle_generate_request = self.scheduler.accepted.append
        self.scheduler.is_fully_idle = lambda: not self.scheduler.waiting_queue and not self.scheduler.running_batch.reqs
        def flush(empty_cache):
            self.scheduler.tree_cache.root_node.children.clear()
            return self.scheduler.is_fully_idle()
        self.scheduler.flush_cache = flush
        token = SimpleNamespace(server_args=SimpleNamespace(tp_size=1), _result_dispatcher=Dispatcher())
        token.auto_create_handle_loop = lambda: None
        async def dispatch(request):
            result = self.worker.handle_control(self.scheduler, request)
            token._result_dispatcher(result)
        token._async_dispatch_to_scheduler = dispatch
        self.api.install_tokenizer_control(token)
        self.backend = self.api.Backend(token, asyncio.get_running_loop())
        self.controller = self.api.Controller(self.backend)
        await asyncio.to_thread(self.controller.initialize)

    async def asyncTearDown(self):
        self.controller.manager.close()
        self.env.stop()
        self.modules.stop()
        for name in list(sys.modules):
            if name.startswith(self.api_name) or name.startswith(self.worker_name):
                sys.modules.pop(name)
        self.tmp.cleanup()

    def write_profile(self, name, tensor, *, model='arbitrary/model', sites=((0,'attn_out'),), transform='projection'):
        directory = self.root / (name + '-' + secrets.token_hex(4))
        directory.mkdir()
        (directory/'profile.json').write_text(json.dumps({'name': name, 'model': {'id': model, 'revision': 'pinned'},
            'transform': transform, 'sites': sites}))
        (directory/'profile.safetensors').write_bytes(tensor)
        return directory

    async def stage(self, source):
        return await asyncio.to_thread(self.controller.operation, {'action': 'stage', 'path': str(source)})

    async def activate(self, generation, revision=1):
        return await asyncio.to_thread(self.controller.operation, {'action': 'activate', 'alias': 'initial',
                                    'generation': generation, 'expected_revision': revision})

    async def test_bootstrap_crosses_wire_and_covers_target_and_native_runners(self):
        state = self.controller.manager.status()
        gid = state['aliases']['initial']['generation']
        self.assertTrue(gid.startswith('g-'))
        self.assertEqual(set(self.worker_runtime.registry()[0]), {gid})
        self.assertIn('initial', self.api_runtime.registry()[0])
        self.assertTrue(all(runner.installs == 1 for runner in self.runners))
        self.assertFalse(self.scheduler._flashalign_fenced)

    async def test_old_signed_namespace_stays_bound_after_alias_switch(self):
        original = self.api_runtime.registry()[0]['initial']
        selected = self.api_gate.Selection('initial', original.digest, 0.8)
        old, lease = self.controller.bind(selected)
        old_salt = self.api_runtime.registry()[1].cache_salt(old, None)
        staged = await self.stage(self.write_profile('another-compatible-name', b'next', sites=((0,'attn_out'),(9,'mlp_out'))))
        await self.activate(staged['generation'])
        new, current = self.controller.bind(selected)
        self.assertNotEqual(old.name, new.name)
        self.assertEqual(self.worker_runtime.registry()[1].from_salt(old_salt).name, old.name)
        self.assertTrue(all((9,'mlp_out') in runner.sites for runner in self.runners))
        with self.assertRaises(self.api.LifecycleError):
            await asyncio.to_thread(self.controller.operation, {'action':'retire','generation':old.name})
        lease.close(); current.close()
        await asyncio.to_thread(self.controller.operation, {'action':'retire','generation':old.name})
        self.assertNotIn(old.name, self.worker_runtime.registry()[0])

    async def test_cached_generation_blocks_retirement_until_explicit_eviction(self):
        first = self.controller.manager.status()['aliases']['initial']['generation']
        old, lease = self.controller.bind(self.api_gate.Selection('initial', '', 1.0))
        salt = self.api_runtime.registry()[1].cache_salt(old, None)
        lease.close()
        self.scheduler.tree_cache.root_node.children['cache'] = SimpleNamespace(
            children={}, key=SimpleNamespace(cache_salt=salt))
        next_generation = await self.stage(self.write_profile('next', b'new'))
        await self.activate(next_generation['generation'])
        with self.assertRaises(self.api.LifecycleError):
            await asyncio.to_thread(self.controller.operation, {'action':'retire','generation':first})
        await asyncio.to_thread(self.controller.operation, {'action':'evict_cache'})
        await asyncio.to_thread(self.controller.operation, {'action':'retire','generation':first})
        self.assertFalse(self.scheduler.tree_cache.root_node.children)

    async def test_worker_failure_does_not_mutate_alias(self):
        before = self.controller.manager.status()['aliases']
        staged = await self.stage(self.write_profile('new', b'new'))
        self.runners[1].fail = True
        with self.assertRaises(self.api.LifecycleError):
            await self.activate(staged['generation'])
        self.assertEqual(self.controller.manager.status()['aliases'], before)
        self.assertNotIn(staged['generation'], self.worker_runtime.registry()[0])
        self.assertFalse(self.scheduler._flashalign_fenced)

    async def test_fence_defers_new_requests_and_keeps_existing_requests(self):
        existing = SimpleNamespace(cache_salt=None)
        self.scheduler.running_batch.reqs.append(existing)
        await asyncio.to_thread(self.backend.call, 'fence')
        incoming = SimpleNamespace(cache_salt=None)
        self.worker.dispatch_generate(self.scheduler, incoming)
        self.assertEqual(self.scheduler.running_batch.reqs, [existing])
        self.assertFalse(self.scheduler.accepted)
        await asyncio.to_thread(self.backend.call, 'resume')
        self.assertEqual(self.scheduler.accepted, [incoming])

    async def test_stale_source_or_different_model_rejected(self):
        with self.assertRaises(ValueError):
            await self.stage(self.write_profile('wrong', b'bad', model='different/model'))
        staged = await self.stage(self.write_profile('snapshot', b'right'))
        location = self.root / 'state/generations' / staged['generation'] / 'profile.safetensors'
        location.chmod(0o600)
        location.write_bytes(b'tampered')
        with self.assertRaises(self.api.LifecycleError):
            await self.activate(staged['generation'])

    async def test_missing_worker_ack_is_rejected(self):
        self.backend.worker_ids = ('tp-0', 'tp-1')
        with self.assertRaises(self.api.LifecycleError):
            await asyncio.to_thread(self.backend.call, 'inspect')

    async def test_internal_command_cannot_be_supplied_as_operation(self):
        with self.assertRaises(ValueError):
            await asyncio.to_thread(self.controller.operation, {'action': 'commit', 'generation': 'guessed'})
        with self.assertRaises(ValueError):
            await asyncio.to_thread(self.controller.operation, {'action': 'recover', 'key': 'generation-signing-key'})

    async def test_timeout_response_cannot_satisfy_next_command(self):
        tokenizer = self.backend.tokenizer
        old = Output(rid='flashalign-control-stale', payload='{"workers":[]}')
        tokenizer._result_dispatcher(old)
        self.assertFalse(tokenizer._flashalign_futures)
        rows = await self.api.control(tokenizer, {'action':'inspect','transaction':{'id':'r-'+'a'*48}})
        self.assertEqual(len(rows), 1)

    async def test_local_failure_still_participates_in_full_rank_ack(self):
        self.scheduler.ps.tp_size = 2
        self.scheduler.tp_cpu_group = object()
        calls = []
        def collective(destination, local, group):
            calls.append(group)
            destination[:] = [local, {**local, 'worker_id':'tp-1'}]
        sys.modules['torch'].distributed = SimpleNamespace(all_gather_object=collective)
        message = Input(rid='control-test', payload=json.dumps({'action':'untrusted-command',
                        'transaction':{'id':'r-'+'a'*48}}))
        output = self.worker.handle_control(self.scheduler, message)
        result = json.loads(output.payload)
        self.assertEqual(calls, [self.scheduler.tp_cpu_group])
        self.assertEqual(len(result['workers']), 2)
        self.assertTrue(all(row['ok'] is False for row in result['workers']))

    async def test_public_http_cannot_reach_control_transport(self):
        http = importlib.import_module(self.api_name + '.http')
        events, called = [], []
        async def app(*args): called.append(True)
        async def receive(): raise AssertionError('Admin data plane body should not be read')
        async def send(event): events.append(event)
        scope = {'type':'http', 'path':'/flashalign/operations', 'method':'POST',
                 'headers':[(b'x-owner-mode-auth', b'valid-or-invalid-generation-signature')]}
        await http.Middleware(app)(scope, receive, send)
        self.assertEqual(events[0]['status'], 405)
        self.assertFalse(called)

    async def test_recovery_finishes_durable_commit_after_ack_loss(self):
        staged = await self.stage(self.write_profile('replacement', b'replacement'))
        original = self.backend.commit
        def lost_ack(transaction, generation):
            original(transaction, generation)
            raise RuntimeError('Simulated transport loss after worker commit')
        with patch.object(self.backend, 'commit', lost_ack):
            with self.assertRaises(self.api.LifecycleError):
                await self.activate(staged['generation'])
        self.assertTrue(self.controller.manager.status()['admission_fenced'])
        await asyncio.to_thread(self.controller.operation, {'action':'recover'})
        state = self.controller.manager.status()
        self.assertFalse(state['admission_fenced'])
        self.assertEqual(state['aliases']['initial']['generation'], staged['generation'])

    async def test_source_changes_do_not_mutate_staged_generation(self):
        source = self.write_profile('snapshot', b'original')
        staged = await self.stage(source)
        (source/'profile.safetensors').write_bytes(b'edited-after-staging')
        await self.activate(staged['generation'])
        active = self.worker_runtime.registry()[0][staged['generation']]
        self.assertEqual(active.digest, staged['artifact']['digest'])

    async def test_generated_engine_modules_have_one_canonical_source(self):
        import subprocess
        subprocess.run([sys.executable, str(ROOT/'engines/common/sync_lifecycle.py'), '--check'],
                       check=True, capture_output=True)

    async def test_deactivate_disables_alias_and_keeps_prior_worker_generation(self):
        selected, lease = self.controller.bind(self.api_gate.Selection('initial', '', 1.0))
        await asyncio.to_thread(self.controller.operation,
                                {'action':'deactivate','alias':'initial','expected_revision':1})
        self.assertNotIn('initial', self.api_runtime.registry()[0])
        self.assertIn(selected.name, self.worker_runtime.registry()[0])
        with self.assertRaises(self.api_gate.FieldError):
            self.controller.bind(self.api_gate.Selection('initial', '', 1.0))
        lease.close()
        await asyncio.to_thread(self.controller.operation, {'action':'retire','generation':selected.name})
        await asyncio.to_thread(self.controller.operation, {'action':'recover'})
        self.assertNotIn('initial', self.api_runtime.registry()[0])

    async def test_deferred_abort_runs_before_and_after_resume(self):
        aborted = []
        self.scheduler.abort_request = aborted.append
        await asyncio.to_thread(self.backend.call, 'fence')
        request = SimpleNamespace(rid='cancelled')
        self.worker.dispatch_abort(self.scheduler, request)
        self.assertEqual(aborted, [request])
        await asyncio.to_thread(self.backend.call, 'resume')
        self.assertEqual(aborted, [request, request])

    async def test_response_headers_retain_admitted_generation_after_alias_switch(self):
        http=importlib.import_module(self.api_name+'.http')
        self.api._CONTROLLER=self.controller
        prior=self.controller.manager.status()['aliases']['initial']['generation']
        staged=await self.stage(self.write_profile('next',b'next'))
        output=[]
        async def app(scope,receive,send):
            await receive()
            await self.activate(staged['generation'])
            await send({'type':'http.response.start','status':200,'headers':[]})
            await send({'type':'http.response.body','body':b'{}'})
        async def receive():return {'type':'http.request','body':b'{"owner_mode":"initial"}'}
        async def send(event):output.append(event)
        selection=self.api_gate.Selection('initial','',1.0)
        gate=self.api_runtime.registry()[1]
        with patch.object(gate,'select',return_value=selection):
            await http.Middleware(app)({'type':'http','path':'/v1/chat/completions','method':'POST','headers':[]},receive,send)
        headers=dict(output[0]['headers'])
        self.assertEqual(headers[b'x-flashalign-generation'],prior.encode())
        self.assertEqual(headers[b'x-owner-mode-effective'],b'initial')
        self.assertNotEqual(headers[b'x-flashalign-generation'],staged['generation'].encode())

    async def test_public_status_has_scalar_identity_and_no_internal_state(self):
        self.api._CONTROLLER=self.controller
        state=await self.api.public_status()
        self.assertEqual(state['model'],'arbitrary/model')
        self.assertIsInstance(state['generation'],str)
        self.assertEqual(state['graph_mode'],'eager')
        self.assertNotIn('lifecycle',state)
        self.assertNotIn(str(self.root),json.dumps(state))

    async def test_explicit_matching_model_fingerprints_allow_primary_id_variants(self):
        support=importlib.import_module(self.api_name+'.lifecycle_support')
        item={'id':'serving/quantized-variant','revision':'a'*40,
              'config_sha256':'b'*64,'index_sha256':'c'*64,'quantization_config_sha256':'d'*64}
        model={'id':item['id'],'revision':item['revision'],'sha256':{
            'config.json':item['config_sha256'],'model.safetensors.index.json':item['index_sha256']}}
        scope=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
        profile=SimpleNamespace(model={'id':'compatible/original','revision':'e'*40},
                                compatible_models=[item],digest='f'*64)
        with patch.object(self.api_runtime,'load_profile',return_value=profile),patch.dict(os.environ,{'FLASHALIGN_MODEL_SCOPE':scope}):
            self.assertIs(support.checked_profile(self.source,profile.digest),profile)
        with patch.object(self.api_runtime,'load_profile',return_value=profile),patch.dict(os.environ,{'FLASHALIGN_MODEL_SCOPE':'wrong'}):
            with self.assertRaises(ValueError):support.checked_profile(self.source,profile.digest)

    async def test_internal_generation_is_not_a_public_admission_alias(self):
        gid=self.controller.manager.status()['aliases']['initial']['generation']
        profile=self.api_runtime.registry()[0][gid]
        with self.assertRaises(self.api_gate.FieldError):
            self.controller.bind(self.api_gate.Selection(gid,profile.digest,1.0))

    async def test_recovery_discards_uncommitted_allocation_after_abort_loss(self):
        graph_allocations=[]
        for runner in self.runners:
            allocated={}
            graph_allocations.append(allocated)
            runner.flashalign_graph=SimpleNamespace(
                prepare_generation=lambda gid,profile,allocated=allocated:allocated.update({gid:profile.digest}),
                retire_generation=lambda gid,digest,quiescent,allocated=allocated:allocated.pop(gid,None),
                status=lambda:{'graph_mode':'eager'})
        staged=await self.stage(self.write_profile('uncommitted',b'new'))
        original=self.backend.prepare
        def lose_ack(transaction,generation):
            original(transaction,generation)
            raise OSError('Lost prepare acknowledgment')
        with patch.object(self.backend,'prepare',lose_ack),patch.object(self.backend,'abort',side_effect=OSError('Abort transport lost')):
            with self.assertRaises(self.api.LifecycleError):await self.activate(staged['generation'])
        self.assertIn(staged['generation'],self.worker._PREPARED)
        self.assertTrue(all(staged['generation'] in entries for entries in graph_allocations))
        self.assertNotIn(staged['generation'],self.worker._COMMITTED)
        await asyncio.to_thread(self.controller.operation,{'action':'recover'})
        self.assertNotIn(staged['generation'],self.worker._PREPARED)
        self.assertNotIn(staged['generation'],self.worker._COMMITTED)
        self.assertNotIn(staged['generation'],self.worker_runtime.registry()[0])
        self.assertNotIn(staged['generation'],self.api_runtime.registry()[0])
        self.assertTrue(all(staged['generation'] not in entries for entries in graph_allocations))
        row=next(row for row in self.controller.manager.status()['generations'] if row['id']==staged['generation'])
        self.assertFalse(row['allocated'])

    async def test_recovery_excludes_pending_retirement_from_active_set(self):
        support=importlib.import_module(self.api_name+'.lifecycle_support')
        rows={'kept':{'allocated':True,'state':'retained'},'retired':{'allocated':True,'state':'retiring'}}
        snapshot={'generations':rows,'pending':{'operation':'retire','generation':'retired','was_allocated':True,'phase':'retiring'}}
        self.assertEqual(set(support.recovery_generations(snapshot)),{'kept'})
        snapshot={'generations':{'kept':rows['kept']},'pending':{'operation':'activate','generation':'kept',
                    'was_allocated':True,'phase':'aborting'}}
        self.assertEqual(set(support.recovery_generations(snapshot)),{'kept'})

    async def test_recovery_preserves_deferred_old_request_after_lost_commit_ack(self):
        selected,lease=self.controller.bind(self.api_gate.Selection('initial','',1.0))
        salt=self.api_runtime.registry()[1].cache_salt(selected,None)
        staged=await self.stage(self.write_profile('committed-next',b'new'))
        original=self.backend.commit
        def lost_ack(transaction,generation):
            original(transaction,generation)
            raise OSError('Commit reply lost')
        with patch.object(self.backend,'commit',lost_ack):
            with self.assertRaises(self.api.LifecycleError):await self.activate(staged['generation'])
        request=SimpleNamespace(cache_salt=salt)
        self.worker.dispatch_generate(self.scheduler,request)
        self.assertEqual(self.scheduler._flashalign_deferred,[request])
        await asyncio.to_thread(self.controller.operation,{'action':'recover'})
        self.assertFalse(self.controller.manager.status()['admission_fenced'])
        self.assertEqual(self.scheduler.accepted,[request])
        self.assertIn(selected.name,self.worker_runtime.registry()[0])
        self.assertFalse(self.scheduler._flashalign_deferred)
        lease.close()

    async def test_unified_cache_facade_and_opaque_cores_cannot_lose_references(self):
        selected,lease=self.controller.bind(self.api_gate.Selection('initial','',1.0))
        salt=self.api_runtime.registry()[1].cache_salt(selected,None)
        lease.close()
        root=SimpleNamespace(key=None,children={'cached':SimpleNamespace(key=SimpleNamespace(cache_salt=salt),children={})})
        self.scheduler.tree_cache=SimpleNamespace(tree_core=SimpleNamespace(root_node=root))
        self.assertEqual(self.worker.references(self.scheduler,selected.name),1)
        class Opaque:
            @property
            def root_node(self):raise NotImplementedError()
        self.scheduler.tree_cache=SimpleNamespace(tree_core=Opaque(),total_size=lambda:(128,0))
        self.assertEqual(self.worker.references(self.scheduler,selected.name),1)
        self.scheduler.tree_cache.total_size=lambda:(0,0)
        self.assertEqual(self.worker.references(self.scheduler,selected.name),0)
        self.scheduler.tree_cache=SimpleNamespace()
        with self.assertRaises(ValueError):self.worker.references(self.scheduler,selected.name)

    async def test_batch_admission_defers_original_request_ids_and_abort_routes(self):
        await asyncio.to_thread(self.backend.call,'fence')
        requests=[SimpleNamespace(rid='batch-'+str(i),http_worker_ipc='return-'+str(i),cache_salt=None) for i in range(3)]
        self.worker.dispatch_batch_generate(self.scheduler,requests)
        self.assertEqual(self.scheduler._flashalign_deferred,requests)
        self.assertFalse(self.scheduler.accepted)
        await asyncio.to_thread(self.backend.call,'resume')
        self.assertEqual(self.scheduler.accepted,requests)
        self.assertEqual([request.http_worker_ipc for request in self.scheduler.accepted],['return-0','return-1','return-2'])

    async def test_probe_base_v1_routes_reach_each_real_middleware(self):
        import httpx
        async def no_fallthrough(*args):
            raise AssertionError('FlashAlign operational route fell through to the model API')
        for engine in ('sglang','sglang-gemma4','sglang-glm53-flash','sglang-qwen4'):
            with self.subTest(engine=engine),patch.dict(sys.modules):
                name='probe_route_'+secrets.token_hex(8)
                package(name,self.source,ROOT/'engines'/engine/'flashalign_sglang')
                middleware=importlib.import_module(name+'.http').Middleware(no_fallthrough)
                transport=httpx.ASGITransport(app=middleware)
                async with httpx.AsyncClient(transport=transport,base_url='http://engine/v1/') as client:
                    capabilities=await client.get('flashalign/capabilities')
                    self.assertEqual(capabilities.status_code,200)
                    self.assertEqual(capabilities.json()['schema'],'flashalign.capabilities.v1')
                    status=await client.get('flashalign/status')
                    self.assertEqual(status.status_code,200)
                    self.assertEqual(status.json()['schema'],'flashalign.runtime-status.v1')
                    admin=await client.post('flashalign/operations',json={'action':'recover'})
                    self.assertEqual(admin.status_code,405)
                async with httpx.AsyncClient(transport=transport,base_url='http://engine/') as client:
                    self.assertEqual((await client.get('flashalign/capabilities')).status_code,200)
                    self.assertEqual((await client.get('flashalign/status')).status_code,200)

    async def test_active_and_cancel_counters_count_distinct_unfinished_work_only(self):
        first=SimpleNamespace(rid='synthetic-first',finished=lambda:False)
        second=SimpleNamespace(rid='synthetic-second',finished=lambda:False)
        complete=SimpleNamespace(rid='synthetic-complete',finished=lambda:True)
        self.scheduler.waiting_queue=[first]
        self.scheduler.running_batch.reqs=[first,second,complete]
        self.scheduler.chunked_req=second
        self.scheduler.abort_request=lambda request:setattr(first,'finished',lambda:True)
        self.assertEqual(len(self.worker.active_requests(self.scheduler)),2)
        self.worker.dispatch_abort(self.scheduler,SimpleNamespace(rid='synthetic-first',abort_all=False))
        self.api._CONTROLLER=self.controller
        status=await self.api.public_status()
        self.assertEqual(status['scheduler_active_requests'],1)
        self.assertEqual(status['active_requests'],1)
        self.assertEqual(status['cancellation_matches_total'],1)
        self.assertNotIn('synthetic-first',json.dumps(status))

    async def test_grammar_and_result_queues_pin_generation_during_finalization(self):
        selection,lease=self.controller.bind(self.api_gate.Selection('initial','',1.0))
        salt=self.api_runtime.registry()[1].cache_salt(selection,None)
        lease.close()
        pending=SimpleNamespace(rid='grammar-request',cache_salt=salt,finished=lambda:False)
        finishing=SimpleNamespace(rid='result-request',cache_salt=salt,finished=lambda:True)
        self.scheduler.grammar_manager=SimpleNamespace(grammar_queue=[pending])
        self.scheduler.result_queue=[(SimpleNamespace(reqs=[finishing]),None)]
        self.assertEqual(len(self.worker.active_requests(self.scheduler)),1)
        self.assertEqual(self.worker.references(self.scheduler,selection.name),2)

    async def test_staged_receiver_requests_and_duplicate_ids_keep_generation_refs(self):
        selected,lease=self.controller.bind(self.api_gate.Selection('initial','',1.0))
        salt=self.api_runtime.registry()[1].cache_salt(selected,None)
        lease.close()
        owner=SimpleNamespace(rid='same-id',cache_salt=salt,finished=lambda:False)
        off=SimpleNamespace(rid='same-id',cache_salt=None,finished=lambda:False)
        self.scheduler.waiting_queue=[off]
        self.scheduler.dllm_manager=SimpleNamespace(waiting_queue=[owner],staging_queue=[])
        self.scheduler.mm_receiver=SimpleNamespace(waiting_list=[SimpleNamespace(recv_req=owner)])
        self.scheduler.hisparse_coordinator=SimpleNamespace(ack_staging_queue=[SimpleNamespace(req=owner)])
        self.scheduler.collect_inflight_reqs=lambda:[owner]
        self.assertEqual(len(self.worker.active_requests(self.scheduler)),2)
        self.assertEqual(self.worker.references(self.scheduler,selected.name),1)


if __name__ == '__main__':
    unittest.main()
