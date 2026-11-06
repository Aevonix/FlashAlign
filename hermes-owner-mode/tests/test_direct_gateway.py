# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real supervised loopback gateway; synthetic upstream, no user deployment."""
import json
from email.message import Message
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest import mock

import httpx

from test_setup import wizard, make_profile
from support import Engine
from owner_mode.automation import run_plan
from owner_mode.client import Client
from owner_mode.contacts import ApprovalStore
from owner_mode.direct_gateway import prepare_gateway, activate_gateway, stop_gateway, read_private
from owner_mode.engine_setup import private_json, process_identity
from owner_mode.setup_runtime import SetupError
from owner_mode import direct_gateway
from owner_mode.proxy import loopback_request


class LocalRequestBoundary(unittest.TestCase):
    def test_ambiguous_authority_and_origin_headers_are_rejected(self):
        for pairs in ([('Host', '127.0.0.1:8000'), ('Host', 'localhost:8000')],
                      [('Host', '127.0.0.1:8000'), ('Origin', 'http://127.0.0.1:8000'),
                       ('Origin', 'http://127.0.0.1:8000')],
                      [('Host', 'user@127.0.0.1:8000')],
                      [('Host', '[::1]:invalid')],
                      [('Host', '127.0.0.1:8000'), ('Origin', 'http://[::1]:8000')]):
            with self.subTest(headers=pairs):
                headers = Message()
                for key, value in pairs:
                    headers[key] = value
                self.assertFalse(loopback_request(headers))


@unittest.skipUnless(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'), 'Linux managed gateway')
class DirectGateway(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.home=self.root/'home';self.home.mkdir()
        self.key=self.root/'engine.key';self.key.write_bytes(b'k'*32);self.key.chmod(0o600)
        self.engine=Engine(b'k'*32,profiles=('custom-owner',));self.addCleanup(self.engine.close)
        self.deployment=prepare_gateway(self.home,base_url=self.engine.base_url,model='bench',
                                        profile='custom-owner',key_file=self.key)
        self.addCleanup(lambda:stop_gateway(self.deployment))

    def activate(self):
        self.deployment=activate_gateway(self.deployment,timeout=8)
        return self.deployment

    def request(self,**body):
        return httpx.post(self.deployment['base_url']+'/chat/completions',json={
            'model':'bench','messages':[{'role':'user','content':'Hello'}],**body},timeout=4)

    def test_prepare_is_idle_and_runtime_is_private(self):
        self.assertEqual(self.deployment['status'],'prepared')
        self.assertFalse((Path(self.deployment['state'])/'running.json').exists())
        self.assertEqual(self.engine.records,[])
        self.assertEqual(prepare_gateway(self.home,base_url=self.engine.base_url,model='bench',
                        profile='custom-owner',key_file=self.key),self.deployment)
        self.assertTrue(ApprovalStore(self.deployment['approval_store']).always_on())
        self.assertEqual(read_private(self.deployment['config'])['key_file'],str(self.key))

    def test_unsigned_curl_and_tool_off_hints_are_always_on(self):
        self.activate()
        before=len(self.engine.records)
        for body in ({},{'owner_mode':'off','owner_mode_scale':0,'tool_choice':'none'}):
            response=self.request(**body)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers['X-Owner-Mode-Effective'],'custom-owner')
            self.assertEqual(self.engine.records[-1]['auth_reason'],'valid')
            self.assertEqual(self.engine.records[-1]['body']['owner_mode'],'custom-owner')
        self.assertEqual(len(self.engine.records),before+2)
        self.assertEqual(self.engine.records[-1]['body']['tool_choice'],'none')

    def test_foreign_browser_origins_and_rebound_hosts_never_reach_upstream(self):
        self.activate()
        before = len(self.engine.records)
        port = self.deployment['base_url'].split(':')[2].split('/')[0]
        for headers in ({'Origin': 'https://attacker.invalid'}, {'Origin': 'null'},
                        {'Origin': 'https://127.0.0.1:' + port},
                        {'Host': 'attacker.invalid:' + port}, {'Host': '[::1'},
                        {'Origin': 'http://localhost:' + port},
                        {'Sec-Fetch-Site': 'cross-site'}):
            with self.subTest(headers=headers):
                response = httpx.post(self.deployment['base_url'] + '/chat/completions',
                    json={'model': 'bench', 'messages': []}, headers=headers, timeout=4)
                self.assertEqual(response.status_code, 403)
                self.assertEqual(len(self.engine.records), before)

    def test_same_origin_loopback_ipv4_hostname_and_ipv6_authorities_work(self):
        self.activate()
        port = self.deployment['base_url'].split(':')[2].split('/')[0]
        for authority in ('127.0.0.1:' + port, 'localhost:' + port, '[::1]:' + port):
            with self.subTest(authority=authority):
                response = httpx.post(self.deployment['base_url'] + '/chat/completions',
                    json={'model': 'bench', 'messages': []}, timeout=4,
                    headers={'Host': authority, 'Origin': 'http://' + authority,
                             'Sec-Fetch-Site': 'same-origin'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'custom-owner')

    def test_idempotent_activation_stop_and_restart_preserve_upstream(self):
        first=self.activate()
        marker=Path(first['state'])/'running.json'
        identity=read_private(marker)['processes'][0]
        second=activate_gateway(first,timeout=4)
        self.assertEqual(first['base_url'],second['base_url'])
        self.assertEqual(read_private(marker)['processes'][0],identity)
        self.assertEqual(stop_gateway(first)['status'],'stopped')
        self.assertIsNone(process_identity(identity['pid']))
        self.assertFalse(marker.exists())
        raw=httpx.post(self.engine.base_url+'/chat/completions',json={'model':'bench','messages':[]},timeout=3)
        self.assertEqual(raw.status_code,200)
        self.assertEqual(raw.headers['X-Owner-Mode-Effective'],'off')
        self.activate()
        self.assertNotEqual(read_private(marker)['processes'][0],identity)
        self.assertEqual(self.request().headers['X-Owner-Mode-Effective'],'custom-owner')

    def test_missing_profile_confirmation_fails_and_cleans_only_gateway(self):
        self.engine.key=b'x'*32
        self.engine.server.handle_error=lambda *_:None
        with self.assertRaisesRegex(SetupError,'configured profile'):
            self.activate()
        self.assertFalse((Path(self.deployment['state'])/'running.json').exists())
        self.assertEqual(len(self.engine.records),1)
        self.assertEqual(httpx.get(self.engine.base_url+'/models',timeout=3).status_code,200)

    def test_missing_persisted_policy_never_downgrades_to_stock(self):
        self.activate();before=len(self.engine.records)
        Path(self.deployment['approval_store']).unlink()
        response=self.request()
        self.assertEqual(response.status_code,502)
        self.assertEqual(len(self.engine.records),before)

    def test_only_the_configured_model_is_advertised_and_forwarded(self):
        self.activate();before=len(self.engine.records)
        response=self.request(model='another-model')
        self.assertEqual(response.status_code,502)
        self.assertEqual(len(self.engine.records),before)
        listing=httpx.get(self.deployment['base_url']+'/models',timeout=3).json()
        self.assertEqual([row['id'] for row in listing['data']],['bench'])

    def test_disabled_deployment_is_not_reenabled_by_activation(self):
        ApprovalStore(self.deployment['approval_store']).update(lambda value:value.update(enabled=False))
        with self.assertRaisesRegex(SetupError,'disabled'):
            self.activate()
        self.assertFalse((Path(self.deployment['state'])/'running.json').exists())
        self.assertEqual(self.engine.records,[])

    def test_changed_runtime_fails_before_launch(self):
        path=Path(self.deployment['runtime'])/'proxy.py';path.write_text('# changed\n')
        with self.assertRaisesRegex(SetupError,'runtime changed'):
            self.activate()
        self.assertFalse((Path(self.deployment['state'])/'running.json').exists())

    def test_existing_upstream_api_credential_stays_in_private_gateway_state(self):
        self.engine.required_authorization='Bearer example-upstream-token'
        deployment=prepare_gateway(self.home,base_url=self.engine.base_url,model='bench',
            profile='custom-owner',key_file=self.key,api_key='example-upstream-token')
        self.addCleanup(lambda:stop_gateway(deployment))
        cfg=read_private(deployment['config'])
        self.assertNotIn('example-upstream-token',Path(deployment['config']).read_text())
        token=Path(cfg['routes']['head']['api_key_file'])
        self.assertEqual(token.stat().st_mode & 0o777,0o600)
        self.assertEqual(token.read_text(),'example-upstream-token')
        activated=activate_gateway(deployment,timeout=8)
        response=httpx.post(activated['base_url']+'/chat/completions',json={
            'model':'bench','messages':[]},timeout=4)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'],'custom-owner')

    def test_declared_template_rules_are_bound_and_applied_after_gateway_staging(self):
        for declaration in ({'kind':'chat_template_reasoning_v1','efforts':['low','max']},
                            {'kind':'gemma_boolean_thinking_v1'}):
            with self.subTest(declaration=declaration):
                deployment=prepare_gateway(self.home,base_url=self.engine.base_url,model='bench',
                    profile='custom-owner',key_file=self.key,request_compatibility=declaration)
                self.addCleanup(lambda d=deployment:stop_gateway(d))
                rule=read_private(deployment['config'])['routes']['head']['request_compatibility']
                self.assertEqual(rule,{**declaration,'model':'bench'})
                self.assertFalse((Path(deployment['state'])/'running.json').exists())
                live=activate_gateway(deployment,timeout=8)
                for effort in ('none','max'):
                    options={'preserve_thinking':True,'other_option':'kept'}
                    if declaration['kind']=='gemma_boolean_thinking_v1':options['enable_thinking']=True
                    response=httpx.post(live['base_url']+'/chat/completions',json={
                        'model':'bench','messages':[{'role':'user','content':'Hello'}],
                        'reasoning_effort':effort,'chat_template_kwargs':options},timeout=4)
                    self.assertEqual(response.headers['X-Owner-Mode-Effective'],'custom-owner')
                    sent=self.engine.records[-1]['body'];self.assertEqual(sent['chat_template_kwargs']['other_option'],'kept')
                    self.assertTrue(sent['chat_template_kwargs']['preserve_thinking'])
                    self.assertIs(sent['chat_template_kwargs']['enable_thinking'],effort!='none')
                    if declaration['kind']=='gemma_boolean_thinking_v1' and effort=='max':
                        self.assertEqual(sent['reasoning_effort'],'max')
                self.assertEqual(stop_gateway(deployment)['status'],'stopped')

    def test_conflicting_compatibility_model_is_rejected_before_staging(self):
        with self.assertRaisesRegex(SetupError,'different model'):
            prepare_gateway(self.home,base_url=self.engine.base_url,model='bench',profile='custom-owner',
                key_file=self.key,request_compatibility={'kind':'gemma_boolean_thinking_v1','model':'other'})

    def test_remote_plan_apply_stages_then_activate_publishes_generic_endpoint(self):
        profile=make_profile(self.root/'profile');plan=self.root/'plan.json'
        private_json(plan,{'engine_only':True,'mode':'always_on','model':'bench','profile':str(profile),
                           'base_url':self.engine.base_url,'engine_key_file':str(self.key)})
        with mock.patch('owner_mode.automation.check_engine',side_effect=AssertionError('apply must not infer')):
            staged=run_plan(plan,self.home,{},wizard.provision,apply=True)
        self.addCleanup(lambda:stop_gateway(staged['gateway']))
        self.assertEqual(staged['status'],'gateway_prepared')
        self.assertEqual(self.engine.records,[])
        self.assertFalse((self.home/'.config/flashalign/client.json').exists())
        result=run_plan(plan,self.home,{},wizard.provision,activate=True)
        self.addCleanup(lambda:stop_gateway(result['gateway']))
        cfg=read_private(result['client_config'])
        self.assertTrue(cfg['gateway']);self.assertEqual(cfg['mode'],'always_on')
        self.assertNotIn('key_file',cfg);self.assertNotIn('api_key_file',cfg)
        self.assertNotEqual(cfg['base_url'],self.engine.base_url)
        self.assertEqual(result['base_url'],cfg['base_url'])
        with Client.from_config(result['client_config']) as client:
            self.assertIn('choices',client.chat([{'role':'user','content':'Hello'}]))
        self.assertEqual(self.engine.records[-1]['auth_reason'],'valid')

    def test_managed_plan_apply_stages_both_services_without_starting(self):
        profile=make_profile(self.root/'managed-profile');plan=self.root/'managed-plan.json'
        private_json(plan,{'engine_only':True,'mode':'always_on','model':'bench','profile':str(profile),
                           'engine':{'kind':'tensorfold','model_path':str(self.root/'checkpoint')}})
        engine={'status':'prepared','base_url':self.engine.base_url,'key_file':str(self.key)}
        with (mock.patch('owner_mode.automation.check_engine_setup',return_value={'kind':'tensorfold','metadata':{}}),
              mock.patch('owner_mode.automation.prepare_engine',return_value=engine),
              mock.patch('owner_mode.automation.activate_engine',side_effect=AssertionError('apply must not start engine')),
              mock.patch('owner_mode.automation.activate_gateway',side_effect=AssertionError('apply must not start gateway'))):
            result=run_plan(plan,self.home,{},wizard.provision,apply=True)
        self.assertEqual(result['engine']['status'],'prepared')
        self.assertEqual(result['gateway']['status'],'prepared')
        self.assertNotEqual(result['base_url'],engine['base_url'])
        self.assertFalse((self.home/'.config/flashalign/client.json').exists())
        self.assertEqual(self.engine.records,[])

    def test_verified_handoff_retires_only_the_advertised_gateway(self):
        profile=make_profile(self.root/'handoff-profile');plan=self.root/'handoff-plan.json'
        data={'engine_only':True,'mode':'always_on','model':'bench','profile':str(profile),
              'base_url':self.engine.base_url,'engine_key_file':str(self.key)}
        private_json(plan,data)
        active=run_plan(plan,self.home,{},wizard.provision,activate=True)
        self.addCleanup(lambda:stop_gateway(active['gateway']))
        old_marker=Path(active['gateway']['state'])/'running.json'
        self.assertTrue(old_marker.exists())
        private_json(plan,{**data,'mode':'verified'})
        verified=run_plan(plan,self.home,{},wizard.provision,activate=True)
        cfg=read_private(verified['client_config'])
        self.assertEqual(cfg['mode'],'verified');self.assertEqual(cfg['base_url'],self.engine.base_url)
        self.assertNotIn('gateway',cfg);self.assertFalse(old_marker.exists())
        self.assertEqual(verified['previous_gateway']['status'],'stopped')
        self.assertEqual(httpx.get(self.engine.base_url+'/models',timeout=3).status_code,200)

    def test_verified_apply_preserves_gateway_pointer_until_activation(self):
        profile=make_profile(self.root/'staged-handoff-profile');plan=self.root/'staged-handoff-plan.json'
        data={'engine_only':True,'mode':'always_on','model':'bench','profile':str(profile),
              'base_url':self.engine.base_url,'engine_key_file':str(self.key)}
        private_json(plan,data)
        active=run_plan(plan,self.home,{},wizard.provision,activate=True)
        self.addCleanup(lambda:stop_gateway(active['gateway']))
        client_path=Path(active['client_config']);before=client_path.read_bytes()
        marker=Path(active['gateway']['state'])/'running.json'
        private_json(plan,{**data,'mode':'verified'})
        staged=run_plan(plan,self.home,{},wizard.provision,apply=True)
        self.assertEqual(staged['status'],'gateway_transition_prepared')
        self.assertFalse(staged['applied'])
        self.assertEqual(client_path.read_bytes(),before)
        self.assertTrue(marker.exists())
        self.assertEqual(staged['next_command'][-1],'--activate')
        with Client.from_config(client_path) as client:
            self.assertIn('choices',client.chat([]))
        finished=run_plan(plan,self.home,{},wizard.provision,activate=True)
        self.assertEqual(read_private(finished['client_config'])['mode'],'verified')
        self.assertFalse(marker.exists())

    def test_failed_retirement_preserves_client_and_stops_replacement(self):
        self.engine.profiles.add('second-owner')
        profile=make_profile(self.root/'old-profile');plan=self.root/'replacement-plan.json'
        data={'engine_only':True,'mode':'always_on','model':'bench','profile':str(profile),
              'base_url':self.engine.base_url,'engine_key_file':str(self.key)}
        private_json(plan,data)
        previous=run_plan(plan,self.home,{},wizard.provision,activate=True)['gateway']
        self.addCleanup(lambda:stop_gateway(previous))
        client_path=self.home/'.config/flashalign/client.json';before=client_path.read_bytes()
        newer=make_profile(self.root/'new-profile',name='second-owner')
        private_json(plan,{**data,'profile':str(newer)})
        real_stop=direct_gateway.stop_gateway;stopped_new=[]
        def stop(deployment):
            if deployment['state']==previous['state']:
                return {'status':'incomplete'}
            stopped_new.append(deployment)
            return real_stop(deployment)
        with mock.patch.object(direct_gateway,'stop_gateway',side_effect=stop):
            with self.assertRaisesRegex(SetupError,'client configuration was preserved'):
                run_plan(plan,self.home,{},wizard.provision,activate=True)
        self.assertEqual(client_path.read_bytes(),before)
        self.assertEqual(len(stopped_new),1)
        self.assertFalse((Path(stopped_new[0]['state'])/'running.json').exists())
        self.assertTrue((Path(previous['state'])/'running.json').exists())

    def start_thread(self, target):
        results=[]
        def run():
            try:results.append(target())
            except BaseException as error:results.append(error)
        thread=threading.Thread(target=run);thread.start()
        return thread,results

    def test_lost_spawn_ack_uses_shared_intent_cleanup(self):
        units=direct_gateway.lifecycle._native_units
        original=units.start
        def lost_ack(request):
            original(request)
            raise OSError('simulated acknowledgement failure')
        with mock.patch.object(units,'start',side_effect=lost_ack):
            with self.assertRaisesRegex(SetupError,'launch failed'):
                self.activate()
        state=Path(self.deployment['state'])
        attempt=direct_gateway.lifecycle._activation(state)
        self.assertEqual(attempt['phase'],'stopped')
        self.assertFalse((state/'running.json').exists())
        self.assertTrue(all(row.get('all_children_reaped') and not row.get('supervisor_alive')
                            for row in direct_gateway.lifecycle._unit_states(attempt)))

    def test_concurrent_gateway_start_has_one_owned_command(self):
        units=direct_gateway.lifecycle._native_units
        original=units.start;entered=threading.Event();release=threading.Event()
        def start(request):
            entered.set()
            if not release.wait(8):raise AssertionError('Missing fixture release')
            return original(request)
        thread=None
        try:
            with mock.patch.object(units,'start',side_effect=start) as starts:
                thread,results=self.start_thread(lambda:activate_gateway(self.deployment,timeout=8))
                self.assertTrue(entered.wait(5))
                with self.assertRaisesRegex(SetupError,'pending'):
                    activate_gateway(self.deployment,timeout=2)
                release.set();thread.join(10)
            self.assertFalse(thread.is_alive());self.assertEqual(starts.call_count,1)
            self.assertEqual(results[0]['status'],'running')
            self.assertEqual(self.request().headers['X-Owner-Mode-Effective'],'custom-owner')
        finally:
            release.set()
            if thread is not None:thread.join(10)

    def test_stop_interrupts_readiness_and_stale_success_cannot_replace_new_owner(self):
        original=direct_gateway._ready_and_profile
        entered=threading.Event();release=threading.Event();calls=[];thread=None
        def readiness(*args):
            result=original(*args);calls.append(1)
            if len(calls)==1:
                entered.set()
                if not release.wait(10):raise AssertionError('Missing fixture release')
            return result
        try:
            with mock.patch.object(direct_gateway,'_ready_and_profile',side_effect=readiness):
                thread,results=self.start_thread(lambda:activate_gateway(self.deployment,timeout=15))
                self.assertTrue(entered.wait(8))
                state=Path(self.deployment['state'])
                old=direct_gateway.lifecycle._activation(state)['nonce']
                begin=time.monotonic()
                self.assertEqual(stop_gateway(self.deployment)['status'],'stopped')
                self.assertLess(time.monotonic()-begin,3)
                newer=activate_gateway(self.deployment,timeout=8)
                new=direct_gateway.lifecycle._activation(state)['nonce']
                self.assertNotEqual(old,new)
                release.set();thread.join(10)
            self.assertIsInstance(results[0],SetupError)
            self.assertEqual(direct_gateway.lifecycle._activation(state)['nonce'],new)
            self.assertEqual(read_private(state/'deployment.json'),newer)
            self.assertEqual(read_private(state/'running.json')['nonce'],new)
            self.assertEqual(self.request().headers['X-Owner-Mode-Effective'],'custom-owner')
        finally:
            release.set()
            if thread is not None:thread.join(10)

    def test_shared_supervision_sources_are_in_the_verified_runtime(self):
        config=read_private(self.deployment['config'])
        for name in ('engine_setup.py','engine_supervisor.py','native_remote.py'):
            self.assertIn(name,config['runtime_hashes'])
            self.assertEqual(direct_gateway.lifecycle.digest(Path(self.deployment['runtime'])/name),
                             config['runtime_hashes'][name])
        path=Path(self.deployment['runtime'])/'native_remote.py'
        path.write_text('# changed shared helper\n')
        with self.assertRaisesRegex(SetupError,'runtime changed'):
            self.activate()
        self.assertFalse((Path(self.deployment['state'])/'activation.json').exists())


if __name__=='__main__':unittest.main()
