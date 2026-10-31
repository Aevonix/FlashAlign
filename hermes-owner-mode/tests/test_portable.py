# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx
from test_gate_mode import Authority, GateBase, OWN
from support import Engine, load_cli
from owner_mode.contacts import ApprovalStore
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.bridge import Bridge
from owner_mode.proxy import ProviderProxy
from owner_mode import hermes_ingress
from owner_mode.identity import IngressFacts, TurnFacts


class Contacts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'approvals.json'
        self.store=ApprovalStore(self.path)

    def test_indefinite_approval_survives_reload_and_is_exactly_scoped(self):
        self.store.set_contact('whatsapp','account-a','15550000011@s.whatsapp.net',True)
        grant=ApprovalStore(self.path).approved('whatsapp','account-a','+15550000011')
        self.assertTrue(grant)
        self.assertIsNone(self.store.approved('whatsapp','account-b','+15550000011'))
        self.assertIsNone(self.store.approved('rcs','account-a','+15550000011'))
        self.assertNotIn('expires',self.store.read()['grants'][0])
        self.assertEqual(self.path.stat().st_mode & 0o777,0o600)

    def test_revocation_and_reapproval_do_not_revive_old_capabilities(self):
        self.store.set_contact('rcs','default','+15550000011',True)
        grant=self.store.approved('rcs','default','+15550000011')
        self.store.set_contact('rcs','default','+15550000011',False)
        self.assertFalse(self.store.active(grant))
        self.store.set_contact('rcs','default','+15550000011',True)
        self.assertFalse(self.store.active(grant))
        self.assertNotEqual(grant,self.store.approved('rcs','default','+15550000011'))

    def test_default_on_off_and_named_tool_overrides(self):
        self.assertTrue(self.store.tool_enabled('delegate_task'))
        self.store.update(lambda s:s['tools'].update(default=False))
        self.assertFalse(self.store.tool_enabled('delegate_task'))
        self.store.set_tool('delegate_task',True)
        self.assertTrue(self.store.tool_enabled('delegate_task'))
        self.assertFalse(self.store.tool_enabled('unknown'))

    def test_unreadable_or_malformed_store_is_not_approval(self):
        self.path.write_text('{}');self.path.chmod(0o600)
        with self.assertRaises(ValueError):self.store.approved('rcs','default','+15550000011')
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):self.store.read()

    def test_native_identity_needs_ingress_and_rechecks_revocation(self):
        self.store.set_contact('rcs','default','+15550000011',True)
        authority=NativeAuthority(self.path,VerifiedTurn)
        incoming=IngressFacts(channel='rcs',native=True,internal=False,chat_type='dm',
            chat_id='chat',sender_id='+15550000011',message_id='m',text=OWN,plain_text=True,allowlisted=True)
        turn=TurnFacts(channel='rcs',sender_id='+15550000011',chat_id='chat',chat_type='dm',
            message_id='m',text=OWN,route_ok=True,synthetic=False)
        with mock.patch.object(hermes_ingress,'turn_facts',return_value=turn):
            self.assertIsNone(authority.verify_turn(session_id='s',turn_id='t'))
        with mock.patch.object(hermes_ingress,'ingress_facts',return_value=incoming):authority.hear()
        with mock.patch.object(hermes_ingress,'turn_facts',return_value=turn):
            proof=authority.verify_turn(session_id='s',turn_id='t')
            self.assertTrue(proof)
            self.assertIsNone(authority.verify_turn(session_id='s',turn_id='t'))
        self.assertTrue(authority.is_active(proof))
        self.store.set_contact('rcs','default','+15550000011',False)
        self.assertFalse(authority.is_active(proof))


class ToolScope(GateBase):
    def setUp(self):
        super().setUp()
        self.flags={'delegate_task':True,'blocked':False,'allowed':True}
        self.policy.tool_policy=lambda n:self.flags.get(n,True)
        self.policy.begin(session_id='s',turn_id='t',user_message=OWN,platform='whatsapp')
        self.send(self.body())

    def test_disabled_tool_model_calls_are_off_and_parent_recovers(self):
        def operation(args):
            self.assertEqual(args,{'unchanged':True})
            return self.send(self.body())[1]['effective']
        self.assertEqual(self.runtime.execute_tool({'unchanged':True},operation,tool_name='blocked'),'off')
        self.assertEqual(self.send(self.body())[1]['effective'],'owner-test')

    def test_tool_exceptions_and_async_results_pass_through(self):
        with self.assertRaisesRegex(RuntimeError,'original'):
            self.runtime.execute_tool({},lambda _:(_ for _ in ()).throw(RuntimeError('original')),tool_name='blocked')
        async def operation(_):return self.send(self.body())[1]['effective']
        self.assertEqual(asyncio.run(self.runtime.execute_tool({},operation,tool_name='blocked')),'off')
        self.assertEqual(self.send(self.body())[1]['effective'],'owner-test')

    def test_nested_enabled_tool_cannot_escape_disabled_outer_tool(self):
        def outer(_):return self.runtime.execute_tool({},lambda _:self.send(self.body())[1]['effective'],tool_name='allowed')
        self.assertEqual(self.runtime.execute_tool({},outer,tool_name='blocked'),'off')

    def test_disabled_spawn_and_dynamic_child_revocation(self):
        self.flags['delegate_task']=False
        self.policy.adopt(parent_session_id='s',parent_turn_id='t',child_session_id='child')
        self.policy.begin(session_id='child',turn_id='ct',user_message='',parent_session_id='s')
        self.assertFalse(self.policy.is_owner_work('child','ct'))
        self.flags['delegate_task']=True
        self.policy.adopt(parent_session_id='s',parent_turn_id='t',child_session_id='child')
        self.policy.begin(session_id='child',turn_id='ct2',user_message='',parent_session_id='s')
        self.assertTrue(self.policy.is_owner_work('child','ct2'))
        self.flags['delegate_task']=False
        self.assertFalse(self.policy.is_owner_work('child','ct2'))
        self.assertTrue(self.policy.owner_turn('s','t'))
        self.assertFalse(self.policy.owner_turn('s','t','delegate_task'))


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.key=root/'key';self.key.write_bytes(os.urandom(32));self.key.chmod(0o600)
        self.store=ApprovalStore(root/'grants.json');self.store.set_contact('whatsapp','default','+15550000011',True)
        self.engine=Engine(self.key.read_bytes());self.addCleanup(self.engine.close)
        self.routes={'head':{'upstream':self.engine.base_url,'profile':'owner-test','owner_mode_scale':1.0}}
        self.bridge=Bridge(self.store,self.key,self.routes)
        self.proxy=ProviderProxy(self.bridge.policy,self.routes,self.key,bridge=self.bridge)
        self.proxy.start();self.addCleanup(self.proxy.close)
        self.url='http://127.0.0.1:'+str(self.proxy.port)

    def claim(self):
        return self.bridge.command('begin',{'session':'s','run':'t','channel':'whatsapp',
            'account':'default','contact':'+15550000011','text':OWN})['capability']

    def test_control_auth_and_end_to_end_signed_model_request(self):
        with httpx.Client() as client:
            self.assertEqual(client.post(self.url+'/_owner/begin',json={}).status_code,403)
            cap=self.claim()
            body={'model':'test','messages':[{'role':'user','content':OWN}]}
            response=client.post(self.url+'/head/v1/chat/completions',json=body,headers={'X-Owner-Mode-Run':cap})
            self.assertEqual(response.headers['X-Owner-Mode-Effective'],'owner-test')
            self.store.set_contact('whatsapp','default','+15550000011',False)
            response=client.post(self.url+'/head/v1/chat/completions',json=body,headers={'X-Owner-Mode-Run':cap})
            self.assertEqual(response.headers['X-Owner-Mode-Effective'],'off')

    def test_child_flags_and_end_revoke_capability(self):
        cap=self.claim()
        self.assertTrue(self.bridge.ticket(cap,{'model':'test','messages':[{'role':'user','content':OWN}]},'head'))
        self.store.set_tool('sessions_spawn',False)
        body={'parent':cap,'session':'child','run':'ct','tool':'sessions_spawn'}
        self.assertEqual(self.bridge.command('adopt',body),{'mode':'off'})
        self.store.set_tool('sessions_spawn',True)
        self.assertEqual(self.bridge.command('adopt',body)['mode'],'owner')
        self.bridge.command('end',{'capability':cap})
        self.assertEqual(self.bridge.ticket(cap,{'messages':[]},'head'),'')

    def test_tool_scope_is_stock_and_cannot_leak_owner_mode_to_a_child(self):
        cap=self.claim()
        request={'model':'test','messages':[{'role':'user','content':OWN}]}
        self.assertTrue(self.bridge.ticket(cap,request,'head'))
        self.store.set_tool('disabled',False)
        self.bridge.command('tool_begin',{'capability':cap,'call':'call-1','tool':'disabled'})
        self.assertEqual(self.bridge.ticket(cap,request,'head'),'')
        self.assertEqual(self.bridge.command('adopt',{'parent':cap,'session':'child',
                         'run':'nested','tool':'sessions_spawn'}),{'mode':'off'})
        self.bridge.command('tool_end',{'capability':cap,'call':'call-1'})
        self.assertTrue(self.bridge.ticket(cap,request,'head'))

    def test_native_claim_needs_a_bound_rendered_message(self):
        claim={'session':'s','run':'media','messages':[{'id':'native-1','channel':'whatsapp',
                'account':'default','contact':'+15550000011'}]}
        cap=self.bridge.command('claim',claim)['capability']
        content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]
        request={'model':'test','messages':[{'role':'user','content':content}]}
        self.assertEqual(self.bridge.ticket(cap,request,'head'),'')
        self.assertEqual(self.bridge.command('bind',{'capability':cap,'content':content}),{'mode':'owner'})
        altered={'model':'test','messages':[{'role':'user','content':[{'type':'text','text':'Other content'}]}]}
        self.assertEqual(self.bridge.ticket(cap,altered,'head'),'')
        self.assertTrue(self.bridge.ticket(cap,request,'head'))
        with self.assertRaisesRegex(ValueError,'changed'):
            self.bridge.command('bind',{'capability':cap,'content':'Replaced'})
        self.store.set_contact('whatsapp','default','+15550000011',False)
        self.assertEqual(self.bridge.ticket(cap,request,'head'),'')

    def test_native_message_cannot_be_claimed_by_another_run(self):
        claim={'session':'s','run':'first','messages':[{'id':'native-1','channel':'whatsapp',
                'account':'default','contact':'+15550000011'}]}
        cap=self.bridge.command('claim',claim)['capability']
        self.bridge.command('end',{'capability':cap})
        with self.assertRaisesRegex(ValueError,'already claimed'):
            self.bridge.command('claim',{**claim,'run':'second'})

    def test_attachment_binding_can_exceed_one_megabyte(self):
        claim={'session':'s','run':'large-media','messages':[{'id':'large-native','channel':'whatsapp',
                'account':'default','contact':'+15550000011'}]}
        cap=self.bridge.command('claim',claim)['capability']
        content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+'A'*(1024*1024)}}]
        with httpx.Client() as client:
            response=client.post(self.url+'/_owner/bind',json={'capability':cap,'content':content},
                                 headers={'Authorization':'Bearer '+self.key.read_bytes().hex()})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json(),{'mode':'owner'})

    def test_openclaw_adapter_wire_contract(self):
        path=Path(__file__).resolve().parents[2]/'openclaw-owner-mode'/'integration-test.mjs'
        result=subprocess.run(['node',str(path),self.url,str(self.key)],capture_output=True,text=True,timeout=25)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_openclaw_canonical_parameter_defaults_are_call_local(self):
        path=Path(__file__).resolve().parents[2]/'openclaw-owner-mode'/'parameters-test.mjs'
        result=subprocess.run(['node',str(path)],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_openclaw_child_continuation_keeps_authority_until_native_completion(self):
        path=Path(__file__).resolve().parents[2]/'openclaw-owner-mode'/'continuation-test.mjs'
        process=subprocess.Popen(['node',str(path),self.url,str(self.key)],
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(),'REVOKE_CONTACT')
            self.store.set_contact('whatsapp','default','+15550000011',False)
            output,error=process.communicate('revoked\n',timeout=25)
            self.assertEqual(process.returncode,0,output+error)
            self.assertIn('OPENCLAW_CONTINUATION_CONTRACT_OK',output)
        finally:
            if process.poll() is None:
                process.kill();process.communicate()


class WizardTests(unittest.TestCase):
    def load(self):
        return load_cli('setup')

    def test_detect_and_model_compatibility_without_launching_harness(self):
        wizard=self.load()
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);p=home/'.hermes/config.yaml';p.parent.mkdir()
            p.write_text(json.dumps({'model':{'base_url':'http://127.0.0.1:9999/v1'}}))
            result=wizard.discover(home,which=lambda _:None,probe=lambda _:['glm-5.3','unknown'])
            self.assertTrue(result['hermes']['detected'])
            self.assertEqual(result['hermes']['models'][0]['profile'],'glm53-full-v1')
            self.assertIsNone(result['hermes']['models'][1]['profile'])

    def test_setup_keeps_other_settings_and_choice_of_tool_default(self):
        wizard=self.load()
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);p=home/'.hermes/config.yaml';p.parent.mkdir()
            p.write_text(json.dumps({'unrelated':{'keep':True}}))
            result=wizard.provision(home,['hermes','openclaw'],'http://127.0.0.1:9871/v1','glm-5.3',
                'glm53-full-v1',[('rcs','default','+15550000011')],False)
            self.assertEqual(json.loads(p.read_text())['unrelated'],{'keep':True})
            store=ApprovalStore(result['approval_store'])
            self.assertTrue(store.approved('rcs','default','+15550000011'))
            self.assertFalse(store.tool_enabled('delegate_task'))
            cfg=json.loads((home/'.openclaw/openclaw.json').read_text())
            self.assertEqual(cfg['agents']['defaults']['model']['primary'],'owner-mode/glm-5.3')
            self.assertEqual(set(cfg['plugins']['entries']),{'owner-mode'})
            self.assertEqual(cfg['models']['providers']['owner-mode']['api'],'openai-completions')
            license_text=(Path(__file__).resolve().parents[1]/'LICENSE').read_text()
            for package in [home/'.hermes/plugins/owner-mode',home/'.openclaw/extensions/owner-mode']:
                self.assertEqual((package/'LICENSE').read_text(),license_text)
                self.assertIn('AGPL-3.0-only',(package/'NOTICE.md').read_text())


if __name__=='__main__':unittest.main()
