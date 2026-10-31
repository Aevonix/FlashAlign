# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Local terminal authority must originate in native submissions and remain revocable."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import json
from unittest import mock

from test_setup import wizard
from owner_mode.access_setup import choose_verified_contact
from owner_mode.bridge import Bridge
from owner_mode.contacts import ApprovalStore
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.plugin import Runtime
from owner_mode.terminal import TerminalAuthority, TerminalBinding, TerminalSessionStore, native_hermes_session


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.approvals = ApprovalStore(self.root/'approvals.json')
        self.approvals.set_channels(['whatsapp', 'local-terminal'])
        import os
        self.contact = 'uid:'+str(os.getuid())
        for harness in ('hermes','openclaw'):
            self.approvals.set_contact('local-terminal', harness, self.contact, True)
        self.now = 1000
        self.sessions = TerminalSessionStore(self.approvals, clock=lambda:self.now)

    def test_expiry_session_revocation_contact_revocation_and_channel_disable(self):
        row = self.sessions.create('hermes', ttl=60)
        self.assertTrue(self.sessions.active(row['id']))
        self.now += 60
        self.assertIsNone(self.sessions.active(row['id']))
        row = self.sessions.create('hermes')
        self.sessions.revoke(row['id'])
        self.assertIsNone(self.sessions.active(row['id']))
        row = self.sessions.create('hermes')
        self.approvals.set_contact('local-terminal','hermes',self.contact,False)
        self.assertIsNone(self.sessions.active(row['id']))
        self.approvals.set_contact('local-terminal','hermes',self.contact,True)
        self.assertIsNone(self.sessions.active(row['id']))
        row = self.sessions.create('hermes')
        self.approvals.set_channels(['whatsapp'])
        self.assertIsNone(self.sessions.active(row['id']))

    def test_native_submission_in_worker_thread_two_turns_and_revoke(self):
        row = self.sessions.create('hermes')
        observed = []
        holder = {}
        class Agent:
            def __init__(self, base_url=''):
                self.base_url=base_url; self.session_id='native-session'
            def run_conversation(self, user_message):
                runtime=holder['runtime']; tid='turn-'+str(len(observed))
                runtime.begin(session_id=self.session_id,turn_id=tid,platform='cli',user_message=user_message)
                observed.append(runtime.policy.is_owner_work(self.session_id,tid))
                body={'messages':[{'role':'user','content':user_message}]}
                runtime.policy.issue(body,session_id=self.session_id,turn_id=tid,profile='p',route='head')
                runtime.policy.adopt(parent_session_id=self.session_id,parent_turn_id=tid,child_session_id='child',tool_name='delegate_task')
                runtime.policy.begin(session_id='child',turn_id='child-turn',parent_session_id=self.session_id,user_message='')
                holder['child'] = runtime.policy.is_owner_work('child','child-turn')
                return observed[-1]
        class CLI:
            agent=None
            def chat(self,message):
                if self.agent is None:self.agent=Agent()
                thread=threading.Thread(target=lambda:self.agent.run_conversation(message))
                thread.start();thread.join()
        old=CLI.chat
        with native_hermes_session(self.sessions,row['id'],CLI,Agent) as binding:
            authority=TerminalAuthority(binding,VerifiedTurn)
            policy=Policy(authority=authority,allowed_channels=['local-terminal'],channel_policy=self.approvals.channel_enabled,
                          tool_policy=self.approvals.tool_enabled,trust={'history':'verified_owner','system_prompt':'host','tools':'host','turn_context':'host'})
            runtime=Runtime(policy,SimpleNamespace(base_url=lambda _:'http://localhost/test'))
            binding.runtime=runtime;holder['runtime']=runtime
            policy.begin(session_id='spoof',turn_id='spoof',platform='local-terminal',user_message='spoof')
            self.assertFalse(policy.is_owner_work('spoof','spoof'))
            cli=CLI();cli.chat('first');self.assertTrue(holder['child'])
            cli.chat('second')
            self.sessions.revoke(row['id'])
            self.assertFalse(policy.is_owner_work('child','child-turn'))
            cli.chat('third')
        self.assertEqual(observed,[True,True,False])
        self.assertIs(CLI.chat,old)

    def test_openclaw_terminal_claim_requires_live_session_and_exact_input(self):
        # Real wall time is used by the separate bridge authority.
        sessions=TerminalSessionStore(self.approvals)
        row=sessions.create('openclaw')
        key=self.root/'control.key';key.write_bytes(b'x'*32);key.chmod(0o600)
        bridge=Bridge(self.approvals,key,{'head':{'profile':'p'}})
        self.approvals.set_contact('whatsapp','default','+15550000011',True)
        native=bridge.command('claim_session',{'session':'native','run':'native',
            'channel':'whatsapp','account':'default','contact':'+15550000011'})
        self.assertEqual(bridge.command('terminal_render',{'capability':native['capability'],
            'session':'native','run':'native','content':'not terminal'}),{'mode':'off'})
        rendered='[Native host timestamp] typed words'
        result=bridge.command('terminal_claim',{'terminal_session':row['id'],'session':'s','run':'r','text':'typed words','rendered':rendered})
        cap=result['capability']
        with self.assertRaises(ValueError):bridge.command('bind',{'capability':cap,'content':'different'})
        self.assertEqual(bridge.command('terminal_render',{'capability':cap,'session':'wrong','run':'r','content':rendered}),{'mode':'off'})
        self.assertEqual(bridge.command('terminal_render',{'capability':'unknown','session':'s','run':'r','content':rendered}),{'mode':'off'})
        bridge.command('terminal_render',{'capability':cap,'session':'s','run':'r','content':rendered})
        with self.assertRaises(ValueError):bridge.command('terminal_render',{'capability':cap,'session':'s','run':'r','content':'changed render'})
        with self.assertRaises(ValueError):bridge.command('bind',{'capability':cap,'content':'typed words'})
        self.assertEqual(bridge.command('bind',{'capability':cap,'content':rendered})['mode'],'owner')
        body={'messages':[{'role':'user','content':rendered}]}
        self.assertTrue(bridge.ticket(cap,body,'head'))
        child=bridge.command('adopt',{'parent':cap,'session':'child','run':'cr','tool':'delegate_task'})
        self.assertEqual(child['mode'],'owner')
        bridge.command('end',{'capability':cap})
        self.assertTrue(bridge.ticket(child['capability'],body,'head'))
        sessions.revoke(row['id'])
        self.assertEqual(bridge.command('terminal_render',{'capability':cap,'session':'s','run':'r','content':rendered}),{'mode':'off'})
        self.assertFalse(bridge.ticket(cap,body,'head'))
        self.assertFalse(bridge.ticket(child['capability'],body,'head'))
        denied=bridge.command('terminal_claim',{'terminal_session':row['id'],'session':'s','run':'r2','text':'typed words','rendered':'typed words'})
        self.assertEqual(denied,{'mode':'off'})
        with self.assertRaises(ValueError):
            bridge.command('claim_session',{'session':'s','run':'fake','channel':'local-terminal','account':'openclaw','contact':self.contact})

    def test_wizard_lists_registered_choices_and_local_identity_is_not_typed(self):
        output=[]; answers=iter(['local-terminal','hermes'])
        value=choose_verified_contact(SimpleNamespace(line=output.append),['hermes'],self.approvals,ask=lambda _:next(answers))
        self.assertEqual(value,('local-terminal','hermes',self.contact))
        self.assertTrue(any('telegram' in line for line in output))
        with self.assertRaises(ValueError):
            choose_verified_contact(SimpleNamespace(line=lambda _:None),['hermes'],self.approvals,ask=lambda _:'uninstalled')

    def test_terminal_plan_does_not_schedule_a_gateway_restart(self):
        from owner_mode.automation import run_plan
        plan=self.root/'plan.json'
        plan.write_text(json.dumps({'harnesses':['hermes'],'base_url':'http://localhost/v1','model':'bench',
            'profile':'owner-test','engine_key_file':str(self.root/'key'),'tools_default':True,
            'contacts':[{'channel':'local-terminal','account':'hermes','contact':self.contact}]}))
        plan.chmod(0o600)
        details={'name':'owner-test','source':'engine'}
        with (mock.patch('owner_mode.automation.model_rows',return_value=[{'id':'bench'}]),
              mock.patch('owner_mode.automation.discover_profiles',return_value=([],[])),
              mock.patch('owner_mode.automation.engine_profiles',return_value=[]),
              mock.patch('owner_mode.automation.resolve_profile',return_value=details),
              mock.patch('owner_mode.automation.check_engine',return_value={'status':'active'}),
              mock.patch('owner_mode.automation.check_harness_contexts',return_value={})):
            result=run_plan(plan,self.root,{'hermes':{'detected':True,'runtime_compatible':True}},
                lambda *a,**k:{'terminal_harnesses':['hermes']},apply=True)
        self.assertEqual(result['restart'],{})
        self.assertEqual(result['activation'],'authorized native terminal launch required')


if __name__ == '__main__':unittest.main()
