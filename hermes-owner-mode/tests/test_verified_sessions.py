# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real bridge/proxy checks for independently claimed verified-contact runs."""
from pathlib import Path
import secrets
import tempfile
import unittest

import httpx

from test_setup import wizard
from support import Engine
from owner_mode.contacts import ApprovalStore
from owner_mode.proxy import ProviderProxy
from owner_mode.bridge import Bridge


class SessionClaims(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='contract-');root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        key,control=root/'engine.key',root/'control.key'
        for p in (key,control):p.write_bytes(secrets.token_bytes(32));p.chmod(0o600)
        self.store=ApprovalStore(root/'approvals.json')
        self.store.set_channels(['sms'])
        for contact in ('+15550000011','+15550000012'):self.store.set_contact('sms','default',contact,True)
        self.store.update(lambda x:x['tools'].update(default=True))
        self.engine=Engine(key.read_bytes());self.addCleanup(self.engine.close)
        routes={'head':{'upstream':self.engine.base_url,'profile':'owner-test'}}
        self.bridge=Bridge(self.store,control,routes)
        self.proxy=ProviderProxy(self.bridge.policy,routes,key,bridge=self.bridge);self.proxy.start();self.addCleanup(self.proxy.close)

    def claim(self,run='one',**overrides):
        body={'session':'isolated-native-session','run':run,'channel':'sms','account':'default','contact':'+15550000011',**overrides}
        return self.bridge.command('claim_session',body)

    def send(self,cap):
        r=httpx.post(self.proxy.base_url('head')+'/chat/completions',json={'model':'bench','messages':[{'role':'user','content':'fixture'}]},
                     headers={'X-Owner-Mode-Run':cap},timeout=5)
        self.assertEqual(r.status_code,200)
        return r.headers['X-Owner-Mode-Effective']

    def bind(self,cap):
        return self.bridge.command('bind',{'capability':cap,'content':'fixture'})

    def test_pending_requires_bind_and_end_revokes(self):
        result=self.claim();self.assertEqual(result['mode'],'pending');cap=result['capability']
        self.assertEqual(self.send(cap),'off');self.assertEqual(self.bind(cap)['mode'],'owner')
        self.assertEqual(self.send(cap),'owner-test')
        self.bridge.command('end',{'capability':cap});self.assertEqual(self.send(cap),'off')

    def test_simultaneous_runs_on_same_verified_session_get_distinct_capabilities(self):
        one=self.claim('one')['capability'];two=self.claim('two')['capability']
        self.assertNotEqual(one,two)
        for cap in (one,two):self.bind(cap);self.assertEqual(self.send(cap),'owner-test')

    def test_exact_tuple_and_source_kind_cannot_change(self):
        self.claim()
        with self.assertRaises(ValueError):self.claim(contact='+15550000012')
        with self.assertRaises(ValueError):
            self.bridge.command('claim',{'session':'isolated-native-session','run':'one','messages':[
                {'id':'cannot-switch','channel':'sms','account':'default','contact':'+15550000011'}]})
        with self.assertRaises(ValueError):
            self.bridge.command('begin',{'session':'isolated-native-session','run':'one','channel':'sms',
                'account':'default','contact':'+15550000011','text':'fixture'})
        parent=self.claim('parent')['capability'];self.bind(parent)
        with self.assertRaises(ValueError):
            self.bridge.command('adopt',{'parent':parent,'session':'isolated-native-session','run':'one','tool':'sessions_spawn'})
        old={'session':'legacy','run':'one','messages':[{'id':'native-id','channel':'sms','account':'default','contact':'+15550000011'}]}
        self.bridge.command('claim',old)
        with self.assertRaises(ValueError):self.claim(session='legacy')

    def test_adopted_child_survives_parent_end_but_rechecks_revocation_and_tool_setting(self):
        parent=self.claim('parent')['capability'];self.bind(parent)
        self.assertEqual(self.send(parent),'owner-test')
        self.bridge.command('tool_begin',{'capability':parent,'call':'spawn','tool':'sessions_spawn'})
        child=self.bridge.command('adopt',{'parent':parent,'session':'child-session','run':'child','tool':'sessions_spawn'})['capability']
        self.bridge.command('tool_end',{'capability':parent,'call':'spawn'})
        self.bridge.command('end',{'capability':parent})
        self.assertEqual(self.send(child),'owner-test')
        self.store.set_tool('sessions_spawn',False);self.assertEqual(self.send(child),'off')
        self.store.set_tool('sessions_spawn',True);self.assertEqual(self.send(child),'owner-test')
        self.store.set_contact('sms','default','+15550000011',False);self.assertEqual(self.send(child),'off')

    def test_revocation_applies_to_each_request_and_reapproval_does_not_resurrect_capability(self):
        cap=self.claim()['capability'];self.bind(cap);self.assertEqual(self.send(cap),'owner-test')
        self.store.set_contact('sms','default','+15550000011',False);self.assertEqual(self.send(cap),'off')
        self.store.set_contact('sms','default','+15550000011',True);self.assertEqual(self.send(cap),'off')
        newer=self.claim('newer')['capability'];self.bind(newer);self.assertEqual(self.send(newer),'owner-test')

    def test_unknown_contact_and_bad_identifiers_reject(self):
        self.assertEqual(self.claim(contact='+15550000013'),{'mode':'off'})
        for value in ('',None,False,' trailing '):
            with self.subTest(value=value),self.assertRaises(ValueError):self.claim(run=value)

    def test_old_native_message_replay_rule_is_preserved(self):
        old={'session':'legacy','run':'one','messages':[{'id':'native-id','channel':'sms','account':'default','contact':'+15550000011'}]}
        self.bridge.command('claim',old)
        with self.assertRaises(ValueError):self.bridge.command('claim',{**old,'run':'two'})


if __name__=='__main__':unittest.main()
