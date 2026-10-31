# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A custom host uses real bridge HTTP, contact grants and tool scopes."""
from pathlib import Path
import secrets
import tempfile
import unittest

from test_setup import wizard
from support import Engine
from owner_mode.bridge import Bridge
from owner_mode.contacts import ApprovalStore
from owner_mode.harness_client import HarnessClient
from owner_mode.proxy import ProviderProxy


class CustomHarness(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        self.control=root/'control';self.key=root/'engine'
        for key in (self.control,self.key):key.write_bytes(secrets.token_bytes(32));key.chmod(0o600)
        self.store=ApprovalStore(root/'approvals.json')
        self.store.set_contact('rcs','default','+15550000011',True)
        self.engine=Engine(self.key.read_bytes());self.addCleanup(self.engine.close)
        routes={'head':{'upstream':self.engine.base_url,'profile':'owner-test'}}
        bridge=Bridge(self.store,self.control,routes)
        self.proxy=ProviderProxy(bridge.policy,routes,self.key,bridge=bridge)
        self.proxy.start();self.addCleanup(self.proxy.close)
        self.client=HarnessClient('http://127.0.0.1:'+str(self.proxy.port),self.control)
        self.addCleanup(self.client.close)

    def claim(self,contact='+15550000011'):
        return self.client.claim_native(session='session',run='run',messages=[{
            'id':'native-message','channel':'rcs','account':'default','contact':contact}])

    def ask(self,cap,content='Explain a loop.'):
        return self.client.chat_response(cap,[{'role':'user','content':content}],model='bench').headers['X-Owner-Mode-Effective']

    def test_media_keeps_verified_status_and_revocation_applies_next_call(self):
        content=[{'type':'text','text':'Explain this picture.'},
                 {'type':'image_url','image_url':{'url':'data:image/png;base64,YQ=='}}]
        cap=self.claim()['capability']
        self.assertEqual(self.client.bind_content(cap,content)['mode'],'owner')
        self.assertEqual(self.ask(cap,content),'owner-test')
        self.store.set_contact('rcs','default','+15550000011',False)
        self.assertEqual(self.ask(cap,content),'off')

    def test_tool_scope_and_run_end_revoke_access(self):
        cap=self.claim()['capability'];self.client.bind_content(cap,'Explain a loop.')
        self.assertEqual(self.ask(cap),'owner-test')
        self.store.set_tool('blocked-tool',False)
        with self.client.tool_scope(cap,call='tool-call',tool='blocked-tool'):
            self.assertEqual(self.ask(cap),'off')
        self.assertEqual(self.ask(cap),'owner-test')
        self.client.end(cap)
        self.assertEqual(self.ask(cap),'off')

    def test_unapproved_sender_has_no_capability(self):
        self.assertNotIn('capability',self.claim('+15550000012'))
        self.assertEqual(self.ask(''),'off')

    def test_verified_session_supports_overlapping_runs_and_rechecks_approval(self):
        caps = [self.client.claim_verified_session(session='isolated-native-session', run=run,
                channel='rcs', account='default', contact='+15550000011')['capability']
                for run in ['first', 'second']]
        self.assertNotEqual(*caps)
        for cap in caps:
            self.assertEqual(self.ask(cap), 'off')
            self.client.bind_content(cap, 'Explain a loop.')
            self.assertEqual(self.ask(cap), 'owner-test')
        self.store.set_contact('rcs', 'default', '+15550000011', False)
        for cap in caps:
            self.assertEqual(self.ask(cap), 'off')

    def test_control_key_cannot_be_sent_to_a_remote_origin(self):
        with self.assertRaises(ValueError):
            HarnessClient('https://example.com',self.control)


if __name__=='__main__':unittest.main()
