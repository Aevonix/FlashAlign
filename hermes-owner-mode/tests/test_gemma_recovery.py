# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Gemma's declared boolean thinking control must honor native recovery off."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.model_metadata import compatible_request, checked_request_compatibility, declared_request_compatibility
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy
from owner_mode.setup_runtime import SetupError

MODEL='RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic'
DECLARATION={'kind':'gemma_boolean_thinking_v1'}
RULE={**DECLARATION,'model':MODEL}

class GemmaRecovery(unittest.TestCase):
    def test_none_overrides_inherited_true_without_mutating_images_or_other_kwargs(self):
        for initial in ({},{'enable_thinking':True},{'enable_thinking':False}):
            with self.subTest(initial=initial):
                body={'model':MODEL,'reasoning_effort':'none','chat_template_kwargs':{
                    **initial,'preserve_thinking':True,'other_option':'untouched'},
                    'messages':[{'role':'user','content':[{'type':'text','text':'Color?'},
                        {'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]}]}
                before=copy.deepcopy(body);result=compatible_request(body,RULE)
                self.assertEqual(body,before);self.assertNotIn('reasoning_effort',result)
                self.assertIs(result['chat_template_kwargs']['enable_thinking'],False)
                self.assertEqual(result['messages'],before['messages'])
                self.assertEqual(result['chat_template_kwargs']['preserve_thinking'],True)
                self.assertEqual(result['chat_template_kwargs']['other_option'],'untouched')

    def test_non_none_including_max_and_undeclared_models_are_unchanged(self):
        for effort in ('minimal','low','medium','high','xhigh','max',None):
            body={'model':MODEL,'reasoning_effort':effort,'chat_template_kwargs':{'enable_thinking':True}}
            self.assertIs(compatible_request(body,RULE),body)
        body={'model':'unrelated-model','reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':True}}
        self.assertIs(compatible_request(body,RULE),body)
        body={'model':MODEL,'reasoning_effort':'none','chat_template_kwargs':{'enable_thinking':True}}
        self.assertIs(compatible_request(body),body)

    def test_bad_kwargs_are_rejected_only_when_recovery_mapping_is_needed(self):
        with self.assertRaises(SetupError):compatible_request({'model':MODEL,'reasoning_effort':'none','chat_template_kwargs':[]},RULE)
        body={'model':MODEL,'reasoning_effort':'max','chat_template_kwargs':[]}
        self.assertIs(compatible_request(body,RULE),body)

    def test_declaration_is_explicit_and_model_bound(self):
        for engine, kinds in (('vllm-gemma4', ('gemma4','gemma4_text')), ('llamacpp', ('gemma4',))):
            metadata=json.loads((wizard.ROOT.parent/'engines'/engine/'backend.json').read_text())
            for kind in kinds:
                self.assertEqual(declared_request_compatibility(metadata,kind),DECLARATION)
            self.assertIsNone(declared_request_compatibility(metadata,'other'))
        for invalid in ({**DECLARATION,'efforts':['max']},{**RULE,'other':True},{**RULE,'model':''}):
            with self.subTest(invalid=invalid),self.assertRaises(SetupError):checked_request_compatibility(invalid)

    def test_provision_binds_the_declaration_to_the_selected_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary)
            with mock.patch.object(wizard,'free_port',return_value=48123):
                wizard.provision(home,['openclaw'],'http://127.0.0.1:48124/v1',MODEL,'profile',[],True,
                    model_info={'request_compatibility':DECLARATION})
            bridge=json.loads((home/'.config/owner-mode/openclaw-bridge.json').read_text())
            self.assertEqual(bridge['routes']['head']['request_compatibility'],RULE)

    def test_signed_and_stock_routes_use_same_recovery_mapping_after_ticket_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            key=Path(temporary)/'key';key.write_bytes(b'k'*32);key.chmod(0o600)
            class Authority:
                def verify_turn(self,session_id,turn_id,user_message,**_):return VerifiedTurn(session_id,turn_id,'sms',user_message)
                def is_active(self,proof):return True
            policy=Policy(authority=Authority(),allowed_channels=['sms']);policy.begin(session_id='s',turn_id='t',user_message='Hello',platform='sms')
            body={'model':MODEL,'messages':[{'role':'user','content':'Hello'}], 'reasoning_effort':'none',
                  'chat_template_kwargs':{'enable_thinking':True}}
            proxy=ProviderProxy(policy,{'head':{'upstream':'http://127.0.0.1:1/v1','profile':'profile','request_compatibility':RULE}},key)
            try:
                token=policy.issue(body,session_id='s',turn_id='t',profile='profile',route='head')
                raw,headers,mode=proxy.prepare('head',body,token)
                self.assertEqual(mode,'profile');self.assertIs(json.loads(raw)['chat_template_kwargs']['enable_thinking'],False)
                _,stamp,nonce,signature=headers['X-Owner-Mode-Auth'].split(':')
                message=stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
                self.assertTrue(hmac.compare_digest(signature,hmac.new(b'k'*32,message.encode(),hashlib.sha256).hexdigest()))
                stock,headers,mode=proxy.prepare('head',body)
                self.assertEqual(mode,'off');self.assertNotIn('X-Owner-Mode-Auth',headers)
                self.assertEqual(json.loads(stock)['chat_template_kwargs'],json.loads(raw)['chat_template_kwargs'])
            finally:proxy.close()

if __name__=='__main__':unittest.main()
