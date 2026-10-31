# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Declared template controls preserve grading and original-body authorization."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.model_metadata import compatible_request, declared_request_compatibility
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy
from owner_mode.setup_runtime import SetupError


DECLARATION={'kind':'chat_template_reasoning_v1','efforts':['low','medium','xhigh']}
RULE={**DECLARATION,'model':'synthetic-model'}


class TemplateEffort(unittest.TestCase):
    def test_declared_grades_and_disabled_mode_are_mapped_without_mutation(self):
        for effort in ('none','low','medium','xhigh'):
            with self.subTest(effort=effort):
                body={'model':'synthetic-model','reasoning_effort':effort,'messages':[]}
                before=copy.deepcopy(body);result=compatible_request(body,RULE)
                self.assertEqual(body,before)
                self.assertNotIn('reasoning_effort',result)
                self.assertIs(result['chat_template_kwargs']['enable_thinking'],effort!='none')
                self.assertEqual(result['chat_template_kwargs'].get('reasoning_effort'),None if effort=='none' else effort)

    def test_other_models_and_missing_effort_remain_identical(self):
        for body in ({'model':'other','reasoning_effort':'low'}, {'model':'synthetic-model'},
                     {'model':'synthetic-model','reasoning_effort':None}):
            self.assertIs(compatible_request(body,RULE),body)

    def test_explicit_template_kwargs_keep_native_precedence(self):
        body={'model':'synthetic-model','reasoning_effort':'low',
              'chat_template_kwargs':{'enable_thinking':False,'reasoning_effort':'medium','preserve_thinking':True}}
        self.assertEqual(compatible_request(body,RULE)['chat_template_kwargs'],body['chat_template_kwargs'])

    def test_unsupported_grade_or_bad_kwargs_is_reported(self):
        for fields in ({'reasoning_effort':'high'},{'reasoning_effort':1},
                       {'reasoning_effort':'low','chat_template_kwargs':[]}):
            with self.subTest(fields=fields),self.assertRaises(SetupError):
                compatible_request({'model':'synthetic-model',**fields},RULE)

    def test_declaration_requires_exact_model_type_and_bounded_shape(self):
        metadata={'model_types':['family','other'],
                  'request_compatibility_by_model_type':{'family':DECLARATION}}
        self.assertEqual(declared_request_compatibility(metadata,'family'),DECLARATION)
        self.assertIsNone(declared_request_compatibility(metadata,'other'))
        for value in (None,'deepseek_v41',{}, {'kind':'unrecognized','efforts':['low']},
                      {'kind':'chat_template_reasoning_v1','efforts':['none']},
                      {**DECLARATION,'headers':{}}, {**DECLARATION,'model':'other'}):
            with self.subTest(value=value),self.assertRaises(SetupError):
                declared_request_compatibility({'model_types':['family'],
                    'request_compatibility_by_model_type':{'family':value}},'family')

    def test_writer_binds_the_rule_to_selected_model_and_preserves_explicit_compat(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary);path=home/'.openclaw/openclaw.json';path.parent.mkdir()
            path.write_text(json.dumps({'models':{'providers':{'custom':{'models':[
                {'id':'synthetic-model','compat':{'thinkingFormat':'qwen-chat-template'}}]}}}}))
            with mock.patch.object(wizard,'free_port',return_value=48123):
                wizard.provision(home,['openclaw'],'http://127.0.0.1:48124/v1','synthetic-model',
                    'synthetic-profile',[],True,model_info={'request_compatibility':DECLARATION,
                        'openclaw_compat':{'thinkingFormat':'openai','supportedReasoningEfforts':['none','low','medium','xhigh']}})
            cfg=json.loads(path.read_text())
            self.assertEqual(cfg['models']['providers']['owner-mode']['models'][0]['compat'],{'thinkingFormat':'qwen-chat-template'})
            bridge=json.loads((home/'.config/owner-mode/openclaw-bridge.json').read_text())
            self.assertEqual(bridge['routes']['head']['request_compatibility'],RULE)


class TemplateAuthorization(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.key=Path(self.tmp.name)/'key';self.key.write_bytes(b'k'*32);self.key.chmod(0o600)
        class Authority:
            active=True
            def verify_turn(self,session_id,turn_id,user_message,**_):
                return VerifiedTurn(session_id,turn_id,'sms',user_message)
            def is_active(self,proof):return self.active
        self.authority=Authority()
        self.policy=Policy(authority=self.authority,allowed_channels=['sms'])
        self.policy.begin(session_id='s',turn_id='t',user_message='Hello',platform='sms')
        self.body={'model':'synthetic-model','messages':[{'role':'user','content':'Hello'}],'reasoning_effort':'low'}
        self.proxy=ProviderProxy(self.policy,{'head':{'upstream':'http://127.0.0.1:1/v1','profile':'fixture',
            'request_compatibility':RULE}},self.key)
        self.addCleanup(self.proxy.close)

    def ticket(self):return self.policy.issue(self.body,session_id='s',turn_id='t',profile='fixture',route='head')

    def test_original_body_binding_then_final_hmac_and_replay_denial(self):
        token=self.ticket();raw,headers,mode=self.proxy.prepare('head',self.body,token)
        self.assertEqual(mode,'fixture')
        self.assertEqual(json.loads(raw)['chat_template_kwargs'],{'enable_thinking':True,'reasoning_effort':'low'})
        _,stamp,nonce,signature=headers['X-Owner-Mode-Auth'].split(':')
        message=stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
        self.assertTrue(hmac.compare_digest(signature,hmac.new(b'k'*32,message.encode(),hashlib.sha256).hexdigest()))
        self.assertEqual(self.proxy.prepare('head',self.body,token)[2],'off')

    def test_changed_grade_cannot_borrow_original_capability(self):
        raw,headers,mode=self.proxy.prepare('head',{**self.body,'reasoning_effort':'medium'},self.ticket())
        self.assertEqual(mode,'off');self.assertNotIn('X-Owner-Mode-Auth',headers)
        self.assertEqual(json.loads(raw)['chat_template_kwargs']['reasoning_effort'],'medium')

    def test_stock_request_gets_same_model_mapping_without_authorization(self):
        raw,headers,mode=self.proxy.prepare('head',{**self.body,'reasoning_effort':'none'})
        self.assertEqual(mode,'off');self.assertNotIn('X-Owner-Mode-Auth',headers)
        self.assertEqual(json.loads(raw)['chat_template_kwargs'],{'enable_thinking':False})

    def test_revoked_authority_cannot_sign_the_normalized_request(self):
        token=self.ticket();self.authority.active=False
        raw,headers,mode=self.proxy.prepare('head',self.body,token)
        self.assertEqual(mode,'off');self.assertNotIn('X-Owner-Mode-Auth',headers)
        self.assertEqual(json.loads(raw)['chat_template_kwargs']['reasoning_effort'],'low')

    def test_ended_turn_cannot_reuse_a_ticket(self):
        token=self.ticket();self.policy.end(session_id='s',turn_id='t')
        _,headers,mode=self.proxy.prepare('head',self.body,token)
        self.assertEqual(mode,'off');self.assertNotIn('X-Owner-Mode-Auth',headers)


if __name__=='__main__':unittest.main()
