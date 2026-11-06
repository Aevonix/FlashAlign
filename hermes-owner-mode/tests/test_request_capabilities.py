# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Setup and verified calls must use the same bounded option vocabulary."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest

from test_setup import wizard
from owner_mode.model_metadata import _portable_generation_settings
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy
from owner_mode.request_capabilities import DEFAULT_REQUEST_CAPABILITIES


class RequestCapabilitiesTests(unittest.TestCase):
    def setUp(self):
        class Authority:
            def verify_turn(self, session_id, turn_id, user_message, **_):
                return VerifiedTurn(session_id, turn_id, 'sms', user_message)
        self.authority = Authority()
        self.policy = Policy(authority=self.authority, allowed_channels=['sms'], trust={'tools': 'host'})
        self.policy.begin(session_id='session', turn_id='turn', user_message='Hello', platform='sms')
        self.proof = VerifiedTurn('session', 'turn', 'sms', 'Hello')
        self.body = {'model': 'fixture', 'messages': [{'role': 'user', 'content': 'Hello'}]}

    def test_preserved_generation_options_are_accepted_by_verified_policy(self):
        options = {'temperature': 0, 'top_p': .9, 'repetition_penalty': 1.1,
                   'length_penalty': 1, 'min_tokens': 0, 'top_logprobs': 3,
                   'ignore_eos': False, 'logprobs': True, 'stop_token_ids': [1, 2],
                   'chat_template_kwargs': {'enable_thinking': True,
                                            'preserve_thinking': True, 'thinking_budget': 1024},
                   'response_format': {'type': 'json_object'}}
        saved = _portable_generation_settings(options)
        self.assertEqual(saved, options)
        self.assertEqual(self.policy.check({**self.body, **saved}, self.proof), '')

    def test_schema_and_named_tool_are_bounded_and_checked(self):
        body = {**self.body, 'tools': [{'type': 'function', 'function': {'name': 'lookup'}}],
                'tool_choice': {'type': 'function', 'function': {'name': 'lookup'}},
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': 'result', 'strict': True,
                    'schema': {'type': 'object', 'properties': {'answer': {'type': 'string'}}}}}}
        self.assertEqual(self.policy.check(body, self.proof), '')
        bad = copy.deepcopy(body)
        bad['tool_choice']['function']['name'] = 'undeclared'
        self.assertEqual(self.policy.check(bad, self.proof), 'tool_choice')
        bad = copy.deepcopy(body)
        bad['response_format']['headers'] = {'Authorization': 'fixture'}
        self.assertEqual(self.policy.check(bad, self.proof), 'response_format')
        deep = {}
        for _ in range(35):
            deep = {'items': deep}
        bad = copy.deepcopy(body)
        bad['response_format']['json_schema']['schema'] = deep
        self.assertEqual(self.policy.check(bad, self.proof), 'response_format')

    def test_bad_types_and_unknown_fields_do_not_gain_authorization(self):
        for option in ({'min_tokens': True}, {'top_p': float('nan')},
                       {'logprobs': 'true'}, {'stop_token_ids': [-1]},
                       {'chat_template_kwargs': {'headers': {}}},
                       {'request_plugin': 'untrusted.module'}):
            with self.subTest(option=option):
                self.assertTrue(self.policy.check({**self.body, **option}, self.proof))
        self.policy.end(session_id='session', turn_id='turn')
        self.assertIsNone(self.policy.issue({**self.body, 'min_tokens': 0},
                         session_id='session', turn_id='turn', profile='fixture', route='head'))

    def test_trusted_extension_does_not_require_policy_branch(self):
        capabilities = DEFAULT_REQUEST_CAPABILITIES.with_field(
            'example_search_width', lambda value, body: type(value) is int and value > 0)
        policy = Policy(request_capabilities=capabilities)
        self.assertEqual(policy.check({**self.body, 'example_search_width': 4}, self.proof), '')
        self.assertEqual(self.policy.check({**self.body, 'example_search_width': 4}, self.proof),
                         'unknown_request_field')
        with self.assertRaises(ValueError):
            capabilities.with_field('model', lambda *_: True)

    def test_preserved_options_are_in_final_signed_body(self):
        body = {**self.body, 'repetition_penalty': 1.1,
                'chat_template_kwargs': {'preserve_thinking': True},
                'response_format': {'type': 'json_object'}}
        with tempfile.TemporaryDirectory() as temp:
            key = Path(temp)/'key'
            key.write_bytes(b'k'*32)
            key.chmod(0o600)
            proxy = ProviderProxy(self.policy, {'head': {'upstream': 'http://127.0.0.1:1/v1',
                                  'profile': 'fixture'}}, key)
            try:
                ticket = self.policy.issue(body, session_id='session', turn_id='turn',
                                           profile='fixture', route='head')
                raw, headers, mode = proxy.prepare('head', body, ticket)
                self.assertEqual(mode, 'fixture')
                signed = json.loads(raw)
                self.assertEqual(signed['response_format'], body['response_format'])
                _, stamp, nonce, mac = headers['X-Owner-Mode-Auth'].split(':')
                material = stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
                self.assertTrue(hmac.compare_digest(mac, hmac.new(b'k'*32, material.encode(), hashlib.sha256).hexdigest()))
                self.assertEqual(proxy.prepare('head', body, ticket)[2], 'off')
            finally:
                proxy.close()


if __name__ == '__main__':
    unittest.main()
