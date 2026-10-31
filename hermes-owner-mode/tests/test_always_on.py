# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Persistent all-request activation is separate from identity and tool access."""
import json
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import httpx

from test_owner_mode import Runtime, Policy, ProviderProxy
from owner_mode.bridge import Bridge
from owner_mode.client import Client, ProfileNotActive
from owner_mode.contacts import ApprovalStore, empty, validate
from owner_mode.policy import flatten_request
from support import Engine


class ActivationStore(unittest.TestCase):
    def test_legacy_store_defaults_to_verified_and_invalid_modes_are_rejected(self):
        legacy = empty()
        del legacy['mode']
        self.assertEqual(validate(legacy)['mode'], 'verified')
        for mode in ('global', '', None, True, {}, []):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate({**legacy, 'mode': mode})

    def test_mode_persists_without_changing_grants_tools_or_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'approvals.json'
            store = ApprovalStore(path)
            store.set_contact('sms', 'default', '+15550000011', True)
            store.set_tool('delegate_task', False)
            before = store.read()
            store.set_mode('always_on')
            self.assertTrue(ApprovalStore(path).always_on())
            after = store.read()
            self.assertEqual(before['grants'], after['grants'])
            self.assertEqual(before['tools'], after['tools'])
            store.update(lambda state: state.update(enabled=False))
            store.set_mode('verified')
            store.set_mode('always_on')
            self.assertFalse(ApprovalStore(path).always_on())
            self.assertFalse(store.read()['enabled'])


class AlwaysOnProxy(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.key = self.root/'key'
        self.key.write_bytes(b'k'*32)
        self.key.chmod(0o600)
        self.store = ApprovalStore(self.root/'approvals.json')
        self.store.set_mode('always_on')
        self.engine = Engine(b'k'*32)
        self.addCleanup(self.engine.close)
        self.policy = Policy(tool_policy=self.store.tool_enabled)
        self.rows = []
        self.audit_recorded = threading.Event()
        def audit(row):
            self.rows.append(row)
            self.audit_recorded.set()
        self.routes = {'head': {'upstream': self.engine.base_url, 'profile': 'owner-test',
                                'owner_mode_scale': .75}}
        self.proxy = ProviderProxy(self.policy, self.routes, self.key, approvals=self.store,
                                   audit=audit)
        self.proxy.start()
        self.addCleanup(self.proxy.close)
        self.runtime = Runtime(self.policy, self.proxy)
        self.body = {'model': 'bench', 'messages': [{'role': 'user', 'content': 'Hello'}]}

    def send(self, body=None, headers=None, proxy=None):
        proxy = proxy or self.proxy
        self.audit_recorded.clear()
        with httpx.Client(timeout=3) as client:
            response = client.post(proxy.base_url('head')+'/chat/completions',
                                   json=self.body if body is None else body, headers=headers)
        if proxy is self.proxy:
            # The error response can arrive before the handler's audit callback.
            self.assertTrue(self.audit_recorded.wait(5), 'Proxy audit callback did not complete')
        return response

    def test_unverified_no_hook_requests_are_signed_and_cannot_override_profile_or_scale(self):
        response = self.send({**self.body, 'owner_mode': 'off', 'owner_mode_scale': 0},
                             {'X-Owner-Mode-Ticket': 'forged', 'X-Owner-Mode-Auth': 'forged'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'owner-test')
        row = self.engine.records[-1]
        self.assertEqual(row['auth_reason'], 'valid')
        self.assertEqual(row['body']['owner_mode_scale'], .75)
        self.assertEqual(row['body']['messages'], self.body['messages'])
        self.assertFalse(self.rows[-1]['owner_verified'])
        self.assertEqual(self.rows[-1]['decision'], 'always_on')
        self.assertEqual(self.rows[-1]['activation_mode'], 'always_on')
        self.assertFalse(self.policy.owner_turn('s', 't'))

    def test_images_and_reasoning_controls_are_preserved(self):
        body = {**self.body, 'chat_template_kwargs': {'reasoning_effort': 'max'},
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Describe this.'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}]}]}
        self.assertEqual(self.send(body).status_code, 200)
        actual = self.engine.records[-1]['body']
        self.assertEqual(actual['messages'], body['messages'])
        self.assertEqual(actual['chat_template_kwargs'], body['chat_template_kwargs'])
        self.assertFalse(self.rows[-1]['owner_verified'])

    def test_tool_model_calls_are_on_without_granting_owner_identity(self):
        self.store.set_tool('disabled_tool', False)
        def operation(arguments):
            request = self.runtime.request(self.body, session_id='child', turn_id='task',
                                           base_url=self.proxy.base_url('head'))['request']
            self.assertEqual(arguments, {'untouched': True})
            return self.send(flatten_request(request), request['extra_headers'])
        response = self.runtime.execute_tool({'untouched': True}, operation, tool_name='disabled_tool')
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'owner-test')
        self.assertFalse(self.store.tool_enabled('disabled_tool'))
        self.assertFalse(self.policy.is_owner_work('child', 'task'))
        self.assertFalse(self.rows[-1]['owner_verified'])

    def test_mode_changes_and_disable_are_effective_without_restart(self):
        self.assertEqual(self.send().headers['X-Owner-Mode-Effective'], 'owner-test')
        self.store.set_mode('verified')
        self.assertEqual(self.send().headers['X-Owner-Mode-Effective'], 'off')
        self.store.set_mode('always_on')
        self.assertEqual(self.send().headers['X-Owner-Mode-Effective'], 'owner-test')
        self.store.update(lambda state: state.update(enabled=False))
        self.assertEqual(self.send().headers['X-Owner-Mode-Effective'], 'off')

    def test_fresh_proxy_uses_persisted_mode_without_session_state(self):
        restarted = ProviderProxy(Policy(), self.routes, self.key,
                                  approvals=ApprovalStore(self.store.path))
        restarted.start()
        self.addCleanup(restarted.close)
        self.assertEqual(self.send(proxy=restarted).headers['X-Owner-Mode-Effective'], 'owner-test')

    def test_missing_or_insecure_key_does_not_send_stock(self):
        self.key.chmod(0o644)
        self.assertEqual(self.send().status_code, 502)
        self.key.unlink()
        self.assertEqual(self.send().status_code, 502)
        self.assertEqual(self.engine.records, [])

    def test_unavailable_profile_is_not_retried_as_stock(self):
        self.engine.refuse_owner_mode = True
        response = self.send()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.engine.records), 1)
        self.assertEqual(self.engine.records[0]['body']['owner_mode'], 'owner-test')

    def test_engine_that_does_not_activate_profile_is_not_served(self):
        self.engine.key = b'x'*32
        # Rejection closes the upstream stream before its body can be exposed.
        self.engine.server.handle_error = lambda *_: None
        response = self.send({**self.body, 'stream': True})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('data:', response.text)
        self.assertEqual(self.rows[-1]['decision'], 'always_on_unconfirmed')
        self.assertEqual(self.rows[-1]['effective'], 'off')

    def test_bad_store_and_disabled_route_fail_without_model_calls(self):
        self.store.path.write_text('{}')
        self.assertEqual(self.send().status_code, 502)
        self.assertEqual(self.engine.records, [])
        self.store.path.unlink()
        self.store.set_mode('always_on')
        self.proxy.disabled.add('head')
        self.assertEqual(self.send().status_code, 502)
        self.assertEqual(self.engine.records, [])

    def test_missing_store_does_not_silently_reset_to_stock(self):
        self.store.path.unlink()
        self.assertEqual(self.send().status_code, 502)
        self.assertEqual(self.engine.records, [])

    def test_kill_switch_prevents_even_a_verified_ticket_from_signing(self):
        self.store.update(lambda state: state.update(enabled=False))
        with mock.patch.object(self.policy, 'consume', return_value='owner-test') as consume:
            response = self.send(headers={'X-Owner-Mode-Ticket': 'synthetic'})
        consume.assert_not_called()
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'off')
        self.assertEqual(self.rows[-1]['activation_mode'], 'off')
        self.assertEqual(self.rows[-1]['decision'], 'deployment_disabled')

    def test_empty_profile_fails_without_model_calls(self):
        self.proxy.routes['head']['profile'] = 'off'
        self.assertEqual(self.send().status_code, 502)
        self.assertEqual(self.engine.records, [])

    def test_bridge_without_a_native_claim_still_applies_global_policy(self):
        bridge = Bridge(self.store, self.key, self.routes)
        proxy = ProviderProxy(bridge.policy, self.routes, self.key, bridge=bridge)
        proxy.start()
        self.addCleanup(proxy.close)
        self.assertEqual(self.send(proxy=proxy).headers['X-Owner-Mode-Effective'], 'owner-test')
        self.assertFalse(bridge.policy.owner_turn('s', 't'))


class AlwaysOnClient(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.key = self.root/'key'
        self.key.write_bytes(b'k'*32)
        self.key.chmod(0o600)
        self.config = self.root/'client.json'
        self.config.write_text(json.dumps({'base_url': 'http://localhost/v1', 'model': 'bench',
            'key_file': str(self.key), 'mode': 'always_on', 'profile': 'owner-test'}))

    def test_config_defaults_every_call_to_the_profile_and_signs_it(self):
        def reply(request):
            self.assertEqual(json.loads(request.content)['owner_mode'], 'owner-test')
            self.assertIn('X-Owner-Mode-Auth', request.headers)
            return httpx.Response(200, headers={'X-Owner-Mode-Effective': 'owner-test'}, json={'choices': []})
        with Client.from_config(self.config, transport=httpx.MockTransport(reply)) as client:
            client.chat([])
            client.chat([], owner_mode='off', owner_mode_scale=0)
            with self.assertRaises(ValueError):
                client.chat([], profile='off')
            with self.assertRaises(ValueError):
                client.chat([], profile='other')

    def test_gateway_client_does_not_need_or_read_the_engine_signing_key(self):
        config = json.loads(self.config.read_text())
        config['gateway'] = True
        del config['key_file']
        self.config.write_text(json.dumps(config))
        def reply(request):
            self.assertEqual(json.loads(request.content)['owner_mode'], 'owner-test')
            self.assertNotIn('X-Owner-Mode-Auth', request.headers)
            return httpx.Response(200, headers={'X-Owner-Mode-Effective': 'owner-test'}, json={'choices': []})
        with Client.from_config(self.config, transport=httpx.MockTransport(reply)) as client:
            client.chat([])
        for value in ('true', 1, None):
            with self.subTest(gateway=value), self.assertRaises(ValueError):
                Client('http://localhost/v1', 'bench', gateway=value)

    def test_streaming_uses_profile_and_rejects_unconfirmed_mode(self):
        for effective in ('owner-test', 'off'):
            transport = httpx.MockTransport(lambda request: httpx.Response(200,
                headers={'X-Owner-Mode-Effective': effective}, text='data: {}\n\ndata: [DONE]\n\n'))
            with Client.from_config(self.config, transport=transport) as client:
                if effective == 'owner-test':
                    with client.stream_chat([]) as events:
                        self.assertEqual(list(events), [{}])
                else:
                    with self.assertRaises(ProfileNotActive):
                        with client.stream_chat([]) as events:
                            list(events)

    def test_always_on_needs_a_profile_and_verified_defaults_stay_off(self):
        for profile in (None, '', 'off'):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                Client('http://localhost/v1', 'bench', mode='always_on', profile=profile)
        with Client('http://localhost/v1', 'bench', profile='owner-test') as client:
            raw, headers = client.prepare({})
            self.assertEqual(json.loads(raw)['owner_mode'], 'off')
            self.assertNotIn('X-Owner-Mode-Auth', headers)

    def test_terminal_client_uses_the_deployment_mode_without_an_owner_flag(self):
        from flashalign import __main__ as terminal
        calls = []
        def reply(request):
            calls.append(json.loads(request.content)['owner_mode'])
            return httpx.Response(200, headers={'X-Owner-Mode-Effective': 'owner-test'}, json={'choices': []})
        with Client.from_config(self.config, transport=httpx.MockTransport(reply)) as client, \
                mock.patch.object(terminal.Client, 'from_config', return_value=client), \
                mock.patch.object(terminal.sys, 'argv', ['flashalign-client', '--config', str(self.config)]), \
                mock.patch.object(terminal.sys, 'stdin', io.StringIO('Hello')), \
                mock.patch.object(terminal.sys, 'stdout', io.StringIO()):
            terminal.main()
        self.assertEqual(calls, ['owner-test'])


if __name__ == '__main__':
    unittest.main()
