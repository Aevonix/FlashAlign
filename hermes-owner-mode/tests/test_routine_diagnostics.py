# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Routine traffic stays silent without gaining native identity or tickets."""
from types import SimpleNamespace
import unittest
from unittest import mock

import test_plugin_registration as fixtures
from owner_mode import plugin
from owner_mode.contacts import ApprovalStore
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import VerifiedTurn


class RoutineDiagnostics(unittest.TestCase):
    setUp = fixtures.ProcessRegistrationTests.setUp
    context = fixtures.ProcessRegistrationTests.context

    def native(self):
        self.store = ApprovalStore(self.root / 'approvals.json')
        self.store.set_contact('whatsapp', 'default', '+10000000000', True)
        self.rows = []
        self.authority = NativeAuthority(self.store.path, VerifiedTurn, audit=self.rows.append)
        self.addCleanup(self.authority.close)
        adapter = SimpleNamespace(platform='whatsapp')
        gateway = SimpleNamespace(adapters={'whatsapp': adapter},
                                  _is_user_authorized_for_source=lambda _: True)
        source = SimpleNamespace(platform='whatsapp', chat_type='dm', chat_id='chat',
                                 user_id='+10000000000', account_id='default', is_bot=False,
                                 _transport_adapter_ref=lambda: adapter, delivered_via_upstream_relay=False)
        event = SimpleNamespace(source=source, text='Hello', message_type='text', media_urls=[],
                                internal=False, message_id='message', reply_to_message_id=None,
                                reply_to_text=None, metadata={'whatsapp_native_type': 'conversation'})
        return event, gateway

    def rejected(self, event, gateway):
        with self.assertNoLogs('owner_mode', level='WARNING'):
            self.authority.hear(event=event, gateway=gateway)
        self.assertEqual(self.rows, [])
        self.assertEqual(self.authority.pending, {})
        self.assertEqual(self.authority.groups, {})
        self.assertEqual(list(self.authority.rendered.values()), [None])

    def test_unapproved_direct_message_is_silent(self):
        event, gateway = self.native()
        event.source.user_id = '+10000000001'
        self.rejected(event, gateway)

    def test_approved_contact_group_message_is_silent(self):
        event, gateway = self.native()
        event.source.chat_type = 'group'
        self.rejected(event, gateway)

    def test_reaction_is_silent(self):
        event, gateway = self.native()
        event.message_type = 'reaction'
        self.rejected(event, gateway)

    def test_own_account_echo_is_silent(self):
        event, gateway = self.native()
        event.metadata['whatsapp_from_owner'] = True
        self.rejected(event, gateway)

    def test_approved_owner_message_while_paused_is_silent(self):
        event, gateway = self.native()
        self.store.set_enabled(False)
        self.rejected(event, gateway)

    def test_invalid_native_composition_is_silent(self):
        event, gateway = self.native()
        with mock.patch.object(self.authority.observer, 'parts', return_value=(event, event)):
            self.rejected(event, gateway)

    def test_reused_native_identity_is_silent_and_invalidates_rendering(self):
        event, gateway = self.native()
        self.authority.hear(event=event, gateway=gateway)
        pending = self.authority.pending.copy()
        self.assertEqual(len(pending), 1)
        event.text = 'Changed'
        with self.assertNoLogs('owner_mode', level='WARNING'):
            self.authority.hear(event=event, gateway=gateway)
        self.assertEqual(self.rows, [])
        self.assertEqual(self.authority.pending, pending)
        self.assertEqual(list(self.authority.rendered.values()), [None])

    def test_other_provider_calls_are_silent_and_strip_owner_fields(self):
        rows = []
        proxy = SimpleNamespace(routes={'head': {}}, base_url=lambda _: 'http://127.0.0.1:8030/head/v1',
                                audit=rows.append)
        runtime = plugin.Runtime(None, proxy)
        for url in ('https://provider.invalid/v1', 'http://provider.invalid:8030/v1',
                    'http://localhost:8031/v1'):
            with self.subTest(url=url), self.assertNoLogs('owner_mode', level='WARNING'):
                result = runtime.request({'messages': [], 'owner_mode': 'owner-test',
                                          'extra_headers': {'X-Owner-Mode-Context': 'untrusted'}}, base_url=url)
                self.assertEqual(result['request'], {'messages': [], 'extra_headers': {}})
                self.assertEqual(rows, [])

    def test_one_shot_process_is_silent_and_registers_nothing(self):
        ctx = self.context(serve_argv=['gateway', 'run'])
        with mock.patch.object(plugin.sys, 'argv', ['fixture', 'chat', '-q', 'Hello']), \
             self.assertNoLogs('owner_mode', level='WARNING'):
            self.assertIsNone(plugin.register(ctx))
        log = self.root / 'logs' / 'owner-mode.log'
        self.assertFalse(log.exists())
        self.assertFalse(ctx.hooks)
        self.assertFalse(ctx.middleware)
        self.assertFalse(ctx.unload)
        self.assertFalse(fixtures.Proxy.instances)
