# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Installed transport identity, opt-in grants and native Telegram contracts."""
from pathlib import Path
import os
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_owner_mode  # Loads the source package as owner_mode.
from owner_mode.channels import (AdapterRegistry, ChannelAdapter, DEFAULT_ENABLED,
                                 DEFAULT_REGISTRY, opaque_id)
from owner_mode.contacts import ApprovalStore
from owner_mode.bridge import Bridge
from owner_mode import hermes_ingress
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy
from support import Engine


class RegistryTests(unittest.TestCase):
    def test_installed_registry_rejects_collisions_and_invalid_capabilities(self):
        with self.assertRaises(ValueError):
            DEFAULT_REGISTRY.with_adapter(DEFAULT_REGISTRY.get('telegram'))
        with self.assertRaises(ValueError):
            ChannelAdapter('api', opaque_id, frozenset({'trust_everything'}))
        with self.assertRaises(ValueError):
            DEFAULT_REGISTRY.select_enabled(['request-invented'])
        with self.assertRaises(ValueError):
            DEFAULT_REGISTRY.select_enabled(['telegram', 'telegram'])
        self.assertFalse(DEFAULT_REGISTRY.supports('local-terminal', 'native_message'))
        self.assertFalse(DEFAULT_REGISTRY.supports('local-terminal', 'native_session'))

    def test_telegram_uses_native_user_ids_without_phone_coercion(self):
        self.assertEqual(DEFAULT_REGISTRY.canonical_contact('telegram', '123456789'), '123456789')
        self.assertEqual(DEFAULT_REGISTRY.canonical_contact('sms', '123456789'), '+123456789')
        for value in ('@alice', '+123456789', '-100123456789', '00123', '0', '', 'a\nb'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DEFAULT_REGISTRY.canonical_contact('telegram', value)

    def test_registry_extension_does_not_mutate_default_or_trust_a_label(self):
        adapter = ChannelAdapter('enterprise-chat', opaque_id, frozenset({'native_message'}))
        installed = DEFAULT_REGISTRY.with_adapter(adapter)
        self.assertIsNone(DEFAULT_REGISTRY.get(adapter.id))
        self.assertIs(installed.get(adapter.id), adapter)
        class Authority:
            def verify_turn(self, **kwargs):
                return VerifiedTurn(kwargs['session_id'], kwargs['turn_id'], adapter.id, kwargs['user_message'])
        denied = Policy(authority=Authority(), allowed_channels=[adapter.id])
        allowed = Policy(authority=Authority(), registry=installed, allowed_channels=[adapter.id])
        for policy in (denied, allowed):
            policy.begin(session_id='s', turn_id='t', user_message='hi', platform=adapter.id)
        self.assertFalse(denied.is_owner_work('s', 't'))
        self.assertTrue(allowed.is_owner_work('s', 't'))


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ApprovalStore(Path(self.tmp.name)/'approvals.json')

    def test_legacy_defaults_and_new_transport_opt_in(self):
        self.store.set_contact('whatsapp', 'default', '15550000011', True)
        self.assertEqual(self.store.channels(), frozenset(DEFAULT_ENABLED))
        self.assertTrue(self.store.approved('whatsapp', 'default', '+15550000011'))
        self.store.set_contact('telegram', 'bot-a', '123456789', True)
        self.assertIsNone(self.store.approved('telegram', 'bot-a', '123456789'))
        self.store.set_channels([*DEFAULT_ENABLED, 'telegram'])
        self.assertTrue(self.store.approved('telegram', 'bot-a', '123456789'))
        self.assertIsNone(self.store.approved('telegram', 'bot-b', '123456789'))

    def test_sms_requires_explicit_channel_selection_even_with_a_contact_grant(self):
        self.store.set_contact('sms', 'default', '+15550000011', True)
        self.assertEqual(self.store.channels(), frozenset({'whatsapp', 'rcs'}))
        self.assertFalse(self.store.channel_enabled('sms'))
        self.assertIsNone(self.store.approved('sms', 'default', '+15550000011'))

        self.store.set_channels(['whatsapp', 'rcs', 'sms'])
        grant = self.store.approved('sms', 'default', '+15550000011')
        self.assertTrue(grant)
        self.assertTrue(self.store.active(grant))
        self.assertIsNone(self.store.approved('sms', 'other-account', '+15550000011'))

        self.store.set_channels(['whatsapp', 'rcs'])
        self.assertFalse(self.store.active(grant))
        self.store.set_channels(['whatsapp', 'rcs', 'sms'])
        self.assertFalse(self.store.active(grant))
        self.assertTrue(self.store.approved('sms', 'default', '+15550000011'))

    def test_explicit_saved_sms_selection_survives_reopening(self):
        self.store.set_channels(['sms'])
        self.store.set_contact('sms', 'default', '+15550000011', True)
        reopened = ApprovalStore(self.store.path)
        self.assertEqual(reopened.channels(), frozenset({'sms'}))
        self.assertTrue(reopened.approved('sms', 'default', '+15550000011'))

    def test_disabled_sms_does_not_call_the_identity_authority(self):
        store = self.store
        store.set_contact('sms', 'default', '+15550000011', True)

        class Authority:
            calls = 0

            def verify_turn(self, **kwargs):
                self.calls += 1
                return VerifiedTurn(kwargs['session_id'], kwargs['turn_id'],
                                    'sms', kwargs['user_message'])

        authority = Authority()
        policy = Policy(authority=authority, allowed_channels=DEFAULT_REGISTRY.ids(),
                        channel_policy=store.channel_enabled)
        policy.begin(session_id='s', turn_id='disabled', user_message='hi', platform='sms')
        self.assertFalse(policy.is_owner_work('s', 'disabled'))
        self.assertEqual(authority.calls, 0)

        store.set_channels(['whatsapp', 'rcs', 'sms'])
        policy.begin(session_id='s', turn_id='enabled', user_message='hi', platform='sms')
        self.assertTrue(policy.is_owner_work('s', 'enabled'))
        self.assertEqual(authority.calls, 1)

    def test_configuration_cannot_register_an_adapter_or_merge_identity_namespaces(self):
        with self.assertRaises(ValueError):
            self.store.set_channels(['request-invented'])
        self.store.update(lambda state: state.update(enabled_channels=['request-invented']))
        self.assertFalse(self.store.channel_enabled('request-invented'))
        self.assertEqual(self.store.channels(), frozenset())
        self.store.set_channels(['sms', 'telegram'])
        self.store.set_contact('sms', 'legacy account', '123456789', True)
        self.store.set_contact('telegram', 'legacy account', '123456789', True)
        sms = self.store.approved('sms', 'legacy account', '+123456789')
        telegram = self.store.approved('telegram', 'legacy account', '123456789')
        self.assertTrue(sms and telegram)
        self.assertNotEqual(sms, telegram)

    def test_disabling_adapter_revokes_capabilities_even_after_reenable(self):
        self.store.set_channels(['telegram'])
        self.store.set_contact('telegram', 'bot-a', '123456789', True)
        old = self.store.approved('telegram', 'bot-a', '123456789')
        self.store.set_channels([])
        self.assertFalse(self.store.active(old))
        self.store.set_channels(['telegram'])
        self.assertFalse(self.store.active(old))
        self.assertNotEqual(old, self.store.approved('telegram', 'bot-a', '123456789'))

    def test_custom_installed_adapter_still_needs_enabled_channel_and_exact_grant(self):
        registry = DEFAULT_REGISTRY.with_adapter(ChannelAdapter('enterprise-chat', opaque_id,
                                    frozenset({'native_message'})))
        store = ApprovalStore(self.store.path, registry=registry)
        with self.assertRaises(ValueError):
            self.store.set_contact('enterprise-chat', 'workspace', 'member:7', True)
        store.set_contact('enterprise-chat', 'workspace', 'member:7', True)
        self.assertIsNone(store.approved('enterprise-chat', 'workspace', 'member:7'))
        store.set_channels(['enterprise-chat'])
        self.assertTrue(store.approved('enterprise-chat', 'workspace', 'member:7'))
        self.assertIsNone(store.approved('enterprise-chat', 'other-workspace', 'member:7'))

    def test_uninstalled_adapter_keeps_records_without_disabling_other_grants(self):
        adapter = ChannelAdapter('enterprise-chat', opaque_id, frozenset({'native_message'}))
        installed = ApprovalStore(self.store.path, registry=DEFAULT_REGISTRY.with_adapter(adapter))
        installed.set_channels(['whatsapp', 'enterprise-chat'])
        installed.set_contact('enterprise-chat', 'workspace', 'member:7', True)
        installed.set_contact('whatsapp', 'default', '+15550000011', True)
        old = installed.approved('enterprise-chat', 'workspace', 'member:7')
        removed = ApprovalStore(self.store.path)
        self.assertEqual(removed.read(), installed.read())
        self.assertEqual(removed.channels(), frozenset({'whatsapp'}))
        self.assertTrue(removed.approved('whatsapp', 'default', '+15550000011'))
        self.assertIsNone(removed.approved('enterprise-chat', 'workspace', 'member:7'))
        self.assertFalse(removed.active(old))
        removed.set_tool('web_search', False)
        self.assertEqual(len(removed.read()['grants']), 2)
        with self.assertRaises(ValueError):
            removed.set_channels(['enterprise-chat'])
        with self.assertRaises(ValueError):
            removed.set_contact('enterprise-chat', 'workspace', 'member:8', True)

    def test_new_identity_version_needs_explicit_grant_renewal(self):
        original = ChannelAdapter('enterprise-chat', opaque_id, frozenset({'native_message'}))
        store = ApprovalStore(self.store.path, registry=DEFAULT_REGISTRY.with_adapter(original))
        store.set_channels(['enterprise-chat'])
        store.set_contact('enterprise-chat', 'workspace', 'member:7', True)
        old = store.approved('enterprise-chat', 'workspace', 'member:7')
        changed = ChannelAdapter('enterprise-chat', opaque_id, frozenset({'native_message'}), identity_version='2')
        upgraded = ApprovalStore(self.store.path, registry=DEFAULT_REGISTRY.with_adapter(changed))
        self.assertIsNone(upgraded.approved('enterprise-chat', 'workspace', 'member:7'))
        self.assertFalse(upgraded.active(old))
        upgraded.set_contact('enterprise-chat', 'workspace', 'member:7', True)
        self.assertTrue(upgraded.approved('enterprise-chat', 'workspace', 'member:7'))
        self.assertFalse(upgraded.active(old))

    def test_normalizer_change_never_rewrites_a_saved_contact(self):
        original = ChannelAdapter('enterprise-chat', opaque_id, frozenset({'native_message'}))
        store = ApprovalStore(self.store.path, registry=DEFAULT_REGISTRY.with_adapter(original))
        store.set_channels(['enterprise-chat'])
        store.set_contact('enterprise-chat', 'workspace', 'MEMBER:7', True)
        old = store.approved('enterprise-chat', 'workspace', 'MEMBER:7')
        changed = ChannelAdapter('enterprise-chat', lambda value: opaque_id(value).lower(),
                                 frozenset({'native_message'}), identity_version='2')
        upgraded = ApprovalStore(self.store.path, registry=DEFAULT_REGISTRY.with_adapter(changed))
        self.assertEqual(upgraded.read()['grants'][0]['contact'], 'MEMBER:7')
        self.assertIsNone(upgraded.approved('enterprise-chat', 'workspace', 'MEMBER:7'))
        self.assertFalse(upgraded.active(old))


class TelegramNativeTests(unittest.TestCase):
    def setUp(self):
        ApprovalTests.setUp(self)
        self.store.set_channels(['telegram'])
        self.store.set_contact('telegram', 'bot-a', '123456789', True)
        self.authority = NativeAuthority(self.store.path, VerifiedTurn)
        self.addCleanup(self.authority.close)
        self.adapter = SimpleNamespace(platform='telegram')
        self.gateway = SimpleNamespace(adapters={'telegram': self.adapter},
            _is_user_authorized_for_source=lambda source: True)

    def event(self, message='m1', **changes):
        # The pinned Hermes Telegram adapter maps from_user.id into user_id,
        # private into dm, and build_source supplies the live adapter reference.
        source = SimpleNamespace(platform='telegram', account_id='bot-a', user_id='123456789',
            chat_id='123456789', chat_type='dm', is_bot=False,
            _transport_adapter_ref=lambda: self.adapter, delivered_via_upstream_relay=False)
        event = SimpleNamespace(source=source, internal=False, metadata={}, message_type='text',
            message_id=message, text='Please answer.', media_urls=[], reply_to_message_id=None)
        for key, value in changes.items():
            setattr(event, key, value)
        return event

    def verify(self, event, content=None):
        content = event.text if content is None else content
        route = {'SESSION_PLATFORM':'telegram', 'SESSION_USER_ID':event.source.user_id,
                 'SESSION_CHAT_ID':event.source.chat_id, 'SESSION_CHAT_TYPE':event.source.chat_type,
                 'SESSION_ACCOUNT_ID':event.source.account_id, 'SESSION_MESSAGE_ID':event.message_id,
                 'CRON_SESSION':''}
        with mock.patch.object(hermes_ingress, '_route', return_value=route):
            return self.authority.verify_turn(session_id='s', turn_id=event.message_id,
                platform='telegram', sender_id=event.source.user_id, user_message=content,
                conversation_history=[{'role':'user', 'content':content,
                                       'platform_message_id':event.message_id}])

    def test_native_transport_and_current_grant_are_required(self):
        event = self.event()
        self.assertIsNone(self.verify(event))
        self.authority.hear(event=event, gateway=self.gateway)
        proof = self.verify(event)
        self.assertTrue(proof)
        self.assertIsNone(self.verify(event))
        self.store.set_contact('telegram', 'bot-a', '123456789', False)
        self.assertFalse(self.authority.is_active(proof))

    def test_forged_adapter_group_wrong_account_and_disabled_channel_fail(self):
        for kind in ('forged', 'group', 'account', 'disabled', 'host-denied'):
            with self.subTest(kind=kind):
                self.store.set_channels(['telegram'])
                self.gateway._is_user_authorized_for_source = lambda source: True
                event = self.event(kind)
                if kind == 'forged':
                    event.source._transport_adapter_ref = lambda: SimpleNamespace(platform='telegram')
                elif kind == 'group':
                    event.source.chat_type = 'group'
                elif kind == 'account':
                    event.source.account_id = 'bot-b'
                elif kind == 'disabled':
                    self.store.set_channels([])
                else:
                    self.gateway._is_user_authorized_for_source = lambda source: False
                self.authority.hear(event=event, gateway=self.gateway)
                self.assertIsNone(self.verify(event))

    def test_media_requires_exact_trusted_render_binding(self):
        event = self.event(media_urls=['synthetic-image'], message_type='photo', text='')
        content = [{'type':'image_url', 'image_url':{'url':'data:image/png;base64,AAAA'}}]
        self.authority.record_render(event=event, gateway=self.gateway, content=content)
        self.assertIsNone(self.verify(event, [{'type':'text', 'text':'substituted'}]))
        event = self.event('m2', media_urls=['synthetic-image'], message_type='photo', text='')
        self.authority.record_render(event=event, gateway=self.gateway, content=content)
        self.assertTrue(self.verify(event, content))


class TelegramBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.key = root/'key'; self.key.write_bytes(os.urandom(32)); self.key.chmod(0o600)
        self.store = ApprovalStore(root/'approvals.json')
        self.store.set_channels(['telegram'])
        self.store.set_contact('telegram', 'bot-a', '123456789', True)
        self.engine = Engine(self.key.read_bytes()); self.addCleanup(self.engine.close)
        self.routes = {'head': {'upstream':self.engine.base_url, 'profile':'owner-test'}}
        self.bridge = Bridge(self.store, self.key, self.routes)

    def claim(self, message='m1', **identity):
        record = {'id':message, 'channel':'telegram', 'account':'bot-a', 'contact':'123456789', **identity}
        return self.bridge.command('claim', {'session':'s', 'run':message, 'messages':[record]})

    def test_label_does_not_register_adapter_or_bypass_native_capability(self):
        for channel, contact in [('invented', '123456789'), ('local-terminal', 'uid:1000')]:
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                self.claim(channel=channel, contact=contact)
        for contact in ('@alice', '+123456789', '-100123456789'):
            with self.subTest(contact=contact), self.assertRaises(ValueError):
                self.claim(contact=contact)
        self.assertEqual(self.claim(account='other-bot'), {'mode':'off'})

    def test_media_binding_tool_inheritance_and_dynamic_revocation(self):
        cap = self.claim()['capability']
        content = [{'type':'text','text':'Describe'},
                   {'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]
        body = {'model':'test', 'messages':[{'role':'user','content':content}]}
        self.assertEqual(self.bridge.ticket(cap, body, 'head'), '')
        self.assertEqual(self.bridge.command('bind', {'capability':cap, 'content':content})['mode'], 'owner')
        self.assertTrue(self.bridge.ticket(cap, body, 'head'))
        self.store.set_tool('sessions_spawn', False)
        child = {'parent':cap, 'session':'child', 'run':'c1', 'tool':'sessions_spawn'}
        self.assertEqual(self.bridge.command('adopt', child), {'mode':'off'})
        self.store.set_tool('sessions_spawn', True)
        child_cap = self.bridge.command('adopt', child)['capability']
        self.assertTrue(self.bridge.ticket(child_cap, body, 'head'))
        self.store.set_channels([])
        self.assertEqual(self.bridge.ticket(cap, body, 'head'), '')
        self.assertEqual(self.bridge.ticket(child_cap, body, 'head'), '')

    def test_adapter_disable_invalidates_an_already_issued_ticket(self):
        cap = self.claim()['capability']
        self.bridge.command('bind', {'capability':cap, 'content':'hello'})
        body = {'model':'test', 'messages':[{'role':'user', 'content':'hello'}]}
        ticket = self.bridge.ticket(cap, body, 'head')
        self.assertTrue(ticket)
        self.store.set_channels([])
        self.assertEqual(self.bridge.policy.consume(ticket, body, 'head'), 'off')

    def test_openclaw_telegram_native_hooks(self):
        proxy = ProviderProxy(self.bridge.policy, self.routes, self.key, bridge=self.bridge)
        proxy.start(); self.addCleanup(proxy.close)
        script = Path(__file__).resolve().parents[2]/'openclaw-owner-mode/telegram-test.mjs'
        result = subprocess.run(['node', str(script), f'http://127.0.0.1:{proxy.port}', str(self.key)],
                                capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
