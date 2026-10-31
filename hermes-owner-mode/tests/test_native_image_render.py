# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Bind the final native image renderer, not only the earlier caption seam."""
import asyncio
import copy
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import threading
from types import ModuleType, SimpleNamespace
from unittest import mock

from test_gate_mode import GateBase
from owner_mode import hermes_ingress
from owner_mode.contacts import ApprovalStore
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import VerifiedTurn


class FinalNativeImageBindings(GateBase):
    def setUp(self):
        super().setUp()
        base = ModuleType('gateway.platforms.base')
        def merge(pending, key, event):
            if key in pending:
                pending[key].text += '\n\n'+event.text
                pending[key].media_urls.extend(event.media_urls)
            else:
                pending[key] = event
        base.merge_pending_message_event = merge
        self.base = base
        module = ModuleType('gateway.run_turn_runner')

        class TurnRunner:
            def __init__(self, gateway, context):
                self._runner, self._ctx = gateway, context

            def _native_image_run_message(self):
                if getattr(self._ctx, 'barrier', None):
                    self._ctx.barrier.wait(timeout=5)
                return self._ctx.rendered

            def _run_conversation_with_approval(self, callback):
                content = self._native_image_run_message()
                callback(content)
                return content

        module.TurnRunner = TurnRunner
        self.original = TurnRunner._native_image_run_message
        self.turn_type = TurnRunner
        patcher = mock.patch.dict(sys.modules, {'gateway.platforms.base': base,
                                               'gateway.run_turn_runner': module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = ApprovalStore(self.keyfile.parent/'approvals.json')
        self.store.set_contact('whatsapp', 'default', '+15550000011', True)
        self.authority = NativeAuthority(self.store.path, VerifiedTurn)
        self.addCleanup(self.authority.close)
        self.policy.authority = self.authority
        self.route_context = ContextVar('test_native_image_route', default={})
        route_patch = mock.patch.object(hermes_ingress, '_route', side_effect=self.route_context.get)
        route_patch.start()
        self.addCleanup(route_patch.stop)
        self.adapter = SimpleNamespace(platform='whatsapp')
        adapter = self.adapter

        class Gateway:
            adapters = {'whatsapp': adapter}

            def _is_user_authorized_for_source(self, source):
                return True

            async def _prepare_profile_scoped_inbound_message_text(self, event):
                return event.text

        self.gateway = Gateway()

    def event(self, caption='', phone='+15550000011'):
        source = SimpleNamespace(platform='whatsapp', chat_type='dm', chat_id='synthetic-chat',
                                 user_id=phone, account_id='default', is_bot=False,
                                 _transport_adapter_ref=lambda: self.adapter, delivered_via_upstream_relay=False)
        return SimpleNamespace(source=source, text=caption, message_type='photo',
                               media_urls=['/synthetic/blue.png'], internal=False,
                               message_id='native-image', reply_to_message_id=None, reply_to_text=None,
                               metadata={'whatsapp_native_type':'imageMessage'})

    def begin(self, event, rendered, *, render_message_id='native-image', revoke=False):
        def authorize(_):
            if revoke:
                self.store.set_contact('whatsapp', 'default', event.source.user_id, False)
            self.authorize(event, rendered)
        self.render(event, rendered, render_message_id=render_message_id, callback=authorize)

    def render(self, event, rendered, *, render_message_id=None, callback=lambda _: None):
        self.authority.hear(event=event, gateway=self.gateway)
        self.assertEqual(asyncio.run(self.gateway._prepare_profile_scoped_inbound_message_text(event)), event.text)
        context = SimpleNamespace(source=event.source, inbound_message_id=render_message_id or event.message_id,
                                  rendered=rendered)
        self.assertIs(self.turn_type(self.gateway, context)._run_conversation_with_approval(callback), rendered)

    def authorize(self, event, rendered, *, missing_id=False, row_id=None, prior=(), sid='s'):
        route = {'SESSION_PLATFORM':'whatsapp', 'SESSION_USER_ID':event.source.user_id,
                 'SESSION_CHAT_ID':event.source.chat_id, 'SESSION_CHAT_TYPE':'dm',
                 'CRON_SESSION':'', 'SESSION_ACCOUNT_ID':'default', 'SESSION_MESSAGE_ID':event.message_id}
        token = self.route_context.set(route)
        try:
            current = {'role':'user', 'content':rendered}
            if not missing_id:
                current['platform_message_id'] = row_id or event.message_id
            self.runtime.begin(session_id=sid, turn_id='t', user_message=event.text, platform='whatsapp',
                              sender_id=event.source.user_id,
                              conversation_history=[*prior, current])
        finally:
            self.route_context.reset(token)

    @staticmethod
    def image(caption=''):
        return [{'type':'text', 'text':caption or 'Image attached'},
                {'type':'image_url', 'image_url':{'url':'data:image/png;base64,AAAA'}}]

    def test_captionless_final_render_is_authorized_and_pixels_remain_bound(self):
        rendered = self.image()
        self.begin(self.event(), rendered)
        changed = copy.deepcopy(rendered)
        changed[-1]['image_url']['url'] = 'data:image/png;base64,BBBB'
        self.assertEqual(self.send(self.body(last=changed))[1]['effective'], 'off')
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'owner-test')

    def test_caption_and_image_final_render_is_authorized(self):
        rendered = self.image('Please describe it.')
        self.begin(self.event('Please describe it.'), rendered)
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'owner-test')

    def test_unknown_or_revoked_image_cannot_be_promoted(self):
        for unknown, revoke in [(True, False), (False, True)]:
            with self.subTest(unknown=unknown, revoke=revoke):
                rendered = self.image()
                self.begin(self.event(phone='+15550000012' if unknown else '+15550000011'), rendered,
                           revoke=revoke)
                self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'off')

    def test_reply_anchor_cannot_replace_raw_inbound_identity(self):
        rendered = self.image()
        self.begin(self.event(), rendered, render_message_id='different-raw-message')
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'off')

    def test_unload_restores_native_renderer(self):
        self.authority.hear(event=self.event(), gateway=self.gateway)
        self.assertIsNot(self.turn_type._native_image_run_message, self.original)
        self.authority.close()
        self.assertIs(self.turn_type._native_image_run_message, self.original)

    def test_concurrent_native_images_do_not_exchange_identity_or_content(self):
        approved, unknown = self.event(), self.event(phone='+15550000012')
        approved.message_id, unknown.message_id = 'approved-image', 'unknown-image'
        first, second = self.image('Approved caption'), self.image('Unknown caption')
        second[-1]['image_url']['url'] = 'data:image/png;base64,BBBB'
        for event in (approved, unknown):
            self.authority.hear(event=event, gateway=self.gateway)
        barrier = threading.Barrier(2)
        turns = [self.turn_type(self.gateway, SimpleNamespace(source=event.source,
                  inbound_message_id=event.message_id, rendered=content, barrier=barrier))
                 for event, content in ((approved, first), (unknown, second))]
        def run(item):
            turn, event, content, sid = item
            return turn._run_conversation_with_approval(lambda _: self.authorize(event, content, sid=sid))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, zip(turns, (approved, unknown), (first, second), ('s', 'unknown'))))
        self.assertIs(results[0], first)
        self.assertIs(results[1], second)
        self.assertEqual(self.send(self.body(last=second))[1]['effective'], 'off')
        self.assertEqual(self.send(self.body(last=first))[1]['effective'], 'owner-test')
        self.assertEqual(self.send(self.body(last=second), sid='unknown')[1]['effective'], 'off')

    def test_projected_hook_rejects_missing_stale_and_ambiguous_current_rows(self):
        for index, options in enumerate(({'missing_id':True}, {'row_id':'older-image'}, {'duplicate':True})):
            with self.subTest(options=options):
                event, rendered = self.event(), self.image()
                event.message_id = 'current-'+str(index)
                if options.get('duplicate'):
                    options = {'prior':[{'role':'user', 'content':rendered,
                                         'platform_message_id':event.message_id}]}
                self.render(event, rendered, callback=lambda _: self.authorize(event, rendered, **options))
                self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'off')

    def test_changed_current_row_cannot_replace_the_native_render_receipt(self):
        event, rendered = self.event(), self.image()
        changed = copy.deepcopy(rendered)
        changed[-1]['image_url']['url'] = 'data:image/png;base64,BBBB'
        self.render(event, rendered, callback=lambda _: self.authorize(event, changed))
        self.assertEqual(self.send(self.body(last=changed))[1]['effective'], 'off')

    def test_current_text_does_not_select_an_older_image_row(self):
        event = self.event('Current text')
        event.message_type, event.media_urls = 'text', []
        event.metadata['whatsapp_native_type'] = 'conversation'
        self.render(event, event.text, callback=lambda _: self.authorize(
            event, event.text, prior=[{'role':'user', 'content':self.image(), 'platform_message_id':'older-image'}]))
        self.assertEqual(self.send(self.body(last=event.text))[1]['effective'], 'owner-test')

    def test_child_without_native_route_keeps_explicit_adoption(self):
        rendered = self.image()
        self.begin(self.event(), rendered)
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'owner-test')
        self.runtime.adopt(parent_session_id='s', parent_turn_id='t', child_session_id='child')
        self.runtime.begin(session_id='child', turn_id='child-turn', parent_session_id='s',
                           user_message='Harmless delegated task', platform='whatsapp')
        self.assertTrue(self.policy.is_owner_work('child', 'child-turn'))

    def test_composed_images_require_every_native_member(self):
        first, second = self.event('First'), self.event('Second')
        first.message_id, second.message_id = 'first', 'second'
        self.authority.hear(event=first, gateway=self.gateway)
        self.authority.hear(event=second, gateway=self.gateway)
        pending = {'chat':first}
        self.base.merge_pending_message_event(pending, 'chat', second)
        rendered = self.image('First and second') + [self.image()[-1]]
        self.render(first, rendered, callback=lambda _: self.authorize(first, rendered))
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'owner-test')

    def test_composed_unapproved_or_untraceable_image_stays_off(self):
        for unapproved in (True, False):
            first = self.event('First')
            second = self.event('Second', phone='+15550000012' if unapproved else '+15550000011')
            first.message_id, second.message_id = str(unapproved)+'-first', str(unapproved)+'-second'
            second.internal = not unapproved
            self.authority.hear(event=first, gateway=self.gateway)
            self.authority.hear(event=second, gateway=self.gateway)
            pending = {'chat':first}
            self.base.merge_pending_message_event(pending, 'chat', second)
            rendered = self.image('Composed') + [self.image()[-1]]
            self.render(first, rendered, callback=lambda _: self.authorize(first, rendered))
            self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'off')

    def test_unconsumed_receipt_expires_when_native_call_returns_or_raises(self):
        def fail(_):
            raise RuntimeError('synthetic interrupted conversation')

        for raises in (False, True):
            event, rendered = self.event(), self.image()
            event.message_id = 'unfinished-'+str(raises)
            if raises:
                with self.assertRaises(RuntimeError):
                    self.render(event, rendered, callback=fail)
            else:
                self.render(event, rendered)
            self.assertEqual(self.authority.render_scope.get(), (False, None))
            self.authorize(event, rendered)
            self.assertEqual(self.send(self.body(last=rendered))[1]['effective'], 'off')
