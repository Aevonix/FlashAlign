# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Native message rendering and composition, through the signed request boundary."""
import asyncio
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from test_gate_mode import GateBase
from owner_mode import hermes_ingress
from owner_mode.contacts import ApprovalStore
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import VerifiedTurn
from owner_mode.content import content_digest, matches

PHONE = '+15550000011'
OTHER = '+15550000012'


class MessageBindings(GateBase):
    def setUp(self):
        super().setUp()
        module = ModuleType('gateway.platforms.base')

        def merge(pending, key, event, *, merge_text=False):
            if key in pending:
                old = pending[key]
                old.text += '\n\n' + event.text
                old.media_urls.extend(event.media_urls)
            else:
                pending[key] = event

        module.merge_pending_message_event = merge
        self.modules = mock.patch.dict(sys.modules, {'gateway.platforms.base':module})
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.module = module
        self.store = ApprovalStore(Path(self.tmp.name)/'approvals.json')
        self.store.set_contact('whatsapp','default',PHONE,True)
        self.authority = NativeAuthority(self.store.path,VerifiedTurn)
        self.addCleanup(self.authority.close)
        self.policy.authority = self.authority
        self.adapter = SimpleNamespace(platform='whatsapp')
        adapter = self.adapter

        class Gateway:
            adapters = {'whatsapp':adapter}

            def _is_user_authorized_for_source(self, source):
                return True

            async def _prepare_profile_scoped_inbound_message_text(self, event):
                if isinstance(event.rendered, Exception):
                    raise event.rendered
                return event.rendered

            def _hmwa_apply_message_timestamp(self, event, message_text):
                return '[native timestamp] '+message_text, message_text, 1

        self.gateway = Gateway()

    def event(self, ident='m1', text='Please help.', kind='text', media=(), sender=PHONE, **metadata):
        source=SimpleNamespace(platform='whatsapp',chat_type='dm',chat_id='conversation',user_id=sender,
                               account_id='default',is_bot=False,_transport_adapter_ref=lambda:self.adapter,
                               delivered_via_upstream_relay=False)
        return SimpleNamespace(source=source,text=text,message_type=kind,media_urls=list(media),
            internal=False,message_id=ident,reply_to_message_id=None,reply_to_text=None,
            metadata={'whatsapp_native_type':'conversation',**metadata},rendered=text)

    def begin(self, event, rendered, *, sid='s', tid='t', timestamp=False):
        event.rendered=rendered
        self.authority.hear(event=event,gateway=self.gateway)
        observed=asyncio.run(self.gateway._prepare_profile_scoped_inbound_message_text(event))
        self.assertIs(observed,rendered)
        if timestamp:
            rendered=self.gateway._hmwa_apply_message_timestamp(event,observed)[0]
        route={'SESSION_PLATFORM':'whatsapp','SESSION_USER_ID':event.source.user_id,
               'SESSION_CHAT_ID':event.source.chat_id,'SESSION_CHAT_TYPE':'dm',
               'CRON_SESSION':'','SESSION_ACCOUNT_ID':'default','SESSION_MESSAGE_ID':event.message_id}
        history=[{'role':'user','content':rendered,'platform_message_id':event.message_id}]
        with mock.patch.object(hermes_ingress,'_route',return_value=route):
            self.policy.begin(session_id=sid,turn_id=tid,user_message=rendered,platform='whatsapp',
                              sender_id=event.source.user_id,conversation_history=history)
        return rendered

    def test_captionless_image_is_bound_and_forwarded_unchanged(self):
        event=self.event(text='',kind='photo',media=['/native/image.png'])
        content=[{'type':'text','text':'Describe this image.'},
                 {'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]
        self.begin(event,content)
        _,row=self.send(self.body(last=content))
        self.assertEqual(row['effective'],'owner-test')
        self.assertEqual(row['body']['messages'][-1]['content'],content)

    def test_changed_image_cannot_use_the_first_request_proof(self):
        event=self.event(text='',kind='photo',media=['/native/image.png'])
        content=[{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]
        self.begin(event,content)
        changed=copy.deepcopy(content);changed[0]['image_url']['url']='data:image/png;base64,BBBB'
        self.assertEqual(self.send(self.body(last=changed))[1]['effective'],'off')
        self.assertEqual(self.send(self.body(last=content))[1]['effective'],'owner-test')

    def test_forwarded_and_quoted_images_keep_the_sending_contact_verification(self):
        for kind in ('forwarded', 'quoted'):
            with self.subTest(kind=kind):
                event=self.event(ident=kind,text='',kind='photo',media=['/native/image.png'],
                                 forwarded=kind == 'forwarded')
                if kind == 'quoted':
                    event.reply_to_message_id='quoted-message'
                    event.reply_to_text='Another sender is quoted here.'
                content=[{'type':'text','text':'Host-rendered image context.'},
                         {'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]
                self.begin(event,content,sid=kind,tid=kind)
                _,row=self.send(self.body(last=content),sid=kind,tid=kind)
                self.assertEqual(row['effective'],'owner-test')
                self.assertEqual(row['body']['messages'][-1]['content'],content)

    def test_host_text_can_follow_bound_media_but_more_media_cannot(self):
        original=[{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]
        self.assertTrue(matches(original+[{'type':'text','text':'Host context'}],original,allow_text_context=True))
        self.assertFalse(matches(original+original,original,allow_text_context=True))

    def test_voice_transcript_and_caption_are_bound(self):
        event=self.event(text='Use my timezone.',kind='voice',media=['/native/audio.ogg'])
        rendered='"Schedule it tomorrow."\n\nUse my timezone.'
        self.begin(event,rendered)
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'],'owner-test')

    def test_native_timestamp_formatting_keeps_the_voice_binding(self):
        event=self.event(text='',kind='voice',media=['/native/audio.ogg'])
        rendered=self.begin(event,'A transcript.',timestamp=True)
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'],'owner-test')

    def test_document_without_caption_is_bound(self):
        event=self.event(text='',kind='document',media=['/native/report.pdf'])
        rendered='[Document attached: /native/report.pdf]'
        self.begin(event,rendered)
        self.assertEqual(self.send(self.body(last=rendered))[1]['effective'],'owner-test')

    def test_forward_and_quote_use_the_sending_contact(self):
        for index,metadata in enumerate([{'forwarded':True},{'reply':True}]):
            event=self.event(ident='m'+str(index),**metadata)
            if metadata.get('reply'):
                event.reply_to_message_id='quoted-message';event.reply_to_text='Some other message'
            rendered='Quoted or forwarded context.\n\nPlease help.'
            self.begin(event,rendered,sid='s'+str(index),tid='t'+str(index))
            self.assertEqual(self.send(self.body(last=rendered),sid='s'+str(index),tid='t'+str(index))[1]['effective'],'owner-test')

    def test_queued_group_consumes_every_native_member(self):
        first=self.event('first','First request.');second=self.event('second','Second request.')
        self.authority.hear(event=first,gateway=self.gateway)
        self.authority.hear(event=second,gateway=self.gateway)
        pending={'chat':first};self.module.merge_pending_message_event(pending,'chat',second,merge_text=True)
        self.begin(first,'[Queued messages]\n\nFirst request.\n\nSecond request.')
        self.assertEqual(self.send(self.body(last=first.rendered))[1]['effective'],'owner-test')
        self.begin(second,'Second request.',sid='replay',tid='replay')
        self.assertFalse(self.policy.is_owner_work('replay','replay'))

    def test_mixed_contact_group_cannot_promote_the_unapproved_member(self):
        first=self.event('first','First request.');second=self.event('second','Second request.',sender=OTHER)
        self.authority.hear(event=first,gateway=self.gateway)
        pending={'chat':first};self.module.merge_pending_message_event(pending,'chat',second,merge_text=True)
        self.begin(first,first.text)
        self.assertFalse(self.policy.is_owner_work('s','t'))

    def test_group_cannot_mix_even_two_approved_contacts(self):
        self.store.set_contact('whatsapp','default',OTHER,True)
        first=self.event('first','First request.');second=self.event('second','Second request.',sender=OTHER)
        self.authority.hear(event=first,gateway=self.gateway)
        pending={'chat':first};self.module.merge_pending_message_event(pending,'chat',second,merge_text=True)
        self.begin(first,first.text)
        self.assertFalse(self.policy.is_owner_work('s','t'))

    def test_custom_host_can_bind_an_explicit_native_message_group(self):
        first=self.event('first','First request.');second=self.event('second','Second request.')
        composed=self.event('first','First request.\n\nSecond request.')
        self.authority.record_render(event=composed,gateway=self.gateway,content=composed.text,
                                     members=[first,second])
        self.begin(composed,composed.text)
        self.assertEqual(self.send(self.body(last=composed.text))[1]['effective'],'owner-test')
        self.assertIn(('whatsapp','default',PHONE,'conversation','second'),self.authority.consumed)

    def test_renderer_with_an_unregistered_transport_cannot_replace_a_binding(self):
        event=self.event()
        self.authority.hear(event=event,gateway=self.gateway)
        event.source._transport_adapter_ref=lambda:SimpleNamespace(platform='whatsapp')
        self.authority.record_render(event=event,gateway=self.gateway,content='Changed content')
        self.assertIsNone(self.authority.rendered[self.authority._key(hermes_ingress.ingress_facts(event,self.gateway))])

    def test_unverified_media_and_reactions_remain_off(self):
        cases=[self.event('outsider','',kind='photo',media=['/x'],sender=OTHER),
               self.event('reaction','Like',reaction=True),
               self.event('echo','Hello',whatsapp_from_owner=True)]
        for index,event in enumerate(cases):
            self.begin(event,'Rendered content.',sid='s'+str(index),tid='t'+str(index))
            self.assertFalse(self.policy.is_owner_work('s'+str(index),'t'+str(index)))

    def test_revocation_also_applies_to_media_turns(self):
        event=self.event(text='',kind='voice',media=['/native/audio.ogg'])
        self.begin(event,'A transcript.')
        self.store.set_contact('whatsapp','default',PHONE,False)
        self.assertEqual(self.send(self.body(last='A transcript.'))[1]['effective'],'off')

    def test_renderer_failures_are_preserved_and_do_not_authorize_media(self):
        event=self.event(text='',kind='voice',media=['/native/audio.ogg'])
        self.authority.hear(event=event,gateway=self.gateway);event.rendered=ValueError('render failed')
        with self.assertRaisesRegex(ValueError,'render failed'):
            asyncio.run(self.gateway._prepare_profile_scoped_inbound_message_text(event))
        self.assertFalse(self.policy.is_owner_work('s','t'))

    def test_observer_restores_host_methods_on_unload(self):
        original=self.gateway._prepare_profile_scoped_inbound_message_text
        self.authority.hear(event=self.event(),gateway=self.gateway)
        self.assertIsNot(self.gateway._prepare_profile_scoped_inbound_message_text,original)
        self.authority.close()
        self.assertEqual(self.gateway._prepare_profile_scoped_inbound_message_text,original)


if __name__=='__main__':unittest.main()
