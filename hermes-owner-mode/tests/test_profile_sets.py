# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A route's profile set: per-profile access, failure and strength rules through the signing proxy."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

import httpx

from test_owner_mode import Runtime, Policy, ProviderProxy, TestAuthority
from owner_mode.bridge import Bridge
from owner_mode.client import Client, ProfileNotActive, profile_label, profile_names
from owner_mode.contacts import ApprovalStore
from owner_mode.policy import flatten_request
from owner_mode.profile_sets import (ProfileRule, applied, digest_check, label, route_profiles, scales_sent,
                                     selected, verified_label, wire)
from support import Engine


SET = [{'name': 'focus', 'access': 'always'},
       {'name': 'careful', 'access': 'always', 'scale': .5},
       {'name': 'owner-test'}]


class Declarations(unittest.TestCase):
    def test_one_profile_form_is_a_set_of_one_verified_optional_profile(self):
        self.assertEqual(route_profiles({'profile': 'owner-test'}), (ProfileRule('owner-test'),))
        self.assertEqual(route_profiles({'profile': 'owner-test', 'owner_mode_scale': .75}),
                         (ProfileRule('owner-test', scale=.75),))
        for route in ({}, {'profile': 'off'}, {'profile': ''}, {'profile': None}, {'profile': 7}):
            with self.subTest(route=route):
                self.assertEqual(route_profiles(route), ())

    def test_list_form_defaults_order_and_shorthand(self):
        rules = route_profiles({'profiles': ['focus', {'name': 'owner-test', 'access': 'verified', 'fail': 'required',
                                                       'scale': 1, 'digest': 'a' * 64}]})
        self.assertEqual(rules, (ProfileRule('focus'), ProfileRule('owner-test', 'verified', 'required', 1.0, 'a' * 64)))
        self.assertEqual(route_profiles({'profiles': []}), ())
        self.assertEqual(label(route_profiles({'profiles': SET})), 'focus,careful,owner-test')
        self.assertEqual(verified_label(route_profiles({'profiles': SET})), 'owner-test')
        self.assertEqual(verified_label(route_profiles({'profiles': SET[:2]})), 'off')

    def test_invalid_declarations_are_rejected(self):
        invalid = [
            {'profiles': ['a'], 'profile': 'a'}, {'profiles': ['a'], 'owner_mode_scale': 1},
            {'profiles': 'a'}, {'profiles': [7]}, {'profiles': [{'name': 'a', 'mode': 'always'}]},
            {'profiles': ['a', 'a']}, {'profiles': ['off']}, {'profiles': ['a,b']}, {'profiles': ['']},
            {'profiles': [{'name': 'a', 'access': 'owner'}]}, {'profiles': [{'name': 'a', 'fail': 'closed'}]},
            {'profiles': [{'name': 'a', 'scale': 0}]}, {'profiles': [{'name': 'a', 'scale': 1.5}]},
            {'profiles': [{'name': 'a', 'scale': True}]}, {'profiles': [{'name': 'a', 'digest': 'abc'}]},
            {'profiles': [str(index) for index in range(9)]},
            {'profile': 'a', 'owner_mode_scale': 2}, {'profile': 'a', 'owner_mode_scale': 'high'},
        ]
        for route in invalid:
            with self.subTest(route=route), self.assertRaises(ValueError):
                route_profiles(route)

    def test_wire_keeps_the_scalar_form_for_one_profile(self):
        rules = route_profiles({'profiles': SET})
        self.assertEqual(wire(()), {})
        self.assertEqual(wire(rules[2:]), {'owner_mode': 'owner-test'})
        self.assertEqual(wire((ProfileRule('a', scale=.25),)), {'owner_mode': 'a', 'owner_mode_scale': .25})
        self.assertEqual(wire(rules), {'owner_mode': ['focus', 'careful', 'owner-test'],
                                       'owner_mode_scale': [1.0, .5, 1.0]})
        self.assertEqual(wire((rules[0], rules[2])), {'owner_mode': ['focus', 'owner-test']})
        self.assertEqual(scales_sent(rules), '1,0.5,1')
        self.assertEqual(scales_sent((ProfileRule('a', scale=.25),)), .25)
        self.assertIsNone(scales_sent(rules[2:]))
        self.assertIsNone(scales_sent(()))

    def test_labels_map_back_to_declared_rules(self):
        rules = route_profiles({'profiles': SET})
        self.assertEqual(selected(rules, 'focus,careful'), rules[:2])
        self.assertEqual(selected(rules, 'off'), ())
        self.assertEqual(applied(rules, 'focus,owner-test'), (rules[0], rules[2]))
        for effective in ('off', 'none', '', None, 'unknown'):
            with self.subTest(effective=effective):
                self.assertEqual(applied(rules, effective), ())
        legacy = route_profiles({'profile': 'legacy, name'})
        self.assertEqual(selected(legacy, 'legacy, name'), legacy)
        self.assertEqual(applied(legacy, 'legacy, name'), legacy)


    def test_digest_check_reports_drift_and_what_could_not_be_checked(self):
        a, b, c = 'a' * 64, 'b' * 64, 'c' * 64
        rules = route_profiles({'profiles': [{'name': 'focus', 'digest': a}, {'name': 'careful'},
                                             {'name': 'owner-test', 'digest': c}]})
        self.assertEqual(digest_check(rules, 'focus,careful,owner-test', ','.join((a, b, c))), ((), ()))
        self.assertEqual(digest_check(rules, 'focus,careful,owner-test', ','.join((b, b, b))),
                         (('focus', 'owner-test'), ()))
        self.assertEqual(digest_check(rules, 'focus,owner-test', ','.join((a, b))), (('owner-test',), ()))
        # No digests, or a list that cannot be matched to the applied profiles, proves nothing either way.
        self.assertEqual(digest_check(rules, 'focus,careful,owner-test', None), ((), ('focus', 'owner-test')))
        self.assertEqual(digest_check(rules, 'focus,careful,owner-test', a), ((), ('focus', 'owner-test')))
        self.assertEqual(digest_check(rules, 'careful', None), ((), ()))
        self.assertEqual(digest_check(rules, 'off', None), ((), ()))


class SetProxy(unittest.TestCase):
    profiles = SET
    loaded = ('focus', 'careful', 'owner-test')

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.key = self.root/'key'
        self.key.write_bytes(b'k'*32)
        self.key.chmod(0o600)
        self.store = ApprovalStore(self.root/'approvals.json')
        self.store.set_mode('verified')
        self.engine = Engine(b'k'*32, profiles=self.loaded)
        self.engine.accept_sets = True
        self.addCleanup(self.engine.close)
        self.policy = Policy(authority=TestAuthority(), tool_policy=self.store.tool_enabled)
        self.rows = []
        self.audit_recorded = threading.Event()
        def audit(row):
            self.rows.append(row)
            self.audit_recorded.set()
        self.routes = {'head': {'upstream': self.engine.base_url, 'profiles': self.profiles}}
        self.proxy = ProviderProxy(self.policy, self.routes, self.key, approvals=self.store, audit=audit)
        self.proxy.start()
        self.addCleanup(self.proxy.close)
        self.runtime = Runtime(self.policy, self.proxy)
        self.body = {'model': 'bench', 'messages': [{'role': 'user', 'content': 'Hello'}]}

    def send(self, *, verified=False, session='s', turn='t'):
        self.audit_recorded.clear()
        if verified:
            self.policy.begin(session_id=session, turn_id=turn, user_message='Hello', platform='whatsapp')
        request = self.runtime.request(self.body, session_id=session, turn_id=turn,
                                       base_url=self.proxy.base_url('head'))['request']
        with httpx.Client(timeout=5) as client:
            response = client.post(self.proxy.base_url('head')+'/chat/completions', json=flatten_request(request),
                                   headers=request.get('extra_headers', {}))
        self.assertTrue(self.audit_recorded.wait(5), 'Proxy audit callback did not complete')
        return response


class AccessRules(SetProxy):
    def test_unverified_requests_get_only_the_always_profiles(self):
        response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,careful')
        row = self.engine.records[-1]
        self.assertEqual(row['auth_reason'], 'valid')
        self.assertEqual(row['body']['owner_mode'], ['focus', 'careful'])
        self.assertEqual(row['body']['owner_mode_scale'], [1.0, .5])
        self.assertEqual(self.rows[-1]['owner_mode_sent'], 'focus,careful')
        self.assertEqual(self.rows[-1]['owner_mode_scale_sent'], '1,0.5')
        self.assertEqual(self.rows[-1]['effective'], 'focus,careful')
        self.assertFalse(self.rows[-1]['owner_verified'])
        self.assertEqual(self.rows[-1]['activation_mode'], 'verified')

    def test_verified_turns_get_the_whole_set_in_declared_order(self):
        response = self.send(verified=True)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,careful,owner-test')
        self.assertEqual(self.engine.records[-1]['body']['owner_mode'], ['focus', 'careful', 'owner-test'])
        self.assertEqual(self.engine.records[-1]['body']['owner_mode_scale'], [1.0, .5, 1.0])
        self.assertTrue(self.rows[-1]['owner_verified'])
        self.assertEqual(self.rows[-1]['decision'], 'owner_clean')

    def test_callers_cannot_choose_profiles_or_strengths(self):
        self.body = {**self.body, 'owner_mode': ['owner-test'], 'owner_mode_scale': [1]}
        with httpx.Client(timeout=5) as client:
            response = client.post(self.proxy.base_url('head')+'/chat/completions', json=self.body,
                                   headers={'X-Owner-Mode-Ticket': 'forged', 'X-Owner-Mode-Auth': 'forged'})
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,careful')
        self.assertEqual(self.engine.records[-1]['body']['owner_mode'], ['focus', 'careful'])

    def test_a_tool_that_does_not_inherit_keeps_the_always_profiles_only(self):
        self.store.set_tool('plain_tool', False)
        self.policy.begin(session_id='s', turn_id='t', user_message='Hello', platform='whatsapp')
        response = self.runtime.execute_tool({}, lambda _: self.send(), tool_name='plain_tool')
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,careful')
        self.assertEqual(self.rows[-1]['decision'], 'tool_owner_mode_disabled')

    def test_pause_turns_every_profile_off_and_always_on_applies_every_profile(self):
        self.store.set_enabled(False)
        self.assertEqual(self.send(verified=True).headers['X-Owner-Mode-Effective'], 'off')
        self.assertNotIn('owner_mode', self.engine.records[-1]['body'])
        self.store.set_enabled(True)
        self.store.set_mode('always_on')
        response = self.send(session='other', turn='u')
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,careful,owner-test')
        self.assertEqual(self.rows[-1]['decision'], 'always_on')

    def test_bridge_tickets_grant_the_verified_profiles(self):
        bridge_key = self.root/'bridge-key'
        bridge_key.write_bytes(b'b'*32)
        bridge_key.chmod(0o600)
        routes = {**self.routes, 'open': {'upstream': self.engine.base_url,
                                          'profiles': [{'name': 'focus', 'access': 'always'}]}}
        bridge = Bridge(self.store, bridge_key, routes)
        self.store.set_contact('rcs', 'default', '+10000000000', True)
        claim = bridge.command('begin', dict(session='session', run='turn', channel='rcs',
                               account='default', contact='+10000000000', text='Hello'))
        ticket = bridge.ticket(claim['capability'], self.body, 'head')
        self.assertTrue(ticket)
        # The ticket grants the route's verified profiles; its always profiles need none.
        self.assertEqual(bridge.policy.consume(ticket, self.body, 'head'), 'owner-test')
        info = {}
        self.assertFalse(bridge.ticket(claim['capability'], self.body, 'open', info=info))
        self.assertEqual(info['decision'], 'route_profile_off')


class DigestPins(SetProxy):
    profiles = [{'name': 'focus', 'access': 'always', 'digest': 'a' * 64},
                {'name': 'careful', 'access': 'always'},
                {'name': 'owner-test', 'digest': 'c' * 64}]

    def test_matching_digests_are_recorded_without_a_warning(self):
        self.engine.digests = {'focus': 'a' * 64, 'careful': 'b' * 64, 'owner-test': 'c' * 64}
        with self.assertNoLogs('owner_mode.proxy', 'WARNING'):
            response = self.send(verified=True)
        self.assertEqual(response.headers['X-FlashAlign-Profile-Digest'], ','.join(('a' * 64, 'b' * 64, 'c' * 64)))
        row = self.rows[-1]
        self.assertEqual(row['profile_digest'], ','.join(('a' * 64, 'b' * 64, 'c' * 64)))
        self.assertIsNone(row['digest_drift'])
        self.assertIsNone(row['digest_unchecked'])

    def test_a_changed_profile_is_logged_and_audited_and_the_request_still_completes(self):
        self.engine.digests = {'focus': 'a' * 64, 'careful': 'b' * 64, 'owner-test': 'd' * 64}
        with self.assertLogs('owner_mode.proxy', 'ERROR') as logs:
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.rows[-1]['digest_drift'], 'owner-test')
        self.assertIn('OWNER_MODE_DIGEST_DRIFT route=head profiles=owner-test', logs.output[0])
        with self.assertNoLogs('owner_mode.proxy', 'WARNING'):
            self.send(session='other', turn='u')
        self.assertIsNone(self.rows[-1]['digest_drift'])

    def test_a_pin_the_engine_gives_no_digest_for_is_marked_unchecked(self):
        response = self.send(verified=True)
        self.assertNotIn('X-FlashAlign-Profile-Digest', response.headers)
        row = self.rows[-1]
        self.assertEqual((row['profile_digest'], row['digest_drift'], row['digest_unchecked']),
                         (None, None, 'focus,owner-test'))

    def test_rows_without_a_profile_carry_no_digest_fields(self):
        self.store.set_enabled(False)
        self.send()
        row = self.rows[-1]
        self.assertEqual((row['profile_digest'], row['digest_drift'], row['digest_unchecked']), (None, None, None))


class FailureRules(SetProxy):
    def test_an_optional_profile_the_engine_lacks_is_dropped_and_reported(self):
        self.engine.profiles = {'focus', 'careful'}
        with self.assertLogs('owner_mode.proxy', 'ERROR') as logs:
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,careful')
        self.assertEqual([row['body'].get('owner_mode') for row in self.engine.records[-2:]],
                         [['focus', 'careful', 'owner-test'], ['focus', 'careful']])
        self.assertEqual(self.engine.records[-1]['auth_reason'], 'valid')
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_profile_dropped')
        self.assertEqual(self.rows[-1]['owner_mode_sent'], 'focus,careful')
        self.assertIn('dropped=owner-test kept=focus,careful', logs.output[0])

    def test_when_every_optional_profile_is_missing_the_request_is_resent_as_stock(self):
        self.engine.profiles = set()
        with self.assertLogs('owner_mode.proxy', 'ERROR'):
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'off')
        self.assertNotIn('owner_mode', self.engine.records[-1]['body'])
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_profile_dropped')
        self.assertEqual(self.rows[-1]['owner_mode_sent'], 'off')

    def test_an_engine_without_profiles_serves_optional_sets_as_stock(self):
        self.engine.refuse_owner_mode = True
        with self.assertLogs('owner_mode.proxy', 'WARNING'):
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'off')
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_unavailable')

    def test_an_engine_that_takes_one_profile_per_request_rejects_a_set(self):
        self.engine.accept_sets = False
        with self.assertLogs('owner_mode.proxy', 'ERROR') as logs:
            response = self.send()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_rejected_bad_owner_mode_type')
        self.assertIn('this engine accepts one profile per request', logs.output[0])

    def test_an_unconfirmed_optional_profile_passes_with_a_warning(self):
        self.engine.unapplied = {'careful'}
        with self.assertLogs('owner_mode.proxy', 'WARNING') as logs:
            response = self.send()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus')
        self.assertEqual(self.rows[-1]['effective'], 'focus')
        self.assertIn('OWNER_MODE_UNCONFIRMED', logs.output[0])


class RequiredRules(SetProxy):
    profiles = [{'name': 'focus', 'access': 'always', 'fail': 'required'},
                {'name': 'careful', 'access': 'always'},
                {'name': 'owner-test', 'fail': 'required'}]

    def test_a_required_profile_the_engine_lacks_fails_the_request(self):
        self.engine.profiles = {'focus', 'careful'}
        with self.assertLogs('owner_mode.proxy', 'ERROR'):
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_rejected_unknown_profile')
        # The same engine still serves requests that do not select the missing required profile.
        self.assertEqual(self.send(session='other', turn='u').headers['X-Owner-Mode-Effective'], 'focus,careful')

    def test_an_optional_profile_is_dropped_while_required_profiles_stay(self):
        self.engine.profiles = {'focus', 'owner-test'}
        with self.assertLogs('owner_mode.proxy', 'ERROR'):
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'focus,owner-test')

    def test_an_engine_without_profiles_fails_a_required_set(self):
        self.engine.refuse_owner_mode = True
        with self.assertLogs('owner_mode.proxy', 'ERROR'):
            response = self.send()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_rejected_no_profiles')

    def test_an_unconfirmed_required_profile_fails_the_request(self):
        self.engine.unapplied = {'focus'}
        response = self.send()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.rows[-1]['decision'], 'required_profile_unconfirmed')
        self.engine.unapplied = {'careful'}
        with self.assertLogs('owner_mode.proxy', 'WARNING'):
            self.assertEqual(self.send().status_code, 200)

    def test_a_required_profile_needs_the_signing_key(self):
        self.key.unlink()
        self.assertEqual(self.send().status_code, 502)
        self.assertEqual(len(self.engine.records), 0)


class SingleProfile(SetProxy):
    profiles = None
    loaded = ('owner-test',)

    def setUp(self):
        super().setUp()
        self.proxy.close()
        self.routes = {'head': {'upstream': self.engine.base_url, 'profile': 'owner-test', 'owner_mode_scale': .75}}
        self.proxy = ProviderProxy(self.policy, self.routes, self.key, approvals=self.store,
                                   audit=lambda row: (self.rows.append(row), self.audit_recorded.set()))
        self.proxy.start()
        self.addCleanup(self.proxy.close)
        self.runtime = Runtime(self.policy, self.proxy)

    def test_the_wire_form_is_unchanged(self):
        self.assertEqual(self.send().headers['X-Owner-Mode-Effective'], 'off')
        self.assertNotIn('owner_mode', self.engine.records[-1]['body'])
        self.assertEqual(self.send(verified=True).headers['X-Owner-Mode-Effective'], 'owner-test')
        body = self.engine.records[-1]['body']
        self.assertEqual((body['owner_mode'], body['owner_mode_scale']), ('owner-test', .75))
        self.assertEqual(self.rows[-1]['owner_mode_scale_sent'], .75)

    def test_an_optional_profile_the_engine_lacks_is_served_as_stock(self):
        self.engine.profiles = set()
        with self.assertLogs('owner_mode.proxy', 'ERROR'):
            response = self.send(verified=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Owner-Mode-Effective'], 'off')
        self.assertEqual(self.rows[-1]['decision'], 'owner_mode_profile_dropped')


class DirectClient(unittest.TestCase):
    def test_names_and_labels(self):
        self.assertEqual(profile_names('a'), ('a',))
        self.assertEqual(profile_names(['a', 'b']), ('a', 'b'))
        self.assertEqual(profile_label(('a', 'b')), 'a,b')
        for value in (None, [], ['a'], ['a', 'a'], ['a', 'off'], ['a', 'b,c'], ['a', 7], '', ['a'] * 9, ['a', '']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile_names(value)

    def test_a_set_is_signed_and_must_be_confirmed_as_a_whole(self):
        with tempfile.TemporaryDirectory() as temporary:
            key = Path(temporary)/'key'
            key.write_bytes(b'k'*32)
            key.chmod(0o600)
            engine = Engine(b'k'*32, profiles=('focus', 'owner-test'))
            engine.accept_sets = True
            self.addCleanup(engine.close)
            with Client(engine.base_url, 'bench', key_file=key) as client:
                messages = [{'role': 'user', 'content': 'Hello'}]
                response = client.chat_response(messages, profile=['focus', 'owner-test'], scale=[.5, 1])
                self.assertEqual(response.metadata['effective_profile'], 'focus,owner-test')
                body = engine.records[-1]['body']
                self.assertEqual((body['owner_mode'], body['owner_mode_scale']), (['focus', 'owner-test'], [.5, 1]))
                self.assertEqual(engine.records[-1]['auth_reason'], 'valid')
                client.chat(messages, profile=['focus', 'owner-test'])
                self.assertEqual(engine.records[-1]['body']['owner_mode_scale'], [1., 1.])
                engine.unapplied = {'focus'}
                with self.assertRaises(ProfileNotActive):
                    client.chat(messages, profile=['focus', 'owner-test'])
                for scale in ([1], [1, 2], [1, 'x'], 0):
                    with self.subTest(scale=scale), self.assertRaises(ValueError):
                        client.prepare({}, profile=['focus', 'owner-test'], scale=scale)
                raw, _ = client.prepare({}, profile='focus', scale=.5)
                self.assertEqual(json.loads(raw)['owner_mode_scale'], .5)


if __name__ == '__main__':
    unittest.main()
