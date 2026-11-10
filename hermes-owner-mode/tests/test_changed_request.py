# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Changed bodies remain unsigned and have a fixed diagnostic reason."""
from types import SimpleNamespace
import unittest
from unittest import mock

import test_owner_mode
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy


class ChangedRequest(unittest.TestCase):
    def test_digest_mismatch_is_reported_without_releasing_identity(self):
        authority = SimpleNamespace(verify_turn=lambda **kw: VerifiedTurn(
            kw['session_id'], kw['turn_id'], 'whatsapp', kw['user_message']))
        policy = Policy(authority=authority)
        policy.begin(session_id='session', turn_id='turn', platform='whatsapp', user_message='Hello')
        body = {'model': 'bench', 'messages': [{'role': 'user', 'content': 'Hello'}]}
        ticket = policy.issue(body, session_id='session', turn_id='turn', profile='owner-test', route='head')
        self.assertTrue(ticket)
        context = policy.context_ticket(body, session_id='session', turn_id='turn', route='head',
                                        info={'owner_verified': True, 'platform': 'whatsapp'})
        changed = dict(body, model='changed')
        info = policy.consume_context(context, changed, 'head')
        self.assertEqual(info, {'decision': 'request_changed'})
        rows = []
        with mock.patch('owner_mode.proxy.ThreadingHTTPServer'):
            proxy = ProviderProxy(policy, {'head': {'upstream': 'http://127.0.0.1:8030/v1',
                                                   'profile': 'owner-test'}}, '', audit=rows.append)
        self.addCleanup(proxy.close)
        # Even without the context ticket, the authorization ticket reports its mismatch.
        info = {}
        raw, headers, effective = proxy.prepare('head', changed, ticket, info=info)
        proxy.record('head', effective, effective, info=info)
        self.assertEqual(effective, 'off')
        self.assertNotIn('X-Owner-Mode-Auth', headers)
        self.assertEqual(rows[0]['decision'], 'request_changed')
        self.assertFalse(rows[0]['owner_verified'])
        self.assertEqual(policy.consume(ticket, body, 'head'), 'off')
        self.assertEqual(policy.consume_context(context, body, 'head'), {})
