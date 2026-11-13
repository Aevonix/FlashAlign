# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Pause and resume change only the existing store flag."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_owner_mode
from owner_mode.contacts import ApprovalStore
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy
from flashalign.__main__ import main


class PauseResume(unittest.TestCase):
    def test_commands_preserve_access_and_update_live_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ApprovalStore(root / 'approvals.json')
            key = root / 'engine.key'
            key.write_bytes(os.urandom(32)); key.chmod(0o600)
            store.set_contact('rcs', 'default', '+10000000000', True)
            grant = store.approved('rcs', 'default', '+10000000000')
            authority = SimpleNamespace(
                verify_turn=lambda **kw: VerifiedTurn(kw['session_id'], kw['turn_id'], 'rcs',
                                                       kw['user_message'], grant_id=grant),
                is_active=lambda proof: store.active(proof.grant_id))
            policy = Policy(authority=authority, channel_policy=store.channel_enabled)
            with mock.patch('owner_mode.proxy.ThreadingHTTPServer'):
                proxy = ProviderProxy(policy, {'head': {'upstream': 'http://127.0.0.1:8030/v1',
                                                       'profile': 'owner-test'}}, key, approvals=store)
            self.addCleanup(proxy.close)
            body = {'model': 'bench', 'messages': [{'role': 'user', 'content': 'Hello'}]}
            for mode in ('verified', 'always_on'):
                with self.subTest(mode=mode):
                    store.set_mode(mode)
                    before = store.read()
                    with redirect_stdout(io.StringIO()):
                        main(['pause', '--store', str(store.path)])
                    self.assertEqual(store.read(), {**before, 'enabled': False})
                    self.assertIsNone(store.approved('rcs', 'default', '+10000000000'))
                    info = {}
                    self.assertEqual(proxy.prepare('head', body, info=info)[2], 'off')
                    self.assertEqual(info['decision'], 'deployment_disabled')
                    with redirect_stdout(io.StringIO()):
                        main(['resume', '--store', str(store.path)])
                    self.assertEqual(store.read(), before)
                    self.assertEqual(store.approved('rcs', 'default', '+10000000000'), grant)
                    self.assertIsNone(store.approved('rcs', 'other', '+10000000000'))
                    policy.begin(session_id='session', turn_id='turn', user_message='Hello', platform='rcs')
                    ticket = policy.issue(body, session_id='session', turn_id='turn', profile='owner-test', route='head')
                    self.assertEqual(proxy.prepare('head', body, ticket)[2], 'owner-test')
