# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fallbacks emit fixed diagnostics without copying caller data."""
from types import SimpleNamespace
import unittest
from unittest import mock

import test_plugin_registration as fixtures
from owner_mode import plugin
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import VerifiedTurn


class FallbackDiagnostics(unittest.TestCase):
    setUp = fixtures.ProcessRegistrationTests.setUp
    context = fixtures.ProcessRegistrationTests.context

    def check_fallback(self, reason, action, rows):
        with self.assertLogs('owner_mode', level='WARNING') as captured:
            action()
        self.assertEqual([row.getMessage() for row in captured.records],
                         ['FlashAlign fallback: ' + reason])
        self.assertEqual(rows, [{'decision': reason}])

    def test_unmatched_route(self):
        rows = []
        proxy = SimpleNamespace(routes={'head': {}}, base_url=lambda _: 'http://127.0.0.1:8030/head/v1',
                                audit=rows.append)
        runtime = plugin.Runtime(None, proxy)
        self.check_fallback('route_mismatch', lambda: runtime.request(
            {'messages': []}, base_url='http://localhost:8030/head/v1'), rows)

    def test_request_middleware_exception(self):
        rows = []
        proxy = SimpleNamespace(routes={'head': {}}, base_url=lambda _: 'route', audit=rows.append)
        policy = SimpleNamespace(is_owner_work=mock.Mock(side_effect=ValueError('unlogged detail')))
        runtime = plugin.Runtime(policy, proxy)
        self.check_fallback('request_middleware_exception', lambda: runtime.request(
            {'messages': []}, base_url='route'), rows)

    def test_nonserving_process(self):
        rows = []
        ctx = self.context(serve_argv=['gateway', 'run'])
        with mock.patch.object(plugin, 'AuditLog', return_value=rows.append), \
             mock.patch.object(plugin.sys, 'argv', ['fixture', 'different']):
            self.check_fallback('serving_process_mismatch', lambda: plugin.register(ctx), rows)
        self.assertFalse(ctx.hooks)
        self.assertFalse(fixtures.Proxy.instances)

    def test_ingress_observer_exception(self):
        rows = []
        authority = NativeAuthority(self.root / 'approvals.json', VerifiedTurn)
        authority.audit = rows.append
        self.addCleanup(authority.close)
        with mock.patch.object(authority.observer, 'attach', side_effect=ValueError('unlogged detail')):
            self.check_fallback('ingress_observation_exception', authority.hear, rows)
        self.assertFalse(authority.pending)
