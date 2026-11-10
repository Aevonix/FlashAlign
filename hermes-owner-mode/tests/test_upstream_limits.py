# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The shared client does not impose an upstream connection cap."""
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import test_owner_mode
from owner_mode.policy import Policy
from owner_mode.proxy import ProviderProxy


class UpstreamLimits(unittest.TestCase):
    def test_active_connections_are_unlimited_and_idle_pool_is_bounded(self):
        server = SimpleNamespace(serve_forever=lambda: None, server_close=lambda: None)
        with mock.patch('owner_mode.proxy.ThreadingHTTPServer', return_value=server):
            proxy = ProviderProxy(Policy(), {}, '')
        self.addCleanup(proxy.close)
        pool = proxy.client._transport._pool
        self.assertEqual(pool._max_connections, sys.maxsize)
        self.assertEqual(pool._max_keepalive_connections, 20)
        self.assertIs(proxy.client._transport._pool, pool)
