# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Every manager uses the live registration until its final unload."""
import importlib
import unittest
from unittest import mock

import test_plugin_registration as fixtures
from owner_mode import plugin


class SharedRegistration(unittest.TestCase):
    setUp = fixtures.ProcessRegistrationTests.setUp
    context = fixtures.ProcessRegistrationTests.context

    def test_second_manager_runs_hooks_and_middleware(self):
        first, second = self.context(), self.context(routes={'ignored': {}})
        runtime = plugin.register(first)
        self.assertIs(plugin.register(second), runtime)
        self.assertEqual(second.config_reads, 0)
        self.assertEqual(len(fixtures.Proxy.instances), 1)
        second.hooks['pre_llm_call'](session_id='session', turn_id='turn',
                                     user_message='Hello', platform='whatsapp')
        self.assertTrue(plugin.owner_turn('session', 'turn'))
        self.assertEqual(second.middleware['tool_execution']({'value': 1}, lambda value: value), {'value': 1})
        second.hooks['post_llm_call'](session_id='session', turn_id='turn')
        self.assertFalse(plugin.owner_turn('session', 'turn'))
        first.close()
        self.assertTrue(runtime.proxy.started)
        second.close()
        self.assertEqual(runtime.proxy.closed, 1)
        first.close()
        self.assertEqual(runtime.proxy.closed, 1)

    def test_reload_then_old_unload_keeps_proxy_and_new_hooks_working(self):
        first = self.context()
        runtime = plugin.register(first)
        reloaded = importlib.reload(plugin)
        second = self.context()
        with mock.patch.object(reloaded, 'ProviderProxy', side_effect=AssertionError('duplicate proxy')):
            self.assertIs(reloaded.register(second), runtime)
        first.close()
        self.assertTrue(runtime.proxy.started)
        self.assertEqual(runtime.proxy.closed, 0)
        second.hooks['pre_llm_call'](session_id='session', turn_id='turn',
                                     user_message='Hello', platform='whatsapp')
        self.assertTrue(reloaded.owner_turn('session', 'turn'))
        self.assertEqual(runtime.proxy.prepare('head', {}), (b'fixture', {}, 'owner-test'))
        second.close()
        self.assertEqual(runtime.proxy.closed, 1)
        self.assertFalse(reloaded.owner_turn('session', 'turn'))
