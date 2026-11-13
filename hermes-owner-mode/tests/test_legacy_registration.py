# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""An older process record keeps its owner when a new module registers."""
from types import SimpleNamespace
import unittest
from unittest import mock

import test_plugin_registration as fixtures
from owner_mode import plugin


class LegacyRegistration(unittest.TestCase):
    setUp = fixtures.ProcessRegistrationTests.setUp
    context = fixtures.ProcessRegistrationTests.context

    def test_record_without_registrations_is_left_with_its_original_owner(self):
        context = self.context()
        state = plugin._process_state(create=True)
        policy = SimpleNamespace(owner_turn=mock.Mock(return_value=True))
        owner = SimpleNamespace(active=True, closed=False, closing=False, terminal=None,
                                runtime=SimpleNamespace(policy=policy),
                                authority=mock.Mock(), proxy=mock.Mock())
        fields = vars(owner).copy()
        with mock.patch.object(state, 'owner', owner):
            self.assertIsNone(plugin.register(context))
            self.assertIs(state.owner, owner)
            self.assertEqual(vars(owner), fields)
            self.assertTrue(plugin.owner_turn('session', 'turn'))
        self.assertEqual(context.config_reads, 0)
        self.assertEqual((context.hooks, context.middleware, context.unload), ({}, {}, []))
        self.assertFalse(fixtures.Proxy.instances)
        owner.proxy.close.assert_not_called()
        owner.authority.close.assert_not_called()
