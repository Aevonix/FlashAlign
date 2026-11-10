# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Setup leaves channel selection to the live approval store."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from support import load_cli

wizard = load_cli('setup')
from owner_mode.contacts import ApprovalStore
from owner_mode.policy import Policy, VerifiedTurn


class LiveChannels(unittest.TestCase):
    def test_setup_does_not_freeze_channels_and_saved_selection_is_live(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with mock.patch.object(wizard, 'free_port', side_effect=[18034, 18035]):
                wizard.provision(home, ['hermes'], 'http://127.0.0.1:8030/v1',
                                 'bench', 'owner-test', [], False)
            settings = wizard.load_config(home / '.hermes/config.yaml')['plugins']['entries']['owner-mode']['settings']
            self.assertNotIn('allowed_channels', settings)
            store = ApprovalStore(settings['approvals_file'])
            store.set_channels(['whatsapp'])
            authority = SimpleNamespace(verify_turn=lambda **kw: VerifiedTurn(
                kw['session_id'], kw['turn_id'], 'telegram', kw['user_message']))
            policy = Policy(authority=authority, allowed_channels=list(store.registry.ids()),
                            channel_policy=store.channel_enabled)
            context = dict(session_id='session', turn_id='turn', user_message='Hello', platform='telegram')
            policy.begin(**context)
            self.assertFalse(policy.owner_turn('session', 'turn'))
            store.set_channels(['whatsapp', 'telegram'])
            policy.begin(**context)
            self.assertTrue(policy.owner_turn('session', 'turn'))
            store.set_channels(['whatsapp'])
            self.assertFalse(policy.owner_turn('session', 'turn'))
