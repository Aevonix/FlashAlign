# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Failed configured starts release their snapshots without hiding the cause."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_profile_lifecycle import ContractBackend, KEY, stage_fixture
from owner_mode.profile_lifecycle import LifecycleError, LifecycleManager, RecoveryRequired, ResourceLimit


class ConfiguredStartCleanupTests(unittest.TestCase):
    def test_repeated_failed_starts_preserve_inventory(self):
        for failure in ('name', 'prepare', 'budget'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'profile.json'
                source.write_text(json.dumps({'name': 'configured', 'model': 'unseen/model',
                                              'operator': 'projection'}))
                backend = ContractBackend()
                limits = {'host': 16, 'worker-0': 64}
                with LifecycleManager(root / 'state', backend=backend, stager=stage_fixture,
                                      admin_key=KEY, resource_limits=limits, max_generations=2) as manager:
                    permit = manager.admin(KEY)
                    prior = manager.stage(permit, source)
                    if failure == 'budget':
                        manager.activate(permit, 'prior', prior.id, expected_revision=0)
                    initial = manager.status()
                    initial_files = set(manager.artifacts.iterdir())
                for _ in range(4):
                    backend = ContractBackend()
                    with LifecycleManager(root / 'state', backend=backend, stager=stage_fixture,
                                          admin_key=KEY, resource_limits=limits, max_generations=2) as manager:
                        permit = manager.admin(KEY)
                        manager.recover(permit)
                        if failure == 'prepare':
                            backend.fail = 'prepared'
                        error_type = ResourceLimit if failure == 'budget' else LifecycleError
                        with self.assertRaises(error_type) as caught:
                            manager.activate_configured(permit, source,
                                                        name='different' if failure == 'name' else 'configured')
                        if failure == 'name':
                            self.assertEqual(str(caught.exception), 'Configured profile name changed during staging')
                        elif failure == 'prepare':
                            self.assertEqual(str(caught.exception), 'Profile preparation failed; prior selection retained')
                            self.assertIsInstance(caught.exception.__cause__, OSError)
                        else:
                            self.assertNotIn('inventory is full', str(caught.exception))
                        self.assertEqual(manager.status(), initial)
                        self.assertEqual(set(manager.artifacts.iterdir()), initial_files)

    def test_retirement_error_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'profile.json'
            source.write_text(json.dumps({'name': 'configured', 'model': 'unseen/model',
                                          'operator': 'projection'}))
            with LifecycleManager(root / 'state', backend=ContractBackend(), stager=stage_fixture,
                                  admin_key=KEY, resource_limits={'host': 16, 'worker-0': 64}) as manager:
                permit = manager.admin(KEY)
                original = LifecycleError('Original activation failure')
                with (patch.object(manager, 'activate', side_effect=original),
                      patch.object(manager, 'retire', side_effect=OSError('Retirement failure')) as retire):
                    with self.assertRaises(LifecycleError) as caught:
                        manager.activate_configured(permit, source, name='configured')
                self.assertIs(caught.exception, original)
                generation = manager.status()['generations'][0]['id']
                retire.assert_called_once_with(permit, generation)

                # A durable commit must remain recoverable if its acknowledgment fails.
                manager.retire(permit, generation)
                manager.backend.fail = 'committed'
                with self.assertRaisesRegex(RecoveryRequired, 'Committed update requires recovery'):
                    manager.activate_configured(permit, source, name='configured')
                self.assertEqual(manager.status()['pending_phase'], 'committing')
                self.assertEqual(len(manager.status()['generations']), 1)
                manager.backend.fail = None
                manager.recover(permit)
                self.assertIsNotNone(manager.status()['aliases']['configured']['generation'])
