# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
import tempfile
import importlib.util
import os
from pathlib import Path
import unittest

from test_setup import wizard
from owner_mode.lifecycle_setup import prepare_lifecycle, check_lifecycle, SCHEMA
from owner_mode.setup_runtime import SetupError


class LifecycleSetupTests(unittest.TestCase):
    def test_launcher_state_is_accepted_by_native_guard_under_normal_umask(self):
        source = Path(__file__).resolve().parents[2] / 'engines/vllm-deepseek-native/supervisor.py'
        spec = importlib.util.spec_from_file_location('lifecycle_permissions_supervisor', source)
        supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervisor)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for option, child in (('--state-dir', 'profiles'), ('--state', 'lifecycle')):
                with self.subTest(option=option):
                    state = root / child
                    state.mkdir(mode=0o700)
                    runtime = state / 'nested/runtime'
                    metadata = {'profile_lifecycle': {'schema': SCHEMA,
                                'source_argument': option, 'subdirectory': child}}
                    previous = os.umask(0o022)
                    try:
                        prepare_lifecycle(metadata, state, ['engine', option, str(runtime)])
                    finally:
                        os.umask(previous)
                    # Exercise the actual launcher guard without loading a model.
                    self.assertEqual(supervisor.private_directory(runtime), runtime)
                    for path in (runtime.parent, runtime, runtime / child, runtime / child / 'incoming'):
                        self.assertEqual(path.stat().st_mode & 0o777, 0o700)

    def test_nested_control_parents_are_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {'profile_lifecycle': {'schema': SCHEMA, 'subdirectory': 'nested/control'}}
            previous = os.umask(0o022)
            try:
                prepare_lifecycle(metadata, root, ['engine'])
            finally:
                os.umask(previous)
            self.assertEqual((root / 'nested').stat().st_mode & 0o777, 0o700)
            self.assertEqual((root / 'nested/control').stat().st_mode & 0o777, 0o700)

    def test_existing_nonprivate_or_linked_state_is_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            runtime.mkdir(mode=0o700)
            runtime.chmod(0o755)
            metadata = {'profile_lifecycle': {'schema': SCHEMA, 'source_argument': '--state'}}
            with self.assertRaises(SetupError):
                prepare_lifecycle(metadata, root, ['engine', '--state', str(runtime)])
            self.assertEqual(runtime.stat().st_mode & 0o777, 0o755)
            target = root / 'target'
            target.mkdir(mode=0o700)
            linked = root / 'linked'
            linked.symlink_to(target, target_is_directory=True)
            with self.assertRaises(SetupError):
                prepare_lifecycle(metadata, root, ['engine', '--state', str(linked)])
            self.assertEqual(list(target.iterdir()), [])

    def test_unseen_launcher_declaration_prepares_paths_and_keeps_engine_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            metadata = {'profile_lifecycle': {'schema': SCHEMA,
                        'state_argument': '--custom-state', 'socket_argument': '--custom-socket'}}
            argv, result = prepare_lifecycle(metadata, base, ['engine', '--', '--draft', 'on'])
            self.assertEqual(argv[-3:], ['--', '--draft', 'on'])
            self.assertEqual(argv[1:3], ['--custom-state', str(base/'control')])
            self.assertEqual(result['admin_socket'], str(base/'control/admin.sock'))
            self.assertEqual((base/'control').stat().st_mode & 0o777, 0o700)
            self.assertTrue(Path(result['admin_staging_directory']).is_dir())

    def test_existing_launcher_state_and_invalid_declarations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {'profile_lifecycle': {'schema': SCHEMA, 'source_argument': '--state', 'subdirectory': 'lifecycle'}}
            argv = ['engine', '--state', str(root/'native')]
            result, control = prepare_lifecycle(metadata, root, argv)
            self.assertEqual(result, argv)
            self.assertEqual(control['admin_socket'], str(root/'native/lifecycle/admin.sock'))
            for invalid in ('../escape', '/absolute', 'a//b'):
                metadata['profile_lifecycle']['subdirectory'] = invalid
                with self.assertRaises(SetupError): prepare_lifecycle(metadata, root, argv)
            self.assertEqual(prepare_lifecycle({}, root, argv), (argv, {}))

    def test_readiness_requires_the_advertised_committed_generation(self):
        from unittest.mock import MagicMock
        deployment = {'profile_lifecycle': {'schema': SCHEMA}, 'profile': 'owner', 'admin_socket': '/fixture'}
        client = MagicMock()
        client.__enter__.return_value.request.return_value = {
            'admission_fenced': False, 'aliases': {'owner': {'generation': 'g-current'}},
            'generations': [{'id': 'g-current', 'allocated': True}]}
        result = check_lifecycle(deployment, client_factory=lambda *a, **k: client)
        self.assertEqual(result['profile_lifecycle']['generation'], 'g-current')
        client.__enter__.return_value.request.return_value['admission_fenced'] = True
        with self.assertRaises(SetupError):
            check_lifecycle(deployment, client_factory=lambda *a, **k: client)
