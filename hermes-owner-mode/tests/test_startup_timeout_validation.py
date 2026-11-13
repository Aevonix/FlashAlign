# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Every managed Python engine validates its timeout before preparation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_setup import make_profile
from owner_mode import backend_setup
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError


class StartupTimeoutValidationTests(unittest.TestCase):
    def test_invalid_timeouts_match_docker_errors_for_every_python_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = validate_profile(make_profile(root / 'profile'))
            model = {'path': str(root / 'model'), 'hidden_size': 2, 'model_type': 'qwen3_5'}
            for value in (0, 59, 7201, '1800', 1800.0, True, None):
                for declared in (False, True):
                    docker_error = None
                    for kind, docker in (('vllm', True), ('vllm', False), ('sglang', False), ('llamacpp', False)):
                        with self.subTest(value=value, declared=declared, kind=kind, docker=docker):
                            metadata = {'runtime': 'docker' if docker else 'python'}
                            spec = {'kind': kind, 'model_path': model['path']}
                            if declared:
                                metadata['startup_timeout'] = value
                            else:
                                spec['startup_timeout'] = value
                            with (patch.object(backend_setup, 'inspect_model', return_value=model),
                                  patch.object(backend_setup, '_select_kit', return_value=(root / 'kit', metadata)),
                                  patch.object(backend_setup, 'select_backend_python', return_value={}),
                                  patch.object(backend_setup, 'read_json',
                                               side_effect=SetupError('Profile read before timeout validation')),
                                  self.assertRaises(SetupError) as caught):
                                backend_setup.check_backend(root / 'home', spec, profile, model_id='bench')
                            if docker:
                                docker_error = str(caught.exception)
                                self.assertIn('startup_timeout must be an integer from 60 to 7200 seconds', docker_error)
                            else:
                                self.assertEqual(str(caught.exception), docker_error)

    def test_preparation_rejects_invalid_timeout_and_records_valid_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = make_profile(root / 'profile')
            manifest = json.loads((profile / 'profile.json').read_text())
            manifest['sites'][0].update(scale=1, tensor='model.language_model.layers.0.self_attn.o_proj.weight')
            (profile / 'profile.json').write_text(json.dumps(manifest))
            model = root / 'model'
            model.mkdir()
            (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'hidden_size': 2}))
            spec = {'kind': 'vllm', 'model_path': str(model), 'devices': ['0'], 'port': 8123}
            with (patch.object(backend_setup, 'active_speculation', return_value=None),
                  patch('owner_mode.lifecycle_setup.prepare_lifecycle', side_effect=lambda metadata, state, argv: (argv, {})),
                  patch.object(backend_setup, '_prepare_runtime', return_value={'python': sys.executable}) as prepare):
                for value in (60, 7200, 1800):
                    deployment = backend_setup.prepare_backend(root / 'home', {**spec, 'startup_timeout': value},
                                                               profile, model_id='bench')
                    saved = json.loads((Path(deployment['state']) / 'deployment.json').read_text())
                    self.assertEqual(deployment['startup_timeout'], value)
                    self.assertEqual(saved['startup_timeout'], value)
                prepare.reset_mock()
                with self.assertRaisesRegex(SetupError, 'startup_timeout must be an integer from 60 to 7200 seconds'):
                    backend_setup.prepare_backend(root / 'home', {**spec, 'startup_timeout': 0},
                                                  profile, model_id='bench')
                prepare.assert_not_called()
