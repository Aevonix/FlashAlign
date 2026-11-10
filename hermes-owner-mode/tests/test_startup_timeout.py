# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Startup budgets survive preparation for each managed launch mechanism."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile
from owner_mode import backend_setup, engine_setup
from owner_mode.profiles import validate_profile


class StartupTimeoutTests(unittest.TestCase):
    def test_sglang_variants_declare_the_consumed_timeout_key(self):
        for name in ('sglang-qwen4', 'sglang-glm53-flash'):
            with self.subTest(kit=name):
                metadata = json.loads((backend_setup.KIT_ROOT/name/'backend.json').read_text())
                self.assertNotIn('startup_timeout_seconds', metadata)
                self.assertEqual(metadata['startup_timeout'], 1800)

    def test_preparation_records_timeout_and_activation_uses_it(self):
        for kind, docker in (('vllm', False), ('sglang', False), ('llamacpp', False), ('vllm', True)):
            for configured, declared, expected in ((None, 1800, 1800), (2400, 1800, 2400),
                                                    (None, None, 3600 if docker else 900)):
                with self.subTest(kind=kind, docker=docker, configured=configured, declared=declared):
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        profile_path = make_profile(root/'profile')
                        profile = validate_profile(profile_path)
                        kit = root/'kit'
                        kit.mkdir()
                        (kit/'launch.py').write_text('')
                        metadata = {'launch': 'launch.py', 'default_arguments': []}
                        if declared is not None:
                            metadata['startup_timeout'] = declared
                        spec = {'kind': kind, 'model_path': str(root/'model'), 'port': 8123}
                        if configured is not None:
                            spec['startup_timeout'] = configured
                        checked = {'kind': kind, 'kit': str(kit), 'metadata': metadata,
                                   'python': {'executable': sys.executable}, 'devices': ['0'],
                                   'model': {'path': str(root/'model')}, 'arguments': [],
                                   'drafting': {'draft_algorithm': 'NGRAM', 'speculative_config': {'method': 'ngram'}}}
                        runtime = {'python': sys.executable, 'source': str(root/'source'),
                                   'binary': str(root/'server'), 'gguf_python': str(root)}
                        if docker:
                            runtime['container_runtime'] = 'docker'
                        with (mock.patch.object(backend_setup, 'check_backend', return_value=checked),
                              mock.patch.object(backend_setup, '_prepare_runtime', return_value=runtime),
                              mock.patch.object(backend_setup, '_prepare_external_cleanup', return_value={}),
                              mock.patch('owner_mode.profile_assets.materialize_profile', return_value=profile),
                              mock.patch('owner_mode.lifecycle_setup.prepare_lifecycle', side_effect=lambda metadata, state, argv: (argv, {}))):
                            deployment = backend_setup.prepare_backend(root/'home', spec, profile_path, model_id='bench')
                        self.assertEqual(deployment.get('startup_timeout'), expected)
                        saved = json.loads((Path(deployment['state'])/'deployment.json').read_text())
                        self.assertEqual(saved['startup_timeout'], expected)
                        with (mock.patch.object(engine_setup, '_require_pidfds'),
                              mock.patch.object(engine_setup.time, 'monotonic', return_value=100),
                              mock.patch.object(engine_setup, '_new_local_attempt', return_value=({'legacy': {}}, False)),
                              mock.patch.object(engine_setup, '_reuse_legacy', return_value={}) as reuse):
                            engine_setup.activate_managed_commands(saved, readiness=mock.Mock())
                        self.assertEqual(reuse.call_args.args[2], 100 + expected)
