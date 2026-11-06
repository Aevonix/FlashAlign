# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""An installed engine extends planning without changing central dispatch."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from test_setup import wizard
from owner_mode.engine_adapters import (EngineAdapter, EngineAdapterRegistry,
                                        DEFAULT_ENGINE_ADAPTERS, engine_adapters,
                                        using_engine_adapters)
from owner_mode.engine_setup import check_engine_setup, prepare_engine
from owner_mode.backend_setup import check_backend, prepare_backend


class EngineAdaptersTests(unittest.TestCase):
    def test_new_engine_passes_public_check_and_prepare_without_core_branch(self):
        check = Mock(return_value={'kind': 'example', 'status': 'compatible'})
        prepare = Mock(return_value={'kind': 'example', 'status': 'prepared'})
        adapter = EngineAdapter('example', check, prepare, requires_local_supervisor=False,
                                probe=lambda endpoint: {'status': 'ready'})
        registry = DEFAULT_ENGINE_ADAPTERS.with_adapter(adapter)
        spec = {'kind': 'example', 'model_path': '/models/example'}
        with using_engine_adapters(registry), patch('owner_mode.engine_setup._require_pidfds',
                                                    side_effect=AssertionError('Not a local process')):
            self.assertEqual(check_engine_setup(Path('/unused'), spec, {}, model_id='fixture')['kind'], 'example')
            self.assertEqual(prepare_engine(Path('/unused'), spec, Path('/profile'), model_id='fixture')['status'], 'prepared')
            self.assertEqual(check_backend(Path('/unused'), spec, {}, model_id='fixture')['status'], 'compatible')
            self.assertEqual(prepare_backend(Path('/unused'), spec, Path('/profile'), model_id='fixture')['status'], 'prepared')
            self.assertEqual(engine_adapters().get('example').probe('unused'), {'status': 'ready'})
        self.assertEqual(check.call_count, 2)
        self.assertEqual(prepare.call_count, 2)
        self.assertNotIn('example', engine_adapters().ids())

    def test_automatic_priority_and_explicit_selection_are_declared(self):
        registry = DEFAULT_ENGINE_ADAPTERS.with_adapter(EngineAdapter(
            'example', lambda *a, **k: None, lambda *a, **k: None,
            select=lambda *a, **k: (Path('/kit'), {'status': 'fixture'}), auto_priority=50))
        self.assertEqual(registry.automatic()[0].id, 'example')
        self.assertNotIn('tensorfold', [adapter.id for adapter in registry.automatic()])

    def test_untrusted_names_cannot_register_callbacks(self):
        with self.assertRaises(ValueError):
            EngineAdapter('example', 'arbitrary.module', lambda: None)
        with self.assertRaises(ValueError):
            EngineAdapterRegistry((DEFAULT_ENGINE_ADAPTERS.get('vllm'),
                                   DEFAULT_ENGINE_ADAPTERS.get('vllm')))
        with self.assertRaisesRegex(ValueError, 'not installed'):
            engine_adapters().get('unregistered')


if __name__ == '__main__':
    unittest.main()
