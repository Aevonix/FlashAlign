# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Installed non-channel extensions use the channel entry-point loader."""
from types import SimpleNamespace
import unittest
from unittest import mock

import test_owner_mode
from owner_mode import channel_plugins, engine_adapters, model_adapters, profile_adapters


def entry(name, value):
    return SimpleNamespace(name=name, dist=SimpleNamespace(name='fixture-adapters', version='1.0'),
                           load=mock.Mock(return_value=value))


class InstalledAdapters(unittest.TestCase):
    def setUp(self):
        for module, fields in [
            (engine_adapters, {'_registry': engine_adapters.DEFAULT_ENGINE_ADAPTERS}),
            (model_adapters, {'_registry': model_adapters.DEFAULT_MODEL_ADAPTERS}),
            (profile_adapters, {'_operators': profile_adapters.DEFAULT_OPERATORS,
                                '_formats': profile_adapters.DEFAULT_ARTIFACT_FORMATS}),
        ]:
            for name, value in {**fields, '_loaded': False}.items():
                patcher = mock.patch.object(module, name, value, create=True)
                patcher.start(); self.addCleanup(patcher.stop)
        self.entries = {}
        patcher = mock.patch.object(channel_plugins.metadata, 'entry_points',
                                    side_effect=lambda *, group: self.entries.get(group, []))
        patcher.start(); self.addCleanup(patcher.stop)

    def test_engine_entry_point_loads_once_and_cannot_replace_builtin(self):
        adapter = engine_adapters.EngineAdapter('fixture', lambda: None, lambda: None)
        installed = entry('fixture-engine', adapter)
        self.entries['flashalign.engine_adapters'] = [installed]
        registry = engine_adapters.engine_adapters()
        self.assertIs(registry.get('fixture'), adapter)
        self.assertIs(engine_adapters.engine_adapters(), registry)
        installed.load.assert_called_once()
        self.assertIn('vllm', registry.ids())

    def test_model_factory_loads_through_shared_loader(self):
        adapter = model_adapters.ModelAdapter('fixture', ('fixture/model',))
        installed = entry('fixture-model', lambda: adapter)
        self.entries['flashalign.model_adapters'] = [installed]
        with mock.patch.object(channel_plugins, '_load', wraps=channel_plugins._load) as load:
            registry = model_adapters.model_adapters()
        self.assertIs(registry.get('fixture/model'), adapter)
        self.assertEqual(load.call_count, 1)
        self.assertIs(model_adapters.model_adapters(), registry)
        installed.load.assert_called_once()

    def test_profile_group_loads_operators_and_artifact_formats(self):
        operator = profile_adapters.OperatorAdapter('fixture', frozenset(), lambda _: None)
        artifact = profile_adapters.ArtifactFormatAdapter('fixture', 'fixture.format', lambda _: None)
        self.entries['flashalign.profile_adapters'] = [entry('fixture-operator', operator),
                                                       entry('fixture-format', lambda: artifact)]
        self.assertIs(profile_adapters.operator_adapters().get('fixture'), operator)
        self.assertIs(profile_adapters.artifact_formats().get('fixture.format'), artifact)
        for installed in self.entries['flashalign.profile_adapters']:
            installed.load.assert_called_once()

    def test_load_exception_propagates_without_publishing_partial_registry(self):
        valid = entry('fixture-engine', engine_adapters.EngineAdapter('fixture', lambda: None, lambda: None))
        broken = entry('broken-engine', None)
        broken.load.side_effect = RuntimeError('fixture load failure')
        self.entries['flashalign.engine_adapters'] = [valid, broken]
        with self.assertRaisesRegex(RuntimeError, 'fixture load failure'):
            engine_adapters.engine_adapters()
        self.assertIs(engine_adapters._registry, engine_adapters.DEFAULT_ENGINE_ADAPTERS)
        self.assertFalse(engine_adapters._loaded)

    def test_duplicate_and_wrong_type_extensions_are_rejected(self):
        self.entries['flashalign.model_adapters'] = [entry('fixture-model', object())]
        with self.assertRaises(ValueError):
            model_adapters.model_adapters()
        adapter = model_adapters.ModelAdapter('fixture', ('fixture/model',))
        self.entries['flashalign.model_adapters'] = [entry('first', adapter), entry('second', adapter)]
        with self.assertRaises(ValueError):
            model_adapters.model_adapters()
        self.assertIs(model_adapters._registry, model_adapters.DEFAULT_MODEL_ADAPTERS)
