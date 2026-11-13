# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Installed extension errors identify their entry point and distribution."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_installed_adapters as fixtures
from owner_mode import channel_plugins, engine_adapters, model_adapters, profile_adapters
from owner_mode.channels import DEFAULT_REGISTRY


class AdapterErrors(unittest.TestCase):
    setUp = fixtures.InstalledAdapters.setUp

    def check_error(self, action, entry, error_type):
        with self.assertRaises(error_type) as caught:
            action()
        self.assertIn(entry.name, str(caught.exception))
        self.assertIn(entry.dist.name, str(caught.exception))

    def test_load_errors_name_the_extension_in_every_registry(self):
        for group, action in (
            ('engine', engine_adapters.engine_adapters),
            ('model', model_adapters.model_adapters),
            ('profile', profile_adapters.operator_adapters),
        ):
            with self.subTest(group=group):
                entry = fixtures.entry('broken-extension', None)
                entry.load.side_effect = RuntimeError('load failed')
                self.entries['flashalign.' + group + '_adapters'] = [entry]
                self.check_error(action, entry, RuntimeError)
                self.assertFalse(getattr({'engine': engine_adapters, 'model': model_adapters,
                                          'profile': profile_adapters}[group], '_loaded'))

    def test_factory_error_names_the_extension(self):
        entry = fixtures.entry('broken-factory', mock.Mock(side_effect=RuntimeError('factory failed')))
        self.entries['flashalign.engine_adapters'] = [entry]
        self.check_error(engine_adapters.engine_adapters, entry, RuntimeError)

    def test_wrong_adapter_type_names_the_extension(self):
        entry = fixtures.entry('wrong-type', object())
        self.entries['flashalign.model_adapters'] = [entry]
        self.check_error(model_adapters.model_adapters, entry, ValueError)

    def test_builtin_engine_collision_names_extension_and_does_not_publish(self):
        builtin = engine_adapters.DEFAULT_ENGINE_ADAPTERS.get(engine_adapters.DEFAULT_ENGINE_ADAPTERS.ids()[0])
        replacement = engine_adapters.EngineAdapter(builtin.id, lambda: None, lambda: None)
        entry = fixtures.entry('colliding-engine', replacement)
        self.entries['flashalign.engine_adapters'] = [entry]
        for _ in range(2):
            self.check_error(engine_adapters.engine_adapters, entry, ValueError)
            self.assertIs(engine_adapters._registry, engine_adapters.DEFAULT_ENGINE_ADAPTERS)
            self.assertIs(engine_adapters._registry.get(builtin.id), builtin)
            self.assertFalse(engine_adapters._loaded)
        self.assertEqual(entry.load.call_count, 2)

    def test_builtin_model_collision_names_extension_and_does_not_publish(self):
        builtin = next(iter(model_adapters.DEFAULT_MODEL_ADAPTERS._entries.values()))
        entry = fixtures.entry('colliding-model', model_adapters.ModelAdapter(builtin.id, ('fixture/model',)))
        self.entries['flashalign.model_adapters'] = [entry]
        self.check_error(model_adapters.model_adapters, entry, ValueError)
        self.assertIs(model_adapters._registry, model_adapters.DEFAULT_MODEL_ADAPTERS)
        self.assertFalse(model_adapters._loaded)

    def test_builtin_operator_collision_names_extension_and_does_not_publish(self):
        entry = fixtures.entry('colliding-operator', profile_adapters.OperatorAdapter(
            'projection', frozenset(), lambda _: None))
        self.entries['flashalign.profile_adapters'] = [entry]
        self.check_error(profile_adapters.operator_adapters, entry, ValueError)
        self.assertIs(profile_adapters._operators, profile_adapters.DEFAULT_OPERATORS)
        self.assertFalse(profile_adapters._loaded)

    def test_builtin_format_collision_names_extension_and_does_not_publish(self):
        entry = fixtures.entry('colliding-format', profile_adapters.ArtifactFormatAdapter(
            'dense-safetensors', 'fixture.format', lambda _: None))
        self.entries['flashalign.profile_adapters'] = [entry]
        self.check_error(profile_adapters.artifact_formats, entry, ValueError)
        self.assertIs(profile_adapters._formats, profile_adapters.DEFAULT_ARTIFACT_FORMATS)
        self.assertFalse(profile_adapters._loaded)

    def test_selected_channel_load_error_names_extension(self):
        entry = fixtures.entry('broken-channel', None)
        entry.load.side_effect = RuntimeError('load failed')
        self.entries[channel_plugins.ENTRYPOINT_GROUP] = [entry]
        with tempfile.TemporaryDirectory() as directory:
            self.check_error(lambda: channel_plugins.enable_channel_adapter(
                Path(directory) / 'approvals.json', entry.name), entry, RuntimeError)

    def test_builtin_channel_collision_names_extension(self):
        entry = fixtures.entry('colliding-channel', DEFAULT_REGISTRY.get('whatsapp'))
        self.entries[channel_plugins.ENTRYPOINT_GROUP] = [entry]
        with tempfile.TemporaryDirectory() as directory:
            self.check_error(lambda: channel_plugins.enable_channel_adapter(
                Path(directory) / 'approvals.json', entry.name), entry, ValueError)
