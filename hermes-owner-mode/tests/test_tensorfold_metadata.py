# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""TensorFold declarations and bounded arguments reach the real setup writer."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from test_engine_setup import make_bundle
from test_setup import make_profile, wizard
from owner_mode.drafting import argument
from owner_mode.engine_setup import check_engine_setup, prepare_engine
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError


class TensorFoldMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.bundle = make_bundle(self.root/'bundle')
        self.profile = make_profile(self.root/'profile')
        self.model = self.root/'model'
        self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type': 'glm_moe_dsa', 'hidden_size': 2}))
        self.spec = {'kind': 'tensorfold', 'model_path': str(self.model), 'bundle': str(self.bundle),
                     'devices': ['0'], 'python': sys.executable}

    def manifest(self, **changes):
        path = self.bundle/'manifest.json'
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value))

    def check(self):
        return check_engine_setup(self.home, self.spec, validate_profile(self.profile), model_id='bench')

    def test_declared_text_reaches_real_writer_and_preserves_auxiliary_route(self):
        self.manifest(input_modalities=['text'], arguments=['--context', '8192'])
        before = {'model': {'default': 'old-vision', 'supports_vision': True},
                  'agent': {'image_input_mode': 'auto'},
                  'auxiliary': {'vision': {'provider': 'existing-vision', 'model': 'image-model'}}}
        (self.home/'.hermes').mkdir()
        (self.home/'.hermes/config.yaml').write_text(json.dumps(before))
        checked = self.check()
        result = wizard.provision(self.home, ['hermes'], 'http://127.0.0.1:12345/v1',
            'bench', 'custom-owner', [], False,
            model_info={'input': checked['input_modalities'], 'contextWindow': checked['context_window']['tokens']})
        actual = json.loads((self.home/'.hermes/config.yaml').read_text())
        self.assertEqual(actual['model']['default'], 'bench')
        self.assertIs(actual['model']['supports_vision'], False)
        self.assertEqual(actual['auxiliary'], before['auxiliary'])
        self.assertFalse(result['hermes_images']['native_images'])
        self.assertEqual(result['hermes_images']['status'], 'preserved explicit image route')
        provider = actual['providers'][actual['model']['provider']]
        self.assertEqual(provider['models'], {'bench': {'supports_vision': False}})

    def test_context_override_is_identical_in_check_and_native_launch(self):
        self.manifest(arguments=['--backend', 'cuda', '--parallel', '2', '--context', '16384'])
        self.spec['arguments'] = ['--context=8192']
        checked = self.check()
        with mock.patch('owner_mode.engine_setup.prepare_python_headers', return_value={}):
            deployment = prepare_engine(self.home, self.spec, self.profile, model_id='bench')
        argv = deployment['commands'][0]['argv']
        self.assertEqual(checked['context_window']['tokens'], 8192)
        self.assertEqual(argument(argv, '--context'), '8192')
        self.assertEqual(argv.count('--context'), 1)
        self.assertEqual(argument(argv, '--parallel'), '2')
        self.assertEqual(argument(argv, '--tp'), '1')
        self.assertNotIn('--no-drafts', argv)
        self.assertFalse((Path(deployment['state'])/'running.json').exists())

    def test_missing_modalities_stay_unverified(self):
        self.assertNotIn('input_modalities', self.check())

    def test_explicit_image_declaration_is_passed_through(self):
        self.manifest(input_modalities=['text', 'image'])
        self.assertEqual(self.check()['input_modalities'], ['text', 'image'])

    def test_invalid_modalities_fail_before_writes(self):
        for value in (None, [], 'text', {}, ['image'], ['text', 'video'], ['text', 'text'], ['text', 1]):
            with self.subTest(value=value):
                self.manifest(input_modalities=value)
                with self.assertRaisesRegex(SetupError, 'input_modalities'):
                    self.check()
                with self.assertRaisesRegex(SetupError, 'input_modalities'):
                    prepare_engine(self.home, self.spec, self.profile, model_id='bench')
                self.assertFalse((self.home/'.config').exists())

    def test_unknown_and_controlled_arguments_fail_before_writes(self):
        for value in (None, '--context=8192', ['--context'], ['--context', '0'],
                      ['--context', '-1'], ['--context', '9'*30], ['--context', 8192],
                      ['--parallel', '1'], ['--tp', '2'], ['--no-drafts'],
                      ['--model', '/different'], ['--drafter', '/different'],
                      ['--context', '8192', '--context', '4096']):
            with self.subTest(value=value):
                self.spec['arguments'] = value
                with self.assertRaises(SetupError):
                    self.check()
                with self.assertRaises(SetupError):
                    prepare_engine(self.home, self.spec, self.profile, model_id='bench')
                self.assertFalse((self.home/'.config').exists())

    def test_custom_command_does_not_silently_ignore_arguments(self):
        self.spec.update(command=[sys.executable, '-c', 'pass'], arguments=['--context', '8192'])
        with self.assertRaisesRegex(SetupError, 'custom command'):
            self.check()
        with self.assertRaisesRegex(SetupError, 'custom command'):
            prepare_engine(self.home, self.spec, self.profile, model_id='bench')
        self.assertFalse((self.home/'.config').exists())

    def test_invalid_package_arguments_are_rejected_cleanly(self):
        for value in (None, 'bad', [1]):
            with self.subTest(value=value):
                self.manifest(arguments=value)
                with self.assertRaisesRegex(SetupError, 'argument array'):
                    self.check()
                self.assertFalse((self.home/'.config').exists())


if __name__ == '__main__':
    unittest.main()
