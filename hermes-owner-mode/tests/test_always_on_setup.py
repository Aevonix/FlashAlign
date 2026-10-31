# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Always On setup policy with temporary state and synthetic engine responses."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from support import load_cli
from test_setup import wizard, make_profile
from test_engine_setup import make_bundle
from owner_mode.automation import read_plan, run_plan
from owner_mode.contacts import ApprovalStore
from owner_mode.engine_setup import private_json
from owner_mode.profiles import validate_profile
from owner_mode.quickstart import make_default_plan
from owner_mode.setup_runtime import SetupError


class AlwaysOnSetup(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.profile = make_profile(self.root/'profile')
        self.key = self.root/'engine.key'
        self.key.write_bytes(os.urandom(32))
        self.key.chmod(0o600)
        self.path = self.root/'plan.json'
        self.plan = {'harnesses': ['hermes'], 'model': 'bench', 'profile': str(self.profile),
                     'base_url': 'http://127.0.0.1:1/v1', 'engine_key_file': str(self.key),
                     'tools_default': False}
        self.found = {'hermes': {'detected': True, 'runtime_compatible': True, 'model_urls': []},
                      'openclaw': {'detected': False, 'runtime_compatible': True, 'model_urls': []}}

    def test_legacy_verified_plan_still_requires_contacts(self):
        private_json(self.path, self.plan)
        with self.assertRaisesRegex(SetupError, 'missing required fields'):
            read_plan(self.path)
        private_json(self.path, {**self.plan, 'contacts': [
            {'channel': 'rcs', 'account': 'default', 'contact': '+15550000011'}]})
        self.assertEqual(read_plan(self.path)['mode'], 'verified')

    def test_always_on_plan_allows_no_contacts_and_preserves_supplied_contacts(self):
        private_json(self.path, {**self.plan, 'mode': 'always_on'})
        self.assertEqual(read_plan(self.path)['contacts'], [])
        private_json(self.path, {**self.plan, 'mode': 'always_on', 'contacts': [
            {'channel': 'rcs', 'account': 'default', 'contact': '+15550000011'}]})
        self.assertEqual(read_plan(self.path)['contacts'], [('rcs', 'default', '+15550000011')])

    def test_invalid_mode_is_rejected(self):
        for mode in ('on', None, False, [], {}):
            with self.subTest(mode=mode):
                private_json(self.path, {**self.plan, 'mode': mode})
                with self.assertRaisesRegex(SetupError, 'mode must be'):
                    read_plan(self.path)

    def test_provision_persists_mode_without_reenabling_disabled_store(self):
        store = ApprovalStore(self.home/'.config/owner-mode/approvals.json')
        store.update(lambda state: state.update(enabled=False))
        result = wizard.provision(self.home, ['hermes'], self.plan['base_url'],
                                  'bench', 'custom-owner', [], False, self.key, mode='always_on')
        self.assertEqual(store.read()['mode'], 'always_on')
        self.assertFalse(store.read()['enabled'])
        self.assertFalse(result['enabled'])
        self.assertEqual(result['mode'], 'always_on')
        self.assertEqual(store.read()['grants'], [])

    def test_default_plan_always_on_never_prompts_for_contacts_or_tools(self):
        model = self.root/'model'
        model.mkdir()
        (model/'config.json').write_text('{"model_type":"glm_moe_dsa","hidden_size":2}')
        bundle = make_bundle(self.root/'bundle')
        args = SimpleNamespace(home=self.home, model_path=model, profile=self.profile,
            engine_bundle=bundle, profiles=[], engine_python=None, devices='0',
            tools_default=None, mode='always_on')
        with mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')), redirect_stdout(io.StringIO()):
            path = make_default_plan(args, self.found, wizard.Console(color=False))
        plan = json.loads(path.read_text())
        self.assertEqual(plan['mode'], 'always_on')
        self.assertEqual(plan['contacts'], [])
        ApprovalStore(self.home/'.config/owner-mode/approvals.json').set_mode('always_on')
        args.mode = None
        with mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')), redirect_stdout(io.StringIO()):
            path = make_default_plan(args, self.found, wizard.Console(color=False))
        self.assertEqual(json.loads(path.read_text())['mode'], 'always_on')

    def test_direct_configuration_stages_always_on_without_starting_a_gateway(self):
        private_json(self.path, {**self.plan, 'harnesses': [], 'engine_only': True, 'mode': 'always_on'})
        details = validate_profile(self.profile)
        with (mock.patch('owner_mode.automation.model_rows', return_value=[{'id': 'bench'}]),
              mock.patch('owner_mode.automation.engine_profiles', return_value=[]),
              mock.patch('owner_mode.automation.resolve_profile', return_value=details),
              mock.patch('owner_mode.automation.check_engine', return_value={'status': 'active'})):
            result = run_plan(self.path, self.home, self.found, wizard.provision, apply=True)
        self.assertEqual(result['status'], 'gateway_prepared')
        self.assertEqual(result['mode'], 'always_on')
        self.assertEqual(result['gateway']['profile'], 'custom-owner')
        self.assertFalse((Path(result['gateway']['state'])/'running.json').exists())
        self.assertFalse((self.home/'.config/flashalign/client.json').exists())
        self.assertFalse((self.home/'.config/owner-mode/approvals.json').exists())

    def test_interactive_always_on_skips_contacts_and_tools(self):
        output = io.StringIO()
        details = validate_profile(self.profile)
        answers = ['hermes', self.plan['base_url'], '1', str(self.key), '2', 'yes']
        with (mock.patch.object(wizard, 'discover', return_value=self.found),
              mock.patch.object(wizard, 'model_rows', return_value=[{'id': 'bench'}]),
              mock.patch.object(wizard, 'engine_profiles', return_value=[]),
              mock.patch.object(wizard, 'select_profile', return_value=details),
              mock.patch.object(wizard, 'check_engine', return_value={'status': 'active'}),
              mock.patch('builtins.input', side_effect=answers) as prompt,
              redirect_stdout(output)):
            result = wizard.main(['--home', str(self.home), '--no-restart', '--no-color'])
        self.assertEqual(result['mode'], 'always_on')
        self.assertEqual(ApprovalStore(result['approval_store']).read()['grants'], [])
        self.assertFalse(any('contact' in call.args[0].lower() or 'Tools default' in call.args[0]
                             for call in prompt.call_args_list))
        self.assertIn('no per-request stock bypass', output.getvalue())

    def test_manage_mode_preserves_kill_switch_and_reports_mode(self):
        store = ApprovalStore(self.home/'.config/owner-mode/approvals.json')
        store.update(lambda state: state.update(enabled=False))
        manage = load_cli('manage')
        for mode in ('always_on', 'verified'):
            output = io.StringIO()
            with redirect_stdout(output):
                manage.main(['--store', str(store.path), 'mode', mode])
            result = json.loads(output.getvalue())
            self.assertEqual(result['mode'], mode)
            self.assertFalse(result['enabled'])

    def test_config_mode_must_be_part_of_reviewed_plan(self):
        with redirect_stdout(io.StringIO()), mock.patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit):
                wizard.main(['--config', str(self.path), '--mode', 'always_on', '--check'])


if __name__ == '__main__':
    unittest.main()
