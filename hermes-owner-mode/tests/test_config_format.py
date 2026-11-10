# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Configuration changes retain YAML order and unrelated JSON5 source."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import json5
import yaml
from support import load_cli

wizard = load_cli('setup')
from owner_mode.setup_transaction import SetupTransaction


class ConfigFormat(unittest.TestCase):
    def test_removing_final_property_preserves_strict_json_for_client_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'client.json'
            path.write_text('{"keep": {"first": 1, "last": 2}, "remove": true}')
            config = {'keep': {'first': 1}}
            wizard.save_config(path, config)
            self.assertEqual(json.loads(path.read_text()), config)

    def test_yaml_stays_yaml_with_mapping_order_through_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = home / '.hermes/config.yaml'
            path.parent.mkdir()
            path.write_text('# local comment\nzebra: 1\nalpha:\n  second: yes\n  first: 2\n')
            config = wizard.load_config(path)
            config['alpha']['first'] = 3
            with SetupTransaction(home) as transaction:
                wizard.save_config(path, config)
                staged = transaction.stage_file(path)
                output = staged.read_text()
                self.assertTrue(output.startswith('zebra: 1\nalpha:\n'))
                self.assertEqual(list(yaml.safe_load(output)['alpha']), ['second', 'first'])
                self.assertEqual(yaml.safe_load(output), config)

    def test_json5_keeps_unowned_properties_comments_and_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'openclaw.json'
            unchanged = "custom: { /* keep */ quoted: 'x,}:y', hex: 0x10, list: [1, 2,], }, duplicate: 1, duplicate: 2,"
            source = "// heading\n{\n  " + unchanged + "\n  plugins: { // plugins\n    custom: 'keep',\n    entries: {'owner-mode': {enabled: false}},\n  },\n}\n// footer\n"
            path.write_text(source)
            config = wizard.load_config(path)
            config['plugins']['entries']['owner-mode']['enabled'] = True
            config['plugins']['entries']['owner-mode']['config'] = {'route': 'head'}
            wizard.save_config(path, config)
            output = path.read_text()
            self.assertIn(unchanged, output)
            self.assertTrue(output.startswith('// heading\n'))
            self.assertTrue(output.endswith('// footer\n'))
            self.assertIn("custom: 'keep'", output)
            self.assertIn('plugins: { // plugins', output)
            self.assertEqual(json5.loads(output), config)
            self.assertEqual(next(path.parent.glob('*.before-owner-mode-*')).read_text(), source)

    def test_json5_patches_nested_values_and_deletions_without_touching_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json5'
            source = "{keep: 'unchanged', remove: 1, values: [ /* replaced */ 1, 2 ], object: {old: 1}}"
            path.write_text(source)
            config = wizard.load_config(path)
            del config['remove']
            config['values'] = [3]
            config['object'] = {'new': 2}
            wizard.save_config(path, config)
            self.assertIn("keep: 'unchanged'", path.read_text())
            self.assertEqual(json5.loads(path.read_text()), config)
            saved = path.read_text()
            wizard.save_config(path, copy.deepcopy(config))
            self.assertEqual(path.read_text(), saved)

    def test_provision_retains_unrelated_json5_and_yaml_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            hermes = home / '.hermes/config.yaml'
            openclaw = home / '.openclaw/openclaw.json'
            hermes.parent.mkdir(); openclaw.parent.mkdir()
            hermes.write_text('zebra: 1\nalpha: 2\n')
            fragment = "custom: { /* keep */ option: 'same' },"
            openclaw.write_text('{ ' + fragment + ' session: {dmScope: "per-channel-peer"} }')
            with mock.patch.object(wizard, 'free_port', side_effect=[18034, 18035]):
                wizard.provision(home, ['hermes', 'openclaw'], 'http://127.0.0.1:8030/v1',
                                 'bench', 'owner-test', [], False)
            self.assertTrue(hermes.read_text().startswith('zebra: 1\nalpha: 2\n'))
            self.assertIn(fragment, openclaw.read_text())
            self.assertTrue(wizard.load_config(openclaw)['plugins']['entries']['owner-mode']['enabled'])
