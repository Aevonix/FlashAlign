# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Setup quotes a YAML string that a YAML 1.2 reader would take for a number."""
from pathlib import Path
import tempfile
import unittest

import yaml

import test_owner_mode
from support import load_cli

wizard = load_cli('setup')

NUMBER_LIKE = ('1e3', '1E3', '-1e-3', '1e+3', '1.5e3', '.5e3', '08', '0o17', '12', '1.0', '0x1F')


class YamlNumberStrings(unittest.TestCase):
    def test_number_like_strings_are_quoted_and_numbers_stay_plain(self):
        config = {'strings': {'value-%d' % index: text for index, text in enumerate(NUMBER_LIKE)},
                  NUMBER_LIKE[0]: 'a key is quoted too',
                  'numbers': {'float': 1000.0, 'large': 1e+20, 'integer': 8},
                  'text': {'name': 'fixture-model', 'version': '1.2.3', 'word': 'e3'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            wizard.save_config(path, config)
            written = path.read_text(encoding='utf-8')
        self.assertEqual(yaml.safe_load(written), config)
        self.assertEqual(list(yaml.safe_load(written)), list(config))
        lines = written.splitlines()
        for index, text in enumerate(NUMBER_LIKE):
            self.assertIn("  value-%d: '%s'" % (index, text), lines)
        self.assertIn("'1e3': a key is quoted too", lines)
        for line in ('  float: 1000.0', '  large: 1.0e+20', '  integer: 8',
                     '  name: fixture-model', '  version: 1.2.3', '  word: e3'):
            self.assertIn(line, lines)


if __name__ == '__main__':
    unittest.main()
