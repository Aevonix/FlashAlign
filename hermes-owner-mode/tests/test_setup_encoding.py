# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Configuration text remains UTF-8 when the process locale is ASCII."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRELUDE = """
import codecs
import locale
from pathlib import Path
import sys
sys.path.insert(0, 'hermes-owner-mode/tests')
from support import load_cli
wizard = load_cli('setup')
print(wizard.__file__)
assert codecs.lookup(locale.getencoding()).name == 'ascii'
path = Path(sys.argv[1])
label = 'caf' + chr(233)
"""


class SetupEncoding(unittest.TestCase):
    def run_ascii(self, path, script):
        environment = dict(os.environ, LC_ALL='C', PYTHONCOERCECLOCALE='0')
        result = subprocess.run([sys.executable, '-X', 'utf8=0', '-c', PRELUDE + script, str(path)],
                                cwd=ROOT, env=environment, capture_output=True)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode('utf-8', errors='replace'))

    def test_load_config_reads_utf8_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, source in (('.json', '{"label": "café"}'), ('.yaml', 'label: café\n')):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / ('config' + suffix)
                    path.write_bytes(source.encode('utf-8'))
                    self.run_ascii(path, "assert wizard.load_config(path) == {'label': label}\n")

    def test_save_config_writes_utf8_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ('.json', '.json5', '.yaml'):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / ('config' + suffix)
                    path.write_bytes(b'{"label": "old"}')
                    self.run_ascii(path, "wizard.save_config(path, {'label': label})\n")
                    self.assertIn('café', path.read_bytes().decode('utf-8'))

    def test_patching_reads_existing_utf8_comments_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json5'
            path.write_bytes('// café\n{label: "old"}\n'.encode('utf-8'))
            self.run_ascii(path, "wizard.save_config(path, {'label': 'new'})\n")
            self.assertIn('// café\n', path.read_bytes().decode('utf-8'))
