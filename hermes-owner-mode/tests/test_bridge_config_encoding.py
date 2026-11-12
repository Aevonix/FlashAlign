# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The bridge reads its configuration file as UTF-8 under any locale."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SERVE = Path(__file__).resolve().parents[1] / 'serve.py'


class BridgeConfigEncoding(unittest.TestCase):
    def test_ascii_locale_reads_utf8_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The file ends before the bridge can start: the process creates the log directory the
            # file names and stops at the first setting that is missing.
            config = {'audit_log': str(root / 'logs/owner-mode.log'), 'model': 'modèle-synthétique'}
            path = root / 'openclaw-bridge.json'
            path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
            environment = {**os.environ, 'LC_ALL': 'C', 'LANG': 'C', 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'}
            result = subprocess.run([sys.executable, str(SERVE), '--config', str(path)],
                                    capture_output=True, text=True, timeout=30, env=environment)
            self.assertNotIn('UnicodeDecodeError', result.stderr)
            self.assertTrue((root / 'logs').is_dir(), result.stderr)


if __name__ == '__main__':
    unittest.main()
