# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The native OpenClaw terminal runner ends on an unusable session receipt without showing it."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

RUNNER = Path(__file__).resolve().parents[2]/'openclaw-owner-mode/terminal-runner.mjs'
MARKER = 'synthetic-marker-value'
# The launcher's receipt: the session, its paths and the parsed configuration.
RECEIPT = json.dumps({'id': 'session', 'home': '/synthetic/home', 'approvals': '/synthetic/approvals.json',
                      'root': '/synthetic/openclaw', 'config': {'custom': {'option': MARKER}, 'padding': 'x'*200}})
RECEIPTS = {
    'cut short': RECEIPT[:-100],
    'empty': '',
    'not JSON': "{id: 'session', config: {custom: {option: '" + MARKER + "'}}}",
}
FIXED = 'The native terminal got no complete session receipt from the launcher\n'


class TerminalReceipt(unittest.TestCase):
    def setUp(self):
        self.node = shutil.which('node')
        if not self.node:
            self.skipTest('Node is required for the native OpenClaw terminal')

    def run_runner(self, text):
        """Run the real runner on a pipe that carries the given receipt text."""
        read_fd, write_fd = os.pipe()
        try:
            child = subprocess.Popen([self.node, str(RUNNER), str(read_fd)], pass_fds=(read_fd,),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        finally:
            os.close(read_fd)
        with os.fdopen(write_fd, 'w') as stream:
            stream.write(text)
        output, errors = child.communicate(timeout=30)
        return child.returncode, output, errors

    def test_unusable_receipt_ends_the_runner_with_one_fixed_line(self):
        self.assertIn(MARKER, RECEIPTS['cut short'])
        for name, text in RECEIPTS.items():
            with self.subTest(receipt=name):
                code, output, errors = self.run_runner(text)
                self.assertNotIn(MARKER, output)
                self.assertNotIn(MARKER, errors)
                self.assertEqual((code, output, errors), (1, '', FIXED))


if __name__ == '__main__':
    unittest.main()
