# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Control error diagnostics use an in-process transport."""
from pathlib import Path
import subprocess
import unittest


class ControlErrors(unittest.TestCase):
    def test_readiness_and_rejection_diagnostics(self):
        script = Path(__file__).resolve().parents[2] / 'openclaw-owner-mode/control-errors-test.mjs'
        result = subprocess.run(['node', str(script)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
