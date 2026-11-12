# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Control replies without a usable reason keep the fixed error; the control tests clean up."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2] / 'openclaw-owner-mode'


class ControlReplies(unittest.TestCase):
    def run_script(self, name):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['node', str(ROOT / name)], capture_output=True, text=True, timeout=10,
                                    env={**os.environ, 'TMPDIR': directory})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # The script removed the temporary directories it made.
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), [])

    def test_null_reply_body_has_the_fixed_error(self):
        self.run_script('control-null-reply-test.mjs')

    def test_control_error_tests_leave_no_temporary_directories(self):
        self.run_script('control-errors-test.mjs')


if __name__ == '__main__':
    unittest.main()
