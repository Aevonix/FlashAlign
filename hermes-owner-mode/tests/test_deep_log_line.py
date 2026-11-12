# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A log line nested too deep to parse does not hide the readable decision rows."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import test_owner_mode
from flashalign.__main__ import main


class DeepLogLine(unittest.TestCase):
    def test_deeply_nested_line_is_skipped(self):
        first = {'ts': '2026-01-01T00:00:00+0000', 'decision': 'owner_clean'}
        last = {'ts': '2026-01-01T00:00:01+0000', 'decision': 'owner_unverified'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner-mode.log'
            path.write_text(json.dumps(first) + '\n' + '[' * 200000 + '\n' + json.dumps(last) + '\n',
                            encoding='utf-8')
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['why', '--log', str(path)]), 0)
        self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()], [first, last])


if __name__ == '__main__':
    unittest.main()
