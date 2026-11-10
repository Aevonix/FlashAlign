# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Unreadable audit data does not hide the readable decision rows."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import test_owner_mode
from flashalign.__main__ import main


class AuditReadErrors(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = {'ts': '2026-01-01T00:00:00+0000', 'decision': 'owner_clean'}
        self.last = {'ts': '2026-01-01T00:00:01+0000', 'decision': 'owner_unverified'}
        self.good = self.root / 'good.log'
        self.good.write_text(json.dumps(self.last) + '\n', encoding='utf-8')

    def check_rows(self, paths, expected):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(['why', *[arg for path in paths for arg in ('--log', str(path))]])
        self.assertEqual(result, 0)
        self.assertEqual([json.loads(line) for line in output.getvalue().splitlines()], expected)

    def test_invalid_utf8_line_leaves_surrounding_rows_readable(self):
        path = self.root / 'damaged.log'
        path.write_bytes((json.dumps(self.first) + '\n').encode('utf-8') + b'\xff\xfe\n' +
                         (json.dumps(self.last) + '\n').encode('utf-8'))
        self.check_rows([path], [self.first, self.last])

    def test_directory_does_not_hide_other_logs(self):
        path = self.root / 'directory.log'
        path.mkdir()
        self.check_rows([path, self.good], [self.last])

    def test_permission_error_does_not_hide_other_logs(self):
        path = self.root / 'unreadable.log'
        original = Path.open
        def open_log(current, *args, **kwargs):
            if current == path:
                raise PermissionError('unreadable fixture')
            return original(current, *args, **kwargs)
        with mock.patch.object(Path, 'open', open_log):
            self.check_rows([path, self.good], [self.last])

    def test_read_error_retains_rows_already_read(self):
        path = self.root / 'interrupted.log'
        first = self.first
        class Interrupted(io.StringIO):
            def __iter__(self):
                yield json.dumps(first) + '\n'
                raise OSError('interrupted fixture')
        original = Path.open
        def open_log(current, *args, **kwargs):
            return Interrupted() if current == path else original(current, *args, **kwargs)
        with mock.patch.object(Path, 'open', open_log):
            self.check_rows([path, self.good], [self.first, self.last])
