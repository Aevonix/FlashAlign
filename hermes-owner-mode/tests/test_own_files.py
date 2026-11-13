# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Setup writes FlashAlign's own files whole; only harness configuration is patched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from support import load_cli

wizard = load_cli('setup')

BRIDGE = '.config/owner-mode/openclaw-bridge.json'
RESULT = '.config/owner-mode/setup-result.json'
CLIENT = '.config/flashalign/client.json'
UPSTREAM = 'http://127.0.0.1:8030/v1'
NOT_UTF8 = '{"a": "café"}'.encode('latin-1')
DAMAGED = [(BRIDGE, 'empty', b''), (BRIDGE, 'cut short', b'{"port": 1'), (BRIDGE, 'not UTF-8', NOT_UTF8),
           (RESULT, 'empty', b''), (RESULT, 'cut short', b'not json at all'), (RESULT, 'not UTF-8', NOT_UTF8)]


def provision(home):
    with mock.patch.object(wizard, 'free_port', side_effect=[18034, 18035]):
        return wizard.provision(home, ['openclaw'], UPSTREAM, 'bench', 'owner-test', [], False)


def detected(home):
    return wizard.discover(home, which=lambda name: None)['openclaw']['model_urls']


class OwnFiles(unittest.TestCase):
    def test_setup_replaces_an_own_file_that_is_empty_cut_short_or_not_utf8(self):
        for relative, state, data in DAMAGED:
            with self.subTest(file=relative, state=state), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                provision(home)
                written = {name: (home/name).read_bytes() for name in (BRIDGE, CLIENT)}
                self.assertEqual(detected(home), [UPSTREAM])
                (home/relative).write_bytes(data)
                # Detection reads the bridge file for its upstream; one it cannot read is left to setup.
                self.assertEqual(detected(home), [] if relative == BRIDGE else [UPSTREAM])
                result = provision(home)
                self.assertEqual(json.loads((home/RESULT).read_bytes()), json.loads(json.dumps(result)))
                self.assertEqual({name: (home/name).read_bytes() for name in written}, written)
                self.assertEqual(detected(home), [UPSTREAM])

    def test_json5_text_added_to_an_own_file_does_not_survive_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            provision(home)
            written = {name: (home/name).read_bytes() for name in (BRIDGE, CLIENT)}
            for name, data in written.items():
                (home/name).write_bytes(b'// local note\n' + data.replace(b'\n}', b',\n}'))
                self.assertEqual(wizard.load_config(home/name), json.loads(data))
            provision(home)
            self.assertEqual({name: (home/name).read_bytes() for name in written}, written)

    def test_harness_configuration_is_still_patched_beside_whole_own_files(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home/'.openclaw').mkdir()
            fragment = "custom: { /* keep */ option: 'same' },"
            (home/'.openclaw/openclaw.json').write_text('{ ' + fragment + ' }')
            provision(home)
            provision(home)
            self.assertIn(fragment, (home/'.openclaw/openclaw.json').read_text())
            for name in (BRIDGE, RESULT, CLIENT):
                json.loads((home/name).read_bytes())


if __name__ == '__main__':
    unittest.main()
