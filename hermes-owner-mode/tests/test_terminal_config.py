# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""The native OpenClaw terminal starts on the configuration text that setup keeps."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from support import load_cli

wizard = load_cli('setup')
from flashalign import terminal_cli

RUNNER = Path(__file__).resolve().parents[2]/'openclaw-owner-mode/terminal-runner.mjs'
FORMS = {
    'comment': '{\n  // local note\n  "custom": {"option": "same"}\n}\n',
    'trailing comma': '{"custom": {"option": "same",}}\n',
    'single quotes': "{'custom': {'option': 'same'}}\n",
    'unquoted key': '{custom: {"option": "same"}}\n',
    'byte order mark': '﻿{"custom": {"option": "same"}}\n',
}
# Marks both streams as terminals, then runs the given runner on the receipt descriptor.
SHIM = '''import { pathToFileURL } from 'node:url';
for (const stream of [process.stdin, process.stdout]) Object.defineProperty(stream, 'isTTY', {value: true});
const [runner] = process.argv.splice(2, 1);
await import(pathToFileURL(runner).href);
'''
# A stand-in runtime: it records the private configuration it is given and opens nothing.
TUI = '''import { readFileSync, statSync, writeFileSync } from 'node:fs';
export async function runTui() {}
export async function withEmbeddedTuiStateLock() {
  const path = process.env.OPENCLAW_CONFIG_PATH;
  const config = JSON.parse(readFileSync(path, 'utf8'));
  writeFileSync(new URL('../seen.json', import.meta.url), JSON.stringify({mode: statSync(path).mode & 0o777,
    custom: config.custom, bridge: config.plugins.entries['owner-mode'].config.bridgeConfig}));
}
'''


class Interactive(io.StringIO):
    def isatty(self):
        return True


class TerminalConfiguration(unittest.TestCase):
    def setUp(self):
        self.node = shutil.which('node')
        if not self.node:
            self.skipTest('Node is required for the native OpenClaw terminal')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()/'openclaw'
        (self.root/'dist').mkdir(parents=True)
        (self.root/'dist/tui-test.mjs').write_text(TUI)
        (self.root/'dist/embedded-backend-test.mjs').write_text('export class EmbeddedTuiBackend {}\n')
        self.shim = self.root.parent/'interactive.mjs'
        self.shim.write_text(SHIM)
        self.homes = 0

    def provision(self, text):
        """Run setup in a new home whose OpenClaw configuration holds the given text."""
        self.homes += 1
        self.home = self.root.parent/f'home-{self.homes}'
        self.config = self.home/'.openclaw/openclaw.json'
        self.config.parent.mkdir(parents=True)
        self.config.write_text(text, encoding='utf-8')
        with mock.patch.object(wizard, 'free_port', side_effect=[18034, 18035]):
            wizard.provision(self.home, ['openclaw'], 'http://127.0.0.1:8030/v1', 'bench', 'owner-test', [], False)
        return self.config.read_bytes()

    def start(self, child=None):
        """Run the real launcher; its child is the real runner with terminal streams, or the given command."""
        popen = subprocess.Popen
        def spawn(command, **options):
            command = [*child, command[-1]] if child else [command[0], str(self.shim), *command[1:]]
            return popen(command, stderr=errors, **options)
        with (tempfile.TemporaryFile('w+') as errors, mock.patch.object(subprocess, 'Popen', spawn),
              mock.patch('sys.stdin', Interactive()), redirect_stdout(Interactive())):
            code = terminal_cli.run(['terminal', '--harness', 'openclaw', '--authorize', '--home', str(self.home),
                                     '--openclaw-root', str(self.root), '--no-color', '--no-animation'])
            errors.seek(0)
            return code, errors.read()

    def seen(self):
        return json.loads((self.root/'seen.json').read_text())

    def test_terminal_starts_after_setup_keeps_json5_text_or_a_byte_order_mark(self):
        for name, text in FORMS.items():
            with self.subTest(form=name):
                kept = self.provision(text)
                with self.assertRaises(ValueError):
                    json.loads(kept.decode('utf-8'))
                code, errors = self.start()
                self.assertEqual(code, 0, errors)
                seen = self.seen()
                self.assertEqual(seen['custom'], {'option': 'same'})
                self.assertEqual(seen['mode'], 0o600)
                private = Path(seen['bridge']).parent
                self.assertEqual(private.parent, self.config.parent)
                self.assertTrue(private.name.startswith('.flashalign-terminal-'))
                self.assertFalse(private.exists())
                self.assertEqual(self.config.read_bytes(), kept)

    def test_configuration_larger_than_a_pipe_buffer_reaches_the_terminal(self):
        self.provision('{\n  // local note\n  "custom": {"option": "same"}, "padding": "' + 'x'*70000 + '"\n}\n')
        code, errors = self.start()
        self.assertEqual(code, 0, errors)
        self.assertEqual(self.seen()['custom'], {'option': 'same'})

    def test_launcher_returns_the_status_of_a_runner_that_ends_before_reading(self):
        self.provision('{"custom": {"option": "same"}, "padding": "' + 'x'*70000 + '"}\n')
        code, _ = self.start([self.node, '-e', 'process.exit(3)'])
        self.assertEqual(code, 3)

    def test_unreadable_configuration_stops_the_launcher_with_fixed_text(self):
        self.provision(FORMS['comment'])
        self.config.write_text('{"custom": "synthetic-value" "next": 1}\n')
        with self.assertRaises(ValueError) as caught:
            self.start()
        self.assertEqual(str(caught.exception), 'OpenClaw configuration is not a JSON or JSON5 object the native '
                         'terminal can read: ' + str(self.config))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_runner_without_a_parsed_configuration_names_the_file_and_the_cause(self):
        def run():
            read_fd, write_fd = os.pipe()
            try:
                child = subprocess.Popen([self.node, str(self.shim), str(RUNNER), str(read_fd)], pass_fds=(read_fd,),
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            finally:
                os.close(read_fd)
            with os.fdopen(write_fd, 'w') as stream:
                json.dump({'id': 'session', 'home': str(self.home), 'root': str(self.root)}, stream)
            output, errors = child.communicate(timeout=30)
            return child.returncode, output, errors
        self.provision('{"custom": {"option": "same"}}\n')
        self.assertEqual(run()[0], 0)
        self.assertEqual(self.seen()['custom'], {'option': 'same'})
        self.provision(FORMS['unquoted key'])
        self.assertEqual(run(), (1, '', f'The native terminal cannot read {self.config}: the file is not strict JSON '
                                        'and the launcher supplied no parsed configuration\n'))


if __name__ == '__main__':
    unittest.main()
