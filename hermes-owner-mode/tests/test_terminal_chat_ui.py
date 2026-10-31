# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Chat presentation reflects checked replies without altering request content."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import httpx

from test_setup import wizard
from owner_mode.client import Client, ProfileNotActive
from flashalign.terminal_cli import run


class TerminalChatUI(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.key = self.root/'engine.key'
        self.key.write_bytes(b'x' * 32)
        self.key.chmod(0o600)

    def config(self, **changes):
        data = {'base_url': 'http://localhost/v1', 'model': 'test-model',
                'profile': 'test-profile', 'key_file': str(self.key)}
        data.update(changes)
        path = self.root/'client.json'
        path.write_text(json.dumps(data))
        return path, data

    def run_chat(self, config, client, inputs, *options):
        output = io.StringIO()
        with (mock.patch('flashalign.Client.from_config', return_value=client),
              mock.patch('builtins.input', side_effect=inputs),
              redirect_stdout(output)):
            self.assertEqual(run(['chat', '--config', str(config), '--no-color',
                                  '--no-animation', *options]), 0)
        return output.getvalue()

    def test_no_reply_reports_only_requested_profile_and_never_credentials(self):
        path, data = self.config(model='model\x1b[2J-name')
        client = Client(data['base_url'], data['model'], key_file=self.key)
        text = self.run_chat(path, client, ['/exit'], '--owner')
        self.assertIn('FLASHALIGN / AEVONIX RESEARCH', text)
        self.assertIn('MODEL  model-name', text)
        self.assertIn('REQUESTED  Owner request / profile: test-profile', text)
        self.assertNotIn('Verified contacts', text)
        self.assertNotIn('(confirmed)', text)
        self.assertNotIn('\x1b', text)
        self.assertNotIn(str(self.key), text)

    def test_checked_reply_confirms_profile_and_sanitizes_only_display(self):
        path, data = self.config()
        answer = 'A\x1b[2JB\x1b]0;bad-title\x07C\x1b[31mD\x1b[0m\nnext'
        received = []

        def respond(request):
            payload = json.loads(request.content)
            received.append(payload)
            return httpx.Response(200, headers={'X-Owner-Mode-Effective': 'test-profile'},
                                  json={'choices': [{'message': {'role': 'assistant', 'content': answer}}]})

        client = Client(data['base_url'], data['model'], key_file=self.key,
                        transport=httpx.MockTransport(respond))
        prompt = 'hello\x1b[2J'
        text = self.run_chat(path, client, [prompt, 'again', '/exit'], '--owner', '--scale', '0.5')
        self.assertIn('Last response profile: test-profile (confirmed)', text)
        self.assertIn('Assistant: ABCD\nnext', text)
        self.assertNotIn('\x1b', text)
        self.assertNotIn('bad-title', text)
        self.assertEqual(received[0]['messages'][0]['content'], prompt)
        self.assertEqual(received[1]['messages'][1]['content'], answer)
        self.assertTrue(all(row['owner_mode_scale'] == 0.5 for row in received))
        self.assertTrue(all('max_tokens' not in row for row in received))

    def test_always_on_without_owner_flag_reports_checked_profile(self):
        path, data = self.config(mode='always_on')

        def respond(request):
            self.assertEqual(json.loads(request.content)['owner_mode'], 'test-profile')
            return httpx.Response(200, headers={'X-Owner-Mode-Effective': 'test-profile'},
                                  json={'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]})

        client = Client(data['base_url'], data['model'], key_file=self.key, mode='always_on',
                        profile=data['profile'], transport=httpx.MockTransport(respond))
        text = self.run_chat(path, client, ['hello', '/exit'])
        self.assertIn('REQUESTED  Always On / profile: test-profile', text)
        self.assertIn('Last response profile: test-profile (confirmed)', text)

    def test_standard_mode_does_not_claim_profile_or_strength(self):
        path, data = self.config()

        def respond(request):
            payload = json.loads(request.content)
            self.assertEqual(payload['owner_mode'], 'off')
            self.assertNotIn('owner_mode_scale', payload)
            self.assertNotIn('X-Owner-Mode-Auth', request.headers)
            return httpx.Response(200, headers={'X-Owner-Mode-Effective': 'off'},
                                  json={'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]})

        client = Client(data['base_url'], data['model'], transport=httpx.MockTransport(respond))
        text = self.run_chat(path, client, ['hello', '/exit'])
        self.assertIn('REQUESTED  Standard request / profile: off', text)
        self.assertIn('Last response profile: off (confirmed)', text)
        self.assertNotIn('STRENGTH', text)

    def test_unconfirmed_profile_never_prints_confirmation_or_answer(self):
        path, data = self.config()
        client = Client(data['base_url'], data['model'], key_file=self.key,
                        transport=httpx.MockTransport(lambda request: httpx.Response(
                            200, headers={'X-Owner-Mode-Effective': 'off'},
                            json={'choices': [{'message': {'role': 'assistant', 'content': 'should not display'}}]})))
        output = io.StringIO()
        with (mock.patch('flashalign.Client.from_config', return_value=client),
              mock.patch('builtins.input', return_value='hello'), redirect_stdout(output),
              self.assertRaises(ProfileNotActive)):
            run(['chat', '--config', str(path), '--owner', '--no-color', '--no-animation'])
        self.assertNotIn('(confirmed)', output.getvalue())
        self.assertNotIn('should not display', output.getvalue())

    def test_interrupting_native_banner_revokes_new_session(self):
        with (mock.patch('sys.stdin.isatty', return_value=True),
              mock.patch('sys.stdout.isatty', return_value=True),
              mock.patch('flashalign.terminal_cli.ApprovalStore') as approval_type,
              mock.patch('flashalign.terminal_cli.TerminalSessionStore') as session_type,
              mock.patch('flashalign.terminal_cli.TerminalUI') as ui_type,
              mock.patch('flashalign.terminal_cli.subprocess.Popen') as spawn):
            approvals = approval_type.return_value
            approvals.read.return_value = {'mode': 'verified', 'enabled': True}
            approvals.channels.return_value = []
            sessions = session_type.return_value
            sessions.create.return_value = {'id': 'temporary-session'}
            ui_type.return_value.banner.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                run(['terminal', '--harness', 'hermes', '--authorize', '--home', str(self.root),
                     '--harness-python', '/not-executed/python', '--no-animation'])
            sessions.revoke.assert_called_once_with('temporary-session')
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
