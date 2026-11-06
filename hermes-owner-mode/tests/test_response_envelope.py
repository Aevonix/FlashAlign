# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Verified metadata and transport completion for strict evaluation callers."""
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


class ResponseEnvelopeTests(unittest.TestCase):
    def client(self, body, headers=None):
        return Client('http://localhost/v1', 'fixture', transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={
                'X-Owner-Mode-Effective': 'off', **(headers or {})}, text=body)))

    def test_complete_stream_has_safe_immutable_metadata(self):
        body = 'data: {"model":"fixture","choices":[]}\n\ndata: [DONE]\n\n'
        with self.client(body, {'X-FlashAlign-Generation': 'generation-2',
                               'X-FlashAlign-Profile-Digest': 'a' * 64,
                               'Authorization': 'not-for-callers'}) as client:
            with client.stream_chat_response([]) as response:
                self.assertFalse(response.complete)
                self.assertEqual(list(response), [{'model': 'fixture', 'choices': []}])
                self.assertTrue(response.complete)
                self.assertFalse(response.closed)
                self.assertEqual(response.metadata['generation'], 'generation-2')
                self.assertNotIn('Authorization', response.metadata)
                with self.assertRaises(TypeError):
                    response.metadata['generation'] = 'other'
            self.assertTrue(response.closed)

    def test_eof_or_partial_done_does_not_certify_completion(self):
        for suffix in ('', 'data: [DONE]', 'data: [DONE]\n'):
            with self.subTest(suffix=suffix), self.client('data: {"choices":[]}\n\n' + suffix) as client:
                with client.stream_chat_response([]) as response:
                    self.assertEqual(list(response), [{'choices': []}])
                    self.assertFalse(response.complete)

    def test_unconsumed_response_is_closed_but_not_complete(self):
        with self.client('data: [DONE]\n\n') as client:
            with client.stream_chat_response([]) as response:
                pass
            self.assertTrue(response.closed)
            self.assertFalse(response.complete)

    def test_multiline_sse_data_and_comments(self):
        body = ': heartbeat\n\nevent: message\ndata: {"choices":\ndata: []}\n\ndata:[DONE]\n\n'
        with self.client(body) as client:
            with client.stream_chat_response([]) as response:
                self.assertEqual(list(response), [{'choices': []}])
                self.assertTrue(response.complete)

    def test_malformed_identity_is_rejected_before_content(self):
        with self.client('data: [DONE]\n\n', {'X-FlashAlign-Profile-Digest': '../invalid'}) as client:
            with self.assertRaisesRegex(ValueError, 'identity metadata'):
                with client.stream_chat_response([]):
                    self.fail('Untrusted metadata was exposed')

    def test_chat_wrapper_keeps_json_return_and_envelope_is_available(self):
        with self.client('{"choices":[]}') as client:
            self.assertEqual(client.chat([]), {'choices': []})
            response = client.chat_response([])
            self.assertEqual(response.data, {'choices': []})
            self.assertEqual(response.metadata['effective_profile'], 'off')

    def test_noninteractive_cli_preserves_omitted_output_limit(self):
        from flashalign.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'client.json'
            config.write_text(json.dumps({'profile': 'fixture'}))
            client = mock.MagicMock()
            client.__enter__.return_value = client
            client.chat.return_value = {'choices': []}
            for arguments, expected in (([], {}), (['--max-tokens', '37'], {'max_tokens': 37})):
                with (mock.patch('flashalign.__main__.Client.from_config', return_value=client),
                      mock.patch('sys.stdin', io.StringIO('Hello')),
                      redirect_stdout(io.StringIO())):
                    main(['--config', str(config), *arguments])
                self.assertEqual(client.chat.call_args.kwargs, {'profile': None, 'scale': 1., **expected})


if __name__ == '__main__':
    unittest.main()
