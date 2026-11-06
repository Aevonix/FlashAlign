# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx

import test_owner_mode
from owner_mode.policy import Policy
from owner_mode.proxy import ProviderProxy


class MetadataRoutesTests(unittest.TestCase):
    def test_completed_responses_preserve_validated_generation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            key = Path(temporary)/'key'; key.write_bytes(b'x'*32); key.chmod(0o600)
            proxy = ProviderProxy(Policy(), {'head': {'upstream': 'http://127.0.0.1:9/v1', 'profile': 'test'}}, key)
            proxy.start()
            try:
                with patch('owner_mode.proxy.httpx.Client') as factory:
                    client = factory.return_value.__enter__.return_value
                    client.send.return_value = httpx.Response(200, stream=httpx.ByteStream(b'{}'), headers={
                        'X-Owner-Mode-Effective': 'off', 'X-FlashAlign-Generation': 'g-example',
                        'X-FlashAlign-Profile-Digest': 'a'*64})
                    connection = http.client.HTTPConnection('127.0.0.1', proxy.port, timeout=2)
                    connection.request('POST', '/head/v1/chat/completions', json.dumps({'model': 'example',
                        'messages': [{'role': 'user', 'content': 'test'}]}), {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.getheader('X-FlashAlign-Generation'), 'g-example')
                    self.assertEqual(response.getheader('X-FlashAlign-Profile-Digest'), 'a'*64)
                    self.assertEqual(response.read(), b'{}')
                    connection.close()
            finally:
                proxy.close()

    def test_only_bounded_read_only_metadata_routes_are_forwarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            key = Path(temporary)/'key'; key.write_bytes(b'x'*32); key.chmod(0o600)
            proxy = ProviderProxy(Policy(), {'head': {'upstream': 'http://127.0.0.1:9/v1', 'profile': 'test'}}, key)
            proxy.start()
            try:
                with patch('owner_mode.proxy.httpx.Client') as factory:
                    client = factory.return_value.__enter__.return_value
                    client.stream.return_value.__enter__.return_value = httpx.Response(200, json={'schema': 'fixture'})
                    for suffix in ('models', 'owner-mode/profiles', 'flashalign/status', 'flashalign/capabilities'):
                        connection = http.client.HTTPConnection('127.0.0.1', proxy.port, timeout=2)
                        connection.request('GET', '/head/v1/'+suffix)
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.loads(response.read()), {'schema': 'fixture'})
                        connection.close()
                        args, kwargs = client.stream.call_args
                        self.assertEqual(args, ('GET', 'http://127.0.0.1:9/v1/'+suffix))
                        self.assertNotIn('X-Owner-Mode-Auth', kwargs['headers'])
                    before = client.stream.call_count
                    connection = http.client.HTTPConnection('127.0.0.1', proxy.port, timeout=2)
                    connection.request('GET', '/head/v1/flashalign/activate')
                    response = connection.getresponse(); response.read(); connection.close()
                    self.assertEqual(response.status, 404)
                    self.assertEqual(client.stream.call_count, before)
                    client.stream.return_value.__enter__.return_value = httpx.Response(200, content=b'x'*(1024*1024+1))
                    connection = http.client.HTTPConnection('127.0.0.1', proxy.port, timeout=2)
                    connection.request('GET', '/head/v1/flashalign/status')
                    response = connection.getresponse(); response.read(); connection.close()
                    self.assertEqual(response.status, 502)
            finally:
                proxy.close()
