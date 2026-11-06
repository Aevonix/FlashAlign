# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real HTTP/1.1 pool expiry; each chat call sends exactly one POST."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[1]
if 'owner_mode' not in sys.modules:
    spec = importlib.util.spec_from_file_location('owner_mode', ROOT/'__init__.py', submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['owner_mode'] = module
    spec.loader.exec_module(module)
from owner_mode.client import Client


class KeepaliveTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.client_address, body))
                encoded = b'{"model":"fixture","choices":[]}'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.send_header('X-Owner-Mode-Effective', 'off')
                self.end_headers()
                self.wfile.write(encoded)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    @staticmethod
    def request(client):
        return client.chat([{'role': 'user', 'content': 'Synthetic connection test.'}], max_tokens=1)

    def test_default_reuses_recent_connection_then_expires_before_five_seconds(self):
        with Client(self.url, 'fixture') as client:
            self.request(client)
            self.request(client)
            self.assertEqual(self.calls[0][0], self.calls[1][0])
            time.sleep(1.1)
            self.request(client)
        self.assertEqual(len(self.calls), 3)
        self.assertNotEqual(self.calls[1][0], self.calls[2][0])

    def test_override_retains_connection_and_supplied_transport(self):
        with Client(self.url, 'fixture', keepalive_expiry=10) as client:
            self.request(client)
            time.sleep(1.1)
            self.request(client)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0][0], self.calls[1][0])
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'choices': []}, headers={'X-Owner-Mode-Effective': 'off'})
        with Client('http://fixture.test/v1', 'fixture', keepalive_expiry=0,
                    transport=httpx.MockTransport(handler)) as client:
            self.request(client)
        self.assertEqual(len(requests), 1)

    def test_invalid_expiries_rejected_without_a_request(self):
        for value in (-1, True, '1', None, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Client(self.url, 'fixture', keepalive_expiry=value)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
