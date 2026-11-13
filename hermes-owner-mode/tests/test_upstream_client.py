# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Upstream transport reuse keeps request state separate and closes responses."""
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx
import test_owner_mode
from owner_mode.policy import Policy
from owner_mode.proxy import ProviderProxy, UNAVAILABLE_MESSAGE


class UpstreamClient(unittest.TestCase):
    def make_proxy(self, respond):
        clients = []
        client_type = httpx.Client
        def client(**kwargs):
            result = client_type(transport=httpx.MockTransport(respond), **kwargs)
            clients.append(result)
            return result
        def server(address, handler):
            return SimpleNamespace(RequestHandlerClass=handler, server_port=0,
                                   serve_forever=lambda: None, server_close=lambda: None)
        patcher = mock.patch('owner_mode.proxy.httpx.Client', side_effect=client)
        self.factory = patcher.start()
        self.addCleanup(patcher.stop)
        with mock.patch('owner_mode.proxy.ThreadingHTTPServer', side_effect=server):
            proxy = ProviderProxy(Policy(), {'head': {'upstream': 'http://127.0.0.1:8030/v1',
                                                     'profile': 'owner-test', 'timeout': 45}}, '')
        self.addCleanup(proxy.close)
        return proxy, clients

    def request(self, proxy, method, headers=None):
        handler = object.__new__(proxy.server.RequestHandlerClass)
        handler.path = '/head/v1/models' if method == 'GET' else '/head/v1/chat/completions'
        body = json.dumps({'model': 'bench', 'messages': []}).encode()
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.headers = {'Content-Length': str(len(body)), **(headers or {})}
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.send_error = mock.Mock()
        getattr(handler, 'do_' + method)()
        return handler

    def test_discovery_and_posts_share_client_without_sharing_cookies_or_headers(self):
        requests, responses = [], []
        def respond(request):
            requests.append(request)
            response = httpx.Response(200, headers={'Set-Cookie': 'fixture=value; Path=/'},
                                      stream=httpx.ByteStream(b'{}'))
            responses.append(response)
            return response
        proxy, clients = self.make_proxy(respond)
        self.assertEqual(len(clients), 0)
        for method, headers in [('GET', {'Authorization': 'Bearer synthetic'}), ('POST', {}), ('GET', {})]:
            handler = self.request(proxy, method, headers)
            handler.send_response.assert_called_with(200)
            handler.send_error.assert_not_called()
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(len(clients), 1)
        self.assertTrue(all(response.is_closed for response in responses))
        self.assertNotIn('authorization', requests[1].headers)
        self.assertNotIn('cookie', requests[1].headers)
        self.assertNotIn('cookie', requests[2].headers)
        self.assertEqual(requests[0].extensions['timeout']['read'], 10)
        self.assertEqual(requests[1].extensions['timeout']['read'], 45)
        self.assertFalse(clients[0].is_closed)
        proxy.close()
        self.assertTrue(clients[0].is_closed)

    def test_client_context_is_opened_on_first_use_and_closed_once(self):
        proxy, clients = self.make_proxy(lambda _: httpx.Response(200, content=b'{}'))
        with mock.patch('owner_mode.proxy.httpx.Client') as factory:
            client = factory.return_value.__enter__.return_value
            self.assertIs(proxy.client, client)
            self.assertIs(proxy.client, client)
            factory.assert_called_once()
            factory.return_value.__enter__.assert_called_once_with()
            factory.return_value.__exit__.assert_not_called()
            proxy.close()
            proxy.close()
            factory.return_value.__exit__.assert_called_once_with(None, None, None)
            with self.assertRaises(RuntimeError):
                _ = proxy.client
            factory.assert_called_once()

    def test_failed_client_cleanup_can_be_retried_without_reopening(self):
        proxy, clients = self.make_proxy(lambda _: httpx.Response(200, content=b'{}'))
        with mock.patch('owner_mode.proxy.httpx.Client') as factory:
            _ = proxy.client
            context = factory.return_value
            context.__exit__.side_effect = [RuntimeError('synthetic cleanup failure'), None]
            with self.assertRaises(RuntimeError):
                proxy.close()
            with self.assertRaises(RuntimeError):
                _ = proxy.client
            proxy.close()
            self.assertEqual(context.__exit__.call_count, 2)
            factory.assert_called_once()

    def test_owner_unavailable_retry_uses_same_client_and_closes_both_responses(self):
        requests, responses = [], []
        def respond(request):
            requests.append(request)
            raw = json.dumps({'error': {'code': 'invalid_owner_mode', 'message': UNAVAILABLE_MESSAGE}}).encode()
            response = httpx.Response(400 if len(requests) == 1 else 200,
                                      stream=httpx.ByteStream(raw if len(requests) == 1 else b'{}'))
            responses.append(response)
            return response
        proxy, clients = self.make_proxy(respond)
        proxy.prepare = mock.Mock(return_value=(b'{}', {'X-Owner-Mode-Auth': 'synthetic'}, 'owner-test'))
        with self.assertLogs('owner_mode.proxy', level='WARNING'):
            handler = self.request(proxy, 'POST')
        handler.send_response.assert_called_with(200)
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(len(requests), 2)
        self.assertNotIn('X-Owner-Mode-Auth', requests[1].headers)
        self.assertTrue(all(response.is_closed for response in responses))
        self.assertFalse(clients[0].is_closed)

    def test_error_body_read_failure_closes_response_and_leaves_client_open(self):
        class BrokenStream(httpx.SyncByteStream):
            def __iter__(self):
                raise httpx.ReadError('synthetic failure')
                yield b''
        response = httpx.Response(400, stream=BrokenStream())
        proxy, clients = self.make_proxy(lambda _: response)
        handler = self.request(proxy, 'POST')
        handler.send_error.assert_called_with(502, 'provider wrapper unavailable')
        self.assertTrue(response.is_closed)
        self.assertFalse(clients[0].is_closed)
