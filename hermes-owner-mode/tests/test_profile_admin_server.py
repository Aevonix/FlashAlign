# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import asyncio
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch

import importlib.util

spec = importlib.util.spec_from_file_location("profile_admin_server", Path(__file__).resolve().parents[1] / "profile_admin_server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Conflict(Exception):
    pass


class AdminServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'admin.sock'
        self.calls = []
        self.server = module.LocalAdminServer(self.path, status=lambda: {'fenced': False},
            operation=self.operation, conflict_errors=(Conflict,))
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.close()
        self.tmp.cleanup()

    def operation(self, request):
        self.calls.append(request)
        if request.get('fail'):
            raise Conflict('SECRET /private/artifact')
        return {'accepted': request}

    async def request(self, method='GET', path='/status', body=b'', headers=()):
        reader, writer = await asyncio.open_unix_connection(self.path)
        raw = f'{method} {path} HTTP/1.1\r\nContent-Length: {len(body)}\r\n'.encode()
        for header in headers:
            raw += header + b'\r\n'
        writer.write(raw + b'\r\n' + body)
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        head, data = response.split(b'\r\n\r\n')
        return int(head.split()[1]), json.loads(data)

    async def test_owner_can_read_and_mutate(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        code, result = await self.request()
        self.assertEqual((code, result), (200, {'ok': True, 'result': {'fenced': False}}))
        code, result = await self.request('POST', '/operations', b'{"action":"recover"}')
        self.assertEqual(code, 200)
        self.assertEqual(self.calls, [{'action': 'recover'}])

    async def test_conflict_does_not_expose_paths_or_secrets(self):
        code, result = await self.request('POST', '/operations', b'{"fail":true}')
        self.assertEqual(code, 409)
        self.assertEqual(result['error']['code'], 'Conflict')
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertNotIn('/private', json.dumps(result))

    async def test_duplicate_and_nonfinite_json_rejected(self):
        for body in (b'{"action":"stage","action":"recover"}', b'{"value":NaN}'):
            self.assertEqual((await self.request('POST', '/operations', body))[0], 400)
        self.assertEqual(self.calls, [])

    async def test_generation_auth_header_has_no_admin_meaning(self):
        # The caller still needs OS owner credentials; no signing key is read.
        code, _ = await self.request('POST', '/v1/chat/completions', b'{}',
                                     [b'X-Owner-Mode-Auth: v1:ignored'])
        self.assertEqual(code, 400)
        self.assertEqual(self.calls, [])

    async def test_framing_and_size_limits(self):
        self.assertEqual((await self.request(headers=[b'Content-Length: 0']))[0], 400)
        self.assertEqual((await self.request(headers=[b'Transfer-Encoding: chunked']))[0], 400)
        self.assertEqual((await self.request('POST', '/operations', b'x' * 65537))[0], 400)

    async def test_foreign_peer_denied_before_operation(self):
        class Sock:
            def getsockopt(self, *args):
                return struct.pack('3i', 10, os.getuid() + 1, 10)
        class Writer:
            def __init__(self): self.output = b''
            def get_extra_info(self, name): return Sock()
            def write(self, output): self.output += output
            async def drain(self): pass
            def close(self): pass
            async def wait_closed(self): pass
        writer = Writer()
        await self.server._serve(asyncio.StreamReader(), writer)
        self.assertIn(b' 403 ', writer.output)
        self.assertEqual(self.calls, [])

    async def test_live_socket_is_not_replaced(self):
        other = module.LocalAdminServer(self.path, status=lambda: None, operation=lambda x: None)
        with self.assertRaises(ValueError):
            await other.start()
        self.assertEqual((await self.request())[0], 200)

    async def test_operation_runs_off_event_loop(self):
        import threading
        thread = threading.get_ident()
        self.server.operation = lambda x: {'same_thread': thread == threading.get_ident()}
        code, result = await self.request('POST', '/operations', b'{}')
        self.assertEqual(code, 200)
        self.assertFalse(result['result']['same_thread'])

    async def test_shutdown_waits_for_inflight_mutation_before_releasing_state(self):
        import threading
        started, release = threading.Event(), threading.Event()
        def mutate(_):
            started.set()
            if not release.wait(3):
                raise RuntimeError('Test did not release operation')
            return {'committed':True}
        self.server.operation = mutate
        request = asyncio.create_task(self.request('POST', '/operations', b'{}'))
        self.assertTrue(await asyncio.to_thread(started.wait, 3))
        close = asyncio.create_task(self.server.close())
        await asyncio.sleep(0)
        self.assertFalse(close.done())
        release.set()
        self.assertEqual((await request)[0], 200)
        await close


if __name__ == '__main__':
    unittest.main()
