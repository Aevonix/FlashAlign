# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Unsupported transports must not bypass the HTTP signing boundary."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = sorted([*ROOT.glob('*/flashalign_*/middleware.py'), *ROOT.glob('*/flashalign_*/http.py')])


def wrapper(path):
    name = '_boundary_' + path.parts[-3].replace('-', '_')
    package = ModuleType(name)
    package.__path__ = [str(path.parent)]
    sys.modules[name] = package
    core = ModuleType(name + '.core')
    for field in ('Gate', 'FieldError', 'cache_key', 'load_key', 'load_profiles'):
        setattr(core, field, ValueError if field == 'FieldError' else object())
    sys.modules[core.__name__] = core
    replay = ModuleType(name + '.flashalign_replay')
    replay.deployment_replay_store = object()
    sys.modules[replay.__name__] = replay
    auth = ModuleType(name + '.authentication')
    auth.PersistentGate = object()
    sys.modules[auth.__name__] = auth
    gate = ModuleType(name + '.gate')
    gate.AUTH, gate.EFFECTIVE, gate.FieldError = 'x-owner-mode-auth', b'x-owner-mode-effective', ValueError
    sys.modules[gate.__name__] = gate
    runtime = ModuleType(name + '.runtime')
    def forbidden_registry():
        raise AssertionError('Unsupported transports must not enter the HTTP registry')
    runtime.registry = forbidden_registry
    sys.modules[runtime.__name__] = runtime
    spec = importlib.util.spec_from_file_location(name + '.boundary', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OwnerMiddleware if hasattr(module, 'OwnerMiddleware') else module.Middleware


class ASGIBoundaries(unittest.IsolatedAsyncioTestCase):
    async def test_websockets_close_before_receiving_or_forwarding_untrusted_metadata(self):
        self.assertGreaterEqual(len(WRAPPERS), 7)
        for path in WRAPPERS:
            for route in ('/v1/realtime', '/v1/chat/completions'):
                with self.subTest(wrapper=str(path.relative_to(ROOT)), route=route):
                    sent, reached = [], []
                    async def app(scope, receive, send):
                        reached.append(scope)
                    async def receive():
                        raise AssertionError('A denied WebSocket must not read request data')
                    async def send(message):
                        sent.append(message)
                    middleware = object.__new__(wrapper(path))
                    middleware.app = app
                    await middleware({'type': 'websocket', 'path': route, 'headers': [],
                                      'query_string': b'_flashalign_profile=forged'}, receive, send)
                    self.assertEqual(reached, [])
                    self.assertEqual(sent, [{'type': 'websocket.close', 'code': 1008}])

    async def test_lifespan_is_preserved_for_server_startup(self):
        for path in WRAPPERS:
            with self.subTest(wrapper=str(path.relative_to(ROOT))):
                sent, reached = [], []
                async def app(scope, receive, send):
                    reached.append(scope['type'])
                    self.assertEqual(await receive(), {'type': 'lifespan.startup'})
                    await send({'type': 'lifespan.startup.complete'})
                async def receive():
                    return {'type': 'lifespan.startup'}
                async def send(message):
                    sent.append(message)
                middleware = object.__new__(wrapper(path))
                middleware.app = app
                await middleware({'type': 'lifespan'}, receive, send)
                self.assertEqual(reached, ['lifespan'])
                self.assertEqual(sent, [{'type': 'lifespan.startup.complete'}])
