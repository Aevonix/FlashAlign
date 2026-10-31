# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Interop of this extension with both supported base-gate interfaces."""
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from flashalign_vllm.core import Gate as OriginalGate


class CounterStore:
    def __init__(self):
        self.calls = 0

    def consume(self, *args, **kwargs):
        self.calls += 1
        return self.calls == 1


class LegacyGate(OriginalGate):
    """Old local-memory signature, using the unchanged protocol implementation."""
    def __init__(self, profiles, key, clock):
        super().__init__(profiles, key, clock=clock)


class ForwardingGate(LegacyGate):
    def __init__(self, profiles, key, clock, *, replay_store=None):
        super().__init__(profiles, key, clock=clock)
        self.store = replay_store

    def authorize(self, raw, body, headers):
        selection = super().authorize(raw, body, headers)
        if selection is not None and self.store is not None:
            _, stamp, nonce, _ = headers[0].split(':', 3)
            if not self.store.consume(nonce, int(stamp)+61, now=self.clock()):
                return None
        return selection


class CompatibilityTests(unittest.TestCase):
    def exercise(self, base, replay_calls=1):
        module = ModuleType('flashalign_vllm.core')
        module.Gate = base
        source = Path(__file__).parent/'flashalign_deepseek'/'authentication.py'
        spec = importlib.util.spec_from_file_location('test_gate_bridge', source)
        loaded = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'flashalign_vllm.core': module}):
            spec.loader.exec_module(loaded)
        key=b'k'*32
        body={'owner_mode':'fixture','owner_mode_scale':1.0}
        raw=json.dumps(body).encode()
        message='1000\nnonce_for_compat_1234\n'+hashlib.sha256(raw).hexdigest()
        signature='v1:1000:nonce_for_compat_1234:'+hmac.new(key,message.encode(),hashlib.sha256).hexdigest()
        store=CounterStore()
        gate=loaded.PersistentGate({'fixture':SimpleNamespace(digest='fixture-digest')},key,store,clock=lambda:1000)
        self.assertIsNotNone(gate.authorize(raw,body,[signature]))
        self.assertEqual(store.calls,1,'One HMAC-valid request must consume the persistent nonce exactly once')
        self.assertIsNone(gate.authorize(raw,body,[signature]))
        self.assertEqual(store.calls,replay_calls,'Each authorization uses only its declared replay path')

    def test_current_base_gate_with_persistent_store(self):
        self.exercise(OriginalGate, replay_calls=2)

    def test_legacy_base_gate(self):
        self.exercise(LegacyGate)

    def test_new_base_gate_with_persistent_store(self):
        self.exercise(ForwardingGate)


if __name__ == '__main__':
    unittest.main(verbosity=2)
