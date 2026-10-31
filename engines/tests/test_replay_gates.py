# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Exercise restart-safe authentication through each shipped Python gate."""
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from flashalign_replay import ReplayStore


ENGINES = Path(__file__).resolve().parents[1]
VENDORS = ['common', 'vllm/flashalign_vllm', 'vllm-gemma4/flashalign_vllm',
           'sglang/flashalign_sglang', 'sglang-qwen4/flashalign_sglang', 'llamacpp',
           'vllm-deepseek/flashalign_deepseek']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReplayGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gates = []
        for name in ['vllm', 'vllm-gemma4', 'sglang', 'sglang-qwen4']:
            file = 'core.py' if name.startswith('vllm') else 'gate.py'
            package = 'flashalign_vllm' if name.startswith('vllm') else 'flashalign_sglang'
            module = load('replay_test_' + name.replace('-', '_'), ENGINES/name/package/file)
            cls.gates.append((name, module.Gate))
        load('profile_gguf', ENGINES/'llamacpp/profile_gguf.py')
        cls.gates.append(('llamacpp', load('replay_test_llamacpp', ENGINES/'llamacpp/launcher.py').Gate))

    def test_vendored_implementations_match(self):
        bodies = [(ENGINES/path/'flashalign_replay.py').read_bytes() for path in VENDORS]
        self.assertTrue(all(body == bodies[0] for body in bodies))

    def exercise(self, name, gate_type):
        key = b't' * 32
        now = 1800000000
        body = {'owner_mode': 'test'}
        raw = json.dumps(body).encode()
        stamp, nonce = str(now + 50), 'a' * 32
        material = stamp + '\n' + nonce + '\n' + hashlib.sha256(raw).hexdigest()
        header = 'v1:' + stamp + ':' + nonce + ':' + hmac.new(key, material.encode(), hashlib.sha256).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'nonces.sqlite3'
            profiles = {'test': SimpleNamespace(digest='test-digest')}

            def make_gate():
                store = ReplayStore(path, key)
                if name == 'llamacpp':
                    return gate_type(key, ['test'], clock=lambda: now, replay_store=store)
                return gate_type(profiles, key, clock=lambda: now, replay_store=store)

            def select(gate, payload):
                method = gate.authorize if name.startswith('vllm') else gate.select
                return method(payload, body, [header])

            first = make_gate()
            invalid = select(first, raw + b' ')
            self.assertTrue(invalid is None or invalid == ('off', 1.))
            accepted = select(first, raw)
            self.assertTrue(accepted is not None and accepted != ('off', 1.))
            restarted = make_gate()
            refused = select(restarted, raw)
            self.assertTrue(refused is None or refused == ('off', 1.))
            # Loss of the existing store cannot silently initialize a fresh one.
            path.unlink()
            unavailable = select(restarted, raw)
            self.assertTrue(unavailable is None or unavailable == ('off', 1.))

    def test_restarts_and_storage_failure_keep_all_gates_closed(self):
        for name, gate in self.gates:
            with self.subTest(engine=name):
                self.exercise(name, gate)


if __name__ == '__main__':
    unittest.main()
