# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Canonical lifecycle operations against mock and real native workers."""
import hashlib
import hmac
import http.client
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('FLASHALIGN_GGUF_PYTHON'):
    sys.path.insert(0, os.environ['FLASHALIGN_GGUF_PYTHON'])
from lifecycle import LlamaCppBackend, NativeControl, NativeLifecycle, stage_projection
from profile_gguf import read_profile
from flashalign_lifecycle import LifecycleError, LifecycleManager, RevisionConflict
from launcher import Gate, handler
from tiny_gemma import write_tiny_gemma


def profile(root, *, name='fixture', width=32, direction=0, model_id='fixture/model'):
    root.mkdir()
    q = np.zeros((1, width), dtype='<f4'); q[0, direction] = 1
    raw = q.tobytes()
    header = json.dumps({'L0.attn_out.Q': {'dtype': 'F32', 'shape': [1, width],
                                         'data_offsets': [0, len(raw)]}}).encode()
    (root/'profile.safetensors').write_bytes(len(header).to_bytes(8, 'little')+header+raw)
    (root/'profile.json').write_text(json.dumps({'name': name, 'hidden_size': width,
        'model': {'id': model_id, 'revision': 'test-revision'}, 'sites': [{'layer': 0, 'site': 'attn_out',
        'rank': 1, 'scale': 1., 'tensor': 'model.language_model.layers.0.self_attn.o_proj.weight'}]}))
    return root


class FakeControl:
    def __init__(self):
        self.rows = {}; self.fenced = False; self.fail_commit = False
        self.lose_prepare_ack = self.lose_retire_ack = False
    def call(self, action, **fields):
        if action == 'prepare':
            self.rows.setdefault(fields['generation'], {'index': len(self.rows), 'digest': fields['digest'], 'references': 0})
            if self.lose_prepare_ack:
                self.lose_prepare_ack = False
                raise LifecycleError('prepare response lost')
            return {'index': self.rows[fields['generation']]['index']}
        if action == 'commit':
            if self.fail_commit: raise LifecycleError('unavailable')
            return {}
        if action == 'inventory': return {'generations': [{'generation': gid, **row} for gid, row in self.rows.items()]}
        if action == 'references': return {'references': self.rows[fields['generation']]['references']}
        if action == 'retire':
            self.rows.pop(fields['generation'], None)
            if self.lose_retire_ack:
                self.lose_retire_ack = False
                raise LifecycleError('retire response lost')
            return {}
        if action in ('fence', 'resume'): self.fenced = action == 'fence'; return {}
        if action == 'evict':
            for row in self.rows.values(): row['references'] = 0
            return {'evicted': True}
        raise AssertionError(action)


class LifecycleAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.control = FakeControl()
        self.backend = LlamaCppBackend(self.control, model=self.root/'model.gguf',
            profile_model={'id': 'fixture/model', 'revision': 'test-revision'}, gguf_python=self.root)
        self.manager = LifecycleManager(self.root/'state', backend=self.backend, stager=stage_projection,
                                        admin_key=b'a'*32, resource_limits={'profile_bytes': 64*1024*1024})
        self.addCleanup(self.manager.close)
        self.permit = self.manager.admin(b'a'*32)
        def convert(source, target, *_args):
            Path(target).write_bytes(b'native')
            return {'profile_digest': 'a'*64, 'adapter_sha256': hashlib.sha256(b'native').hexdigest()}
        self.converter = mock.patch('lifecycle.convert', side_effect=convert)
        self.converter.start(); self.addCleanup(self.converter.stop)

    def test_generation_swap_preserves_old_lease_and_rollback(self):
        old = self.manager.stage(self.permit, profile(self.root/'a'))
        new = self.manager.stage(self.permit, profile(self.root/'b', direction=1))
        self.manager.activate(self.permit, 'fixture', old.id, expected_revision=0)
        with self.manager.pin('fixture') as lease:
            self.manager.activate(self.permit, 'fixture', new.id, expected_revision=1)
            self.assertEqual(lease.generation.id, old.id)
            with self.assertRaises(LifecycleError): self.manager.retire(self.permit, old.id)
        self.manager.rollback(self.permit, 'fixture', old.id, expected_revision=2)
        with self.manager.pin('fixture') as lease: self.assertEqual(lease.generation.id, old.id)
        with self.assertRaises(RevisionConflict):
            self.manager.activate(self.permit, 'fixture', new.id, expected_revision=1)

    def test_model_mismatch_rejected_before_worker_control(self):
        with self.assertRaises(LifecycleError):
            self.manager.stage(self.permit, profile(self.root/'wrong', model_id='different/model'))
        self.assertFalse(self.control.rows)

    def test_lost_commit_fences_admission_until_recovery(self):
        generation = self.manager.stage(self.permit, profile(self.root/'a'))
        self.control.fail_commit = True
        with self.assertRaises(LifecycleError):
            self.manager.activate(self.permit, 'fixture', generation.id, expected_revision=0)
        self.assertTrue(self.manager.status()['admission_fenced'])
        self.control.fail_commit = False
        self.manager.recover(self.permit)
        with self.manager.pin('fixture') as lease: self.assertEqual(lease.generation.id, generation.id)

    def test_lost_prepare_ack_releases_the_actual_worker_generation(self):
        generation = self.manager.stage(self.permit, profile(self.root/'a'))
        self.control.lose_prepare_ack = True
        with self.assertRaises(LifecycleError):
            self.manager.activate(self.permit, 'fixture', generation.id, expected_revision=0)
        self.assertFalse(self.control.rows)
        self.assertFalse(self.manager.status()['admission_fenced'])

    def test_lost_retire_ack_does_not_resurrect_a_released_generation(self):
        old = self.manager.stage(self.permit, profile(self.root/'a'))
        new = self.manager.stage(self.permit, profile(self.root/'b', direction=1))
        self.manager.activate(self.permit, 'fixture', old.id, expected_revision=0)
        self.manager.activate(self.permit, 'fixture', new.id, expected_revision=1)
        self.control.lose_retire_ack = True
        with self.assertRaises(LifecycleError): self.manager.retire(self.permit, old.id)
        self.assertNotIn(old.id, self.control.rows)
        self.manager.recover(self.permit)
        self.assertNotIn(old.id, self.control.rows)
        self.assertFalse(self.manager.status()['admission_fenced'])


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, address): super().__init__('localhost', timeout=15); self.address = address
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout); self.sock.connect(str(self.address))


@unittest.skipUnless(os.environ.get('FLASHALIGN_LLAMA_TEST_BINARY'), 'Set FLASHALIGN_LLAMA_TEST_BINARY for real native CPU tests')
class NativeLifecycleTests(unittest.TestCase):
    def test_real_native_generation_swap_cache_release_and_uds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'tiny.gguf'; write_tiny_gemma(model)
            first = profile(root/'first')
            second = profile(root/'second', direction=1)
            with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            token = secrets.token_hex(32)
            with (root/'native.log').open('wb') as log:
                process = subprocess.Popen([os.environ['FLASHALIGN_LLAMA_TEST_BINARY'], '--model', str(model),
                    '--host', '127.0.0.1', '--port', str(port), '--ctx-size', '256', '--parallel', '2',
                    '--n-gpu-layers', '0', '--threads', '2', '--threads-batch', '2', '--no-webui',
                    '--spec-type', 'ngram-simple', '--cache-ram', '8', '--lora-init-without-apply'],
                    env={**os.environ, 'FLASHALIGN_INTERNAL_ADMIN_TOKEN': token}, stdout=log, stderr=log)
                runtime = gateway = None
                try:
                    deadline = time.monotonic()+20
                    while time.monotonic() < deadline:
                        if process.poll() is not None: self.fail('Native fixture failed: '+(root/'native.log').read_text()[-1500:])
                        try:
                            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=.2)
                            connection.request('GET', '/health'); response = connection.getresponse(); response.read(); connection.close()
                            if response.status == 200: break
                        except OSError: time.sleep(.05)
                    else: self.fail('Native fixture startup timed out')
                    control = NativeControl(port, token)
                    runtime = NativeLifecycle(root/'state', control=control, model=model,
                        profiles=[first], gguf_python=os.environ['FLASHALIGN_GGUF_PYTHON'])
                    with self.assertRaises(LifecycleError): NativeControl(port, '0'*64).call('inventory')
                    key = b'k'*32
                    gateway = ThreadingHTTPServer(('127.0.0.1', 0), handler(Gate(key, runtime.names), [], port, runtime))
                    thread = threading.Thread(target=gateway.serve_forever, daemon=True); thread.start()
                    self.assertEqual(self.request(gateway.server_port, key, owner=False), (200, 'off'))
                    self.assertEqual(self.request(gateway.server_port, key, owner=True), (200, 'fixture'))
                    connection = http.client.HTTPConnection('127.0.0.1', gateway.server_port)
                    connection.request('GET', '/v1/flashalign/status')
                    response = connection.getresponse(); safe = json.loads(response.read()); connection.close()
                    self.assertEqual(safe['schema'], 'flashalign.runtime-status.v1')
                    self.assertEqual(safe['active_requests'], 0)
                    self.assertFalse({'backend', 'workers', 'generations', 'artifact_root'} & set(safe))
                    old = runtime.manager.status()['aliases']['fixture']['generation']
                    with self.assertRaises(LifecycleError):
                        control.call('prepare', generation='g-'+'f'*48, digest='f'*64,
                            path=str(runtime.manager.artifacts/old/'runtime.gguf'),
                            native_digest=read_profile(first)[2], file_sha256='0'*64)
                    self.assertEqual(len(control.call('inventory')['generations']), 1)
                    def admin(payload):
                        connection = UnixConnection(runtime.state/'admin.sock')
                        connection.request('POST', '/operations', json.dumps(payload), {'Content-Type': 'application/json'})
                        response = connection.getresponse(); value = json.loads(response.read()); connection.close()
                        return response.status, value
                    status, staged = admin({'action': 'stage', 'path': str(second)})
                    self.assertEqual(status, 200, staged)
                    new = staged['result']['generation']
                    status, result = admin({'action': 'activate', 'alias': 'fixture', 'generation': new, 'expected_revision': 1})
                    self.assertEqual(status, 200, result)
                    self.assertEqual(self.request(gateway.server_port, key, owner=True), (200, 'fixture'))
                    status, _ = admin({'action': 'rollback', 'alias': 'fixture', 'generation': old, 'expected_revision': 2})
                    self.assertEqual(status, 200)
                    status, _ = admin({'action': 'activate', 'alias': 'fixture', 'generation': new, 'expected_revision': 3})
                    self.assertEqual(status, 200)
                    status, _ = admin({'action': 'retire', 'generation': old})
                    self.assertEqual(status, 409)  # Native graph/cache still retains the old generation.
                    status, result = admin({'action': 'evict_cache'})
                    self.assertEqual(status, 200, result)
                    status, result = admin({'action': 'retire', 'generation': old})
                    self.assertEqual(status, 200, result)
                    self.assertEqual(self.request(gateway.server_port, key, owner=True), (200, 'fixture'))
                    self.assertEqual(self.request(gateway.server_port, key, owner=False), (200, 'off'))
                    status, _ = admin({'action': 'activate', 'alias': 'custom', 'generation': new, 'expected_revision': 0})
                    self.assertEqual(status, 200)
                    self.assertEqual(self.request(gateway.server_port, key, owner=True, alias='custom'), (200, 'custom'))
                    status, _ = admin({'action': 'deactivate', 'alias': 'custom', 'expected_revision': 1})
                    self.assertEqual(status, 200)
                    self.assertIsNone(process.poll())
                    gateway.shutdown(); gateway.server_close(); gateway = None
                    runtime.close(); runtime = None
                    # Startup defaults must not undo the administrator's last
                    # committed selection, including an intentionally removed alias.
                    runtime = NativeLifecycle(root/'state', control=control, model=model,
                        profiles=[first], gguf_python=os.environ['FLASHALIGN_GGUF_PYTHON'])
                    self.assertEqual(runtime.manager.status()['aliases']['fixture']['generation'], new)
                    runtime.operation({'action': 'deactivate', 'alias': 'fixture', 'expected_revision': 4})
                    runtime.close(); runtime = None
                    runtime = NativeLifecycle(root/'state', control=control, model=model,
                        profiles=[first], gguf_python=os.environ['FLASHALIGN_GGUF_PYTHON'])
                    self.assertNotIn('fixture', runtime.names())
                finally:
                    if gateway is not None: gateway.shutdown(); gateway.server_close()
                    if runtime is not None: runtime.close()
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()

    def request(self, port, key, *, owner, alias='fixture'):
        raw = json.dumps({'model': 'fixture/model', 'prompt': 'A test', 'max_tokens': 3,
                          'temperature': 0, 'owner_mode': alias if owner else 'off'}).encode()
        headers = {'Content-Type': 'application/json'}
        if owner:
            stamp, nonce = str(int(time.time())), secrets.token_hex(16)
            mac = hmac.new(key, (stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()).encode(), hashlib.sha256).hexdigest()
            headers['X-Owner-Mode-Auth'] = 'v1:'+stamp+':'+nonce+':'+mac
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        connection.request('POST', '/v1/completions', raw, headers)
        response = connection.getresponse(); response.read(); connection.close()
        return response.status, response.getheader('X-Owner-Mode-Effective')


if __name__ == '__main__': unittest.main()
