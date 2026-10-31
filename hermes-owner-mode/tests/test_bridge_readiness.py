# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Authenticated startup identity and real conflicting native bridge children."""
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

import httpx
from test_setup import ROOT
from support import Engine
from owner_mode.bridge import Bridge
from owner_mode.contacts import ApprovalStore
from owner_mode.proxy import ProviderProxy


class BridgeReadiness(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.control, self.key = self.root/'control', self.root/'key'
        for path in (self.control, self.key):
            path.write_bytes(secrets.token_bytes(32))
            path.chmod(0o600)
        self.engine = Engine(self.key.read_bytes())
        self.addCleanup(self.engine.close)
        self.routes = {'head': {'upstream': self.engine.base_url, 'profile': 'owner-test'}}

    def test_identity_is_authenticated_and_does_not_create_a_capability(self):
        bridge = Bridge(ApprovalStore(self.root/'approvals'), self.control, self.routes,
                        instance_id='test-instance')
        proxy = ProviderProxy(bridge.policy, self.routes, self.key, bridge=bridge)
        proxy.start()
        self.addCleanup(proxy.close)
        url = f'http://127.0.0.1:{proxy.port}/_owner/ready'
        self.assertEqual(httpx.post(url, json={}).status_code, 403)
        self.assertEqual(httpx.post(url, json={}, headers={'Authorization': 'Bearer wrong'}).status_code, 403)
        result = httpx.post(url, json={}, headers={
            'Authorization': 'Bearer '+self.control.read_bytes().hex()})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {'pid': os.getpid(), 'instance': 'test-instance'})
        self.assertEqual(len(bridge.runs), 0)
        self.assertEqual(len(bridge.claims), 0)

    def test_real_native_children_do_not_adopt_a_foreign_listener(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node is required for the native plugin lifecycle')

        def port():
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                return listener.getsockname()[1]

        first_port, race_port = port(), port()
        configurations = []
        for index, value in enumerate((first_port, first_port, race_port, race_port)):
            config = self.root/f'bridge-{index}.json'
            config.write_text(json.dumps({'port': value, 'key_file': str(self.key),
                'control_key_file': str(self.control), 'approvals_file': str(self.root/'approvals'),
                'routes': self.routes}))
            configurations.append({'python': sys.executable, 'corePath': str(ROOT),
                'bridgeConfig': str(config), 'bridgeUrl': f'http://127.0.0.1:{value}',
                'controlKeyFile': str(self.control), 'model': {'id': 'bench'}})
        manifest = self.root/'configurations.json'
        manifest.write_text(json.dumps(configurations))
        result = subprocess.run([node, str(ROOT.parent/'openclaw-owner-mode/readiness-test.mjs'),
                                 str(manifest)], capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('BRIDGE_READINESS_IDENTITY_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
