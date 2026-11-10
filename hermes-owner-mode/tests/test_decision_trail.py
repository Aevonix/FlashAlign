# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Content-free diagnostics work without a listening gateway or engine."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_owner_mode
from owner_mode import direct_gateway
from owner_mode.bridge import Bridge
from owner_mode.contacts import ApprovalStore
from owner_mode.proxy import ProviderProxy
from flashalign.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


class DecisionTrail(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = ApprovalStore(self.root / 'approvals.json')
        self.store.set_mode('verified')
        self.key = self.root / 'control.key'
        self.key.write_bytes(os.urandom(32))
        self.key.chmod(0o600)
        self.routes = {'head': {'upstream': 'http://127.0.0.1:8030/v1', 'profile': 'owner-test'}}

    def bridge(self):
        return Bridge(self.store, self.key, self.routes)

    def test_bridge_errors_expose_fixed_reason_and_record_it(self):
        bridge = self.bridge()
        rows = []
        bridge.audit = rows.append
        handler = SimpleNamespace(headers={'Authorization': 'Bearer ' + bridge.key, 'Content-Length': '2'},
                                  path='/_owner/claim', rfile=io.BytesIO(b'{}'), wfile=io.BytesIO(),
                                  send_response=mock.Mock(), send_header=mock.Mock(), end_headers=mock.Mock())
        bridge.http(handler)
        self.assertEqual(json.loads(handler.wfile.getvalue()),
                         {'error': 'control request rejected', 'reason': 'missing_native_message_claims'})
        self.assertEqual(rows, [{'decision': 'missing_native_message_claims', 'status': 400}])
        handler.headers = {}
        handler.wfile = io.BytesIO()
        bridge.http(handler)
        self.assertEqual(json.loads(handler.wfile.getvalue())['reason'], 'control_authentication_required')

    def test_bridge_ticket_records_verified_and_unverified_decisions(self):
        bridge = self.bridge()
        self.store.set_contact('rcs', 'default', '+10000000000', True)
        claim = bridge.command('begin', dict(session='session', run='turn', channel='rcs',
                               account='default', contact='+10000000000', text='Hello'))
        body = {'model': 'bench', 'messages': [{'role': 'user', 'content': 'Hello'}]}
        info = {}
        ticket = bridge.ticket(claim['capability'], body, 'head', info=info)
        self.assertTrue(ticket)
        self.assertEqual((info['decision'], info['owner_verified'], info['platform']), ('owner_clean', True, 'rcs'))
        self.store.set_contact('rcs', 'default', '+10000000000', False)
        info = {}
        self.assertFalse(bridge.ticket(claim['capability'], body, 'head', info=info))
        self.assertEqual((info['decision'], info['owner_verified']), ('owner_unverified', False))

    def run_server(self, direct):
        config = dict(approvals_file=str(self.store.path), control_key_file=str(self.key),
                      key_file=str(self.key), routes=self.routes, port=0, model='bench')
        path = self.root / 'gateway.json'
        path.write_text(json.dumps(config)); path.chmod(0o600)
        def server(address, handler):
            return SimpleNamespace(RequestHandlerClass=handler, server_port=0,
                                   serve_forever=lambda: None, server_close=lambda: None)
        def start(proxy):
            proxy.record('head', 'off', 'off', info={'decision': 'owner_unverified'})
        with mock.patch('owner_mode.proxy.ThreadingHTTPServer', side_effect=server), \
             mock.patch.object(ProviderProxy, 'start', start), \
             mock.patch('threading.Event.wait'), mock.patch('signal.signal'):
            if direct:
                direct_gateway.serve(path)
            else:
                with mock.patch.dict(sys.modules), mock.patch.object(sys, 'argv', ['serve.py', '--config', str(path)]):
                    runpy.run_path(str(ROOT / 'serve.py'), run_name='__main__')
        log = self.root / 'owner-mode.log'
        self.assertTrue(log.exists())
        self.assertEqual(json.loads(log.read_text())['decision'], 'owner_unverified')

    def test_direct_gateway_writes_audit(self):
        self.run_server(True)

    def test_bridge_server_writes_audit(self):
        self.run_server(False)

    def test_why_reads_only_recent_decision_fields(self):
        path = self.root / '.config/owner-mode/owner-mode.log'
        path.parent.mkdir(parents=True)
        rows = [dict(ts=f'2026-01-01T00:00:0{i}+00:00', decision=reason, effective='off',
                     messages=[{'content': 'private fixture'}], session='private session',
                     request={'content': 'private fixture'})
                for i, reason in enumerate(['owner_unverified', 'request_changed', 'deployment_disabled'])]
        path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n{"incomplete":')
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['why', '--home', str(self.root), '--limit', '2']), 0)
        result = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['decision'] for row in result], ['request_changed', 'deployment_disabled'])
        self.assertEqual(set(result[0]), {'ts', 'decision', 'effective'})
        self.assertNotIn('private', output.getvalue())

    def test_openclaw_control_diagnostics(self):
        result = subprocess.run(['node', str(ROOT.parent / 'openclaw-owner-mode/audit-test.mjs')],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_why_merges_python_and_javascript_timestamps(self):
        first, second = self.root / 'python.log', self.root / 'javascript.log'
        first.write_text(json.dumps({'ts': '2026-01-01T00:00:01+0000', 'decision': 'owner_clean'}) + '\n')
        second.write_text(json.dumps({'ts': '2026-01-01T00:00:00.123Z', 'decision': 'control_unavailable'}) + '\n')
        output = io.StringIO()
        with redirect_stdout(output):
            main(['why', '--log', str(first), '--log', str(second)])
        self.assertEqual([json.loads(line)['decision'] for line in output.getvalue().splitlines()],
                         ['control_unavailable', 'owner_clean'])
