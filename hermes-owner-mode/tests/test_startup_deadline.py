# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Cold model probes share a managed startup budget; ordinary probes stay short."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

from test_engine_setup import make_bundle, SERVER, fixture_deployment
from test_setup import make_profile
from owner_mode.engine_setup import activate_engine, prepare_engine, stop_engine
from owner_mode.setup_runtime import SetupError, check_engine, model_rows


DELAYED_SERVER = SERVER.replace('import hashlib,hmac,json,os,sys', 'import hashlib,hmac,json,os,sys,time')
DELAYED_SERVER = DELAYED_SERVER.replace(
    "  data=json.dumps({'data':[{'id':'bench'}]}).encode();",
    "  time.sleep(float(os.environ.get('MODEL_DELAY','0')));data=json.dumps({'data':[{'id':'bench'}]}).encode();")
DELAYED_SERVER = DELAYED_SERVER.replace(
    "  data=b'{\"choices\":[]}'",
    "  time.sleep(float(os.environ.get('OWNER_DELAY' if effective!='off' else 'STOCK_DELAY','0')))\n"
    "  with open(os.environ['PROBE_RECORDS'],'a') as out:out.write(json.dumps({'mode':effective,'stamp':int(auth[1]) if len(auth)==4 else None,'finished':time.time()})+'\\n')\n"
    "  data=b'{\"choices\":[]}'")


class StartupDeadline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / 'home'
        self.home.mkdir()
        self.profile = make_profile(self.root / 'profile')
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps({'model_type': 'glm_moe_dsa', 'hidden_size': 2}))
        self.bundle = make_bundle(self.root / 'bundle')
        self.server = self.root / 'delayed.py'
        self.server.write_text(DELAYED_SERVER)
        self.records = self.root / 'requests.jsonl'

    def prepare(self, **delays):
        environment = {'PROBE_RECORDS': str(self.records), **{key: str(value) for key, value in delays.items()}}
        result = fixture_deployment(self.home, {'model_path': str(self.model), 'bundle': str(self.bundle),
            'devices': ['0'], 'command': [sys.executable, str(self.server), '{port}'], 'environment': environment},
            self.profile, model_id='bench')
        self.addCleanup(stop_engine, result)
        return result

    def test_ordinary_existing_endpoint_keeps_fifteen_second_probes(self):
        deployment = self.prepare()
        with mock.patch('owner_mode.setup_runtime.request', return_value=(200, {'X-Owner-Mode-Effective': 'off'}, None)) as request:
            with self.assertRaises(SetupError):
                check_engine('http://127.0.0.1:1/v1', 'bench', 'custom-owner', deployment['key_file'])
        self.assertEqual([row.kwargs['timeout'] for row in request.call_args_list], [15, 15])

    def test_expired_budget_sends_no_request(self):
        deployment = self.prepare()
        with mock.patch('owner_mode.setup_runtime.request') as request:
            with self.assertRaisesRegex(SetupError, 'startup timed out'):
                check_engine(deployment['base_url'], 'bench', 'custom-owner', deployment['key_file'], deadline=time.monotonic()-1)
        request.assert_not_called()

    def test_real_delayed_model_stock_and_owner_probes_share_budget(self):
        deployment = self.prepare(MODEL_DELAY=.1, STOCK_DELAY=.2, OWNER_DELAY=.2)
        started = time.monotonic()
        result = activate_engine(deployment, timeout=3)
        self.assertEqual(result['status'], 'running')
        self.assertLess(time.monotonic()-started, 3)
        rows = [json.loads(line) for line in self.records.read_text().splitlines()]
        self.assertEqual([row['mode'] for row in rows], ['off', 'custom-owner'])
        # The owner signature is generated only after the standard probe returns.
        self.assertGreaterEqual(rows[1]['stamp'], int(rows[0]['finished']))

    def test_owner_probe_has_only_remaining_budget_and_failure_cleans_process(self):
        deployment = self.prepare(STOCK_DELAY=.25, OWNER_DELAY=2)
        started = time.monotonic()
        with self.assertRaises(SetupError):
            activate_engine(deployment, timeout=1)
        self.assertLess(time.monotonic()-started, 1.8)
        self.assertFalse((Path(deployment['state'])/'running.json').exists())
        rows = [json.loads(line) for line in self.records.read_text().splitlines()]
        self.assertEqual([row['mode'] for row in rows], ['off'])
        self.assertFalse((self.home/'.hermes/config.yaml').exists())

    def test_slow_model_listing_cannot_consume_a_fresh_probe_budget(self):
        deployment = self.prepare(MODEL_DELAY=2)
        started = time.monotonic()
        with self.assertRaisesRegex(SetupError, 'startup timed out'):
            activate_engine(deployment, timeout=.7)
        self.assertLess(time.monotonic()-started, 1.5)
        self.assertFalse(self.records.exists())
        self.assertFalse((Path(deployment['state'])/'running.json').exists())

    def test_each_probe_recomputes_remaining_time(self):
        deployment = self.prepare()
        timeouts = []
        def request(_base, _suffix, **kwargs):
            timeouts.append(kwargs['timeout'])
            time.sleep(.03)
            mode = kwargs['payload']['owner_mode']
            return 200, {'X-Owner-Mode-Effective': mode}, None
        with mock.patch('owner_mode.setup_runtime.request', side_effect=request):
            check_engine(deployment['base_url'], 'bench', 'custom-owner', deployment['key_file'], deadline=time.monotonic()+1)
        self.assertGreater(timeouts[0], timeouts[1]+.02)

    def test_model_rows_preserves_default_and_accepts_remaining_bound(self):
        with mock.patch('owner_mode.setup_runtime.request', return_value=(200, {}, {'data': []})) as request:
            model_rows('http://127.0.0.1:1/v1')
            self.assertEqual(request.call_args.kwargs['timeout'], 5)
            model_rows('http://127.0.0.1:1/v1', timeout=.2)
            self.assertEqual(request.call_args.kwargs['timeout'], .2)


if __name__ == '__main__':
    unittest.main()
