# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A request that needs a failing installed extension gets the fixed reply and one fixed warning."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import httpx

import test_installed_adapters as fixtures
from owner_mode import model_adapters
from owner_mode.policy import Policy
from owner_mode.proxy import ProviderProxy
from support import Engine

MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
CONTENT = 'synthetic request text'
CAUSE = 'synthetic failure text'
WARNING = ("OWNER_MODE_EXTENSION route=head: Adapter entry point 'broken-model' from distribution "
           "'fixture-adapters' could not be loaded or registered; answered HTTP 502")


def valid():
    return model_adapters.ModelAdapter('fixture', ('fixture/model',))


def colliding():
    builtin = next(iter(model_adapters.DEFAULT_MODEL_ADAPTERS._entries.values()))
    return model_adapters.ModelAdapter(builtin.id, ('fixture/model',))


def failing(error):
    installed = fixtures.entry('broken-model', None)
    installed.load.side_effect = error
    return installed


FAILURES = {
    'import error': lambda: failing(ImportError(CAUSE)),
    'value error': lambda: failing(ValueError(CAUSE)),
    'permission error': lambda: failing(PermissionError(13, CAUSE, '/fixture/settings')),
    'factory error': lambda: fixtures.entry('broken-model', mock.Mock(side_effect=TypeError(CAUSE))),
    'wrong type': lambda: fixtures.entry('broken-model', object()),
    'built-in collision': lambda: fixtures.entry('broken-model', colliding()),
}


class ExtensionRequests(unittest.TestCase):
    def setUp(self):
        fixtures.InstalledAdapters.setUp(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        key = os.urandom(32)
        keyfile = Path(directory.name) / 'key'
        keyfile.write_bytes(key)
        keyfile.chmod(0o600)
        self.engine = Engine(key)
        self.addCleanup(self.engine.close)
        self.rows, self.recorded = [], threading.Event()

        def audit(row):
            self.rows.append(row)
            self.recorded.set()
        route = {'upstream': self.engine.base_url, 'profile': 'owner-test'}
        self.proxy = ProviderProxy(Policy(), {'head': dict(route, request_compatibility='deepseek_v41'),
                                              'plain': route}, keyfile, audit=audit)
        self.proxy.start()
        self.addCleanup(self.proxy.close)

    def post(self, route):
        self.recorded.clear()
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': CONTENT}]}
        with httpx.Client(timeout=5) as client:
            response = client.post(self.proxy.base_url(route) + '/chat/completions', json=body)
        # The decision row is written after the reply; wait for it before reading the rows.
        self.assertTrue(self.recorded.wait(5), 'Proxy audit callback did not complete')
        return response

    def test_failed_request_names_the_extension_in_one_fixed_warning(self):
        for label, make in FAILURES.items():
            with self.subTest(failure=label):
                self.entries['flashalign.model_adapters'] = [make()]
                del self.rows[:]
                with self.assertLogs('owner_mode', level='WARNING') as logs:
                    response = self.post('head')
                self.assertEqual(response.status_code, 502)
                self.assertIn('provider wrapper unavailable', response.text)
                self.assertEqual([record.getMessage() for record in logs.records], [WARNING])
                self.assertEqual([record.exc_info for record in logs.records], [None])
                # Neither the reply nor the decision row names the extension or repeats the failure.
                recorded = response.text + json.dumps(self.rows)
                for text in ('broken-model', 'fixture-adapters', CAUSE, CONTENT):
                    self.assertNotIn(text, recorded)
                self.assertEqual([(row['route'], row['status']) for row in self.rows], [('head', 502)])
                self.assertEqual(self.engine.records, [])

    def test_requests_that_do_not_meet_the_failure_stay_silent(self):
        self.entries['flashalign.model_adapters'] = [failing(ImportError(CAUSE))]
        with self.assertNoLogs('owner_mode', level='WARNING'):
            self.assertEqual(self.post('plain').status_code, 200)
        with self.assertLogs('owner_mode', level='WARNING'):
            self.assertEqual(self.post('head').status_code, 502)
        # A corrected extension loads on the next request, which is served without a warning.
        self.entries['flashalign.model_adapters'] = [fixtures.entry('fixture-model', valid())]
        with self.assertNoLogs('owner_mode', level='WARNING'):
            self.assertEqual(self.post('head').status_code, 200)
            self.assertEqual(self.post('plain').status_code, 200)
        self.assertEqual([row['status'] for row in self.rows], [200, 502, 200, 200])
        self.assertEqual(len(self.engine.records), 3)


if __name__ == '__main__':
    unittest.main()
