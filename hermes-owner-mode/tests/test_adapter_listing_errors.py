# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""An installed entry point that cannot be named in full is reported with its group, name and value."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_extension_requests as requests
import test_installed_adapters as fixtures
from owner_mode import channel_plugins, engine_adapters, model_adapters, profile_adapters

VALUE = 'fixture_extension:adapter'
GROUPS = {
    'engine': (engine_adapters, engine_adapters.engine_adapters,
               lambda: engine_adapters.EngineAdapter('fixture', lambda: None, lambda: None)),
    'model': (model_adapters, model_adapters.model_adapters, requests.valid),
    'profile': (profile_adapters, profile_adapters.operator_adapters,
                lambda: profile_adapters.OperatorAdapter('fixture', frozenset(), lambda _: None)),
}


def entry(name, adapter, distribution='fixture-adapters'):
    dist = None if distribution is None else SimpleNamespace(name=distribution, version='1.0')
    return SimpleNamespace(name=name, value=VALUE, dist=dist, load=mock.Mock(return_value=adapter))


LISTED = {
    'invalid name': (lambda adapter: [entry('Bad Name', adapter)], 'Bad Name', 'Invalid adapter ID'),
    'no distribution metadata': (lambda adapter: [entry('fixture-extension', adapter, None)], 'fixture-extension',
                                 'An adapter needs installed distribution metadata'),
    'invalid distribution name': (lambda adapter: [entry('fixture-extension', adapter, 'bad name')],
                                  'fixture-extension', 'Invalid transport identity'),
    'same name twice': (lambda adapter: [entry('fixture-extension', adapter), entry('fixture-extension', adapter)],
                        'fixture-extension', 'Duplicate installed adapter entry point'),
}


class AdapterListingErrors(unittest.TestCase):
    setUp = fixtures.InstalledAdapters.setUp

    def test_listing_errors_name_group_entry_point_and_value_in_every_registry(self):
        for label, (module, registry, adapter) in GROUPS.items():
            group = 'flashalign.' + label + '_adapters'
            for error, (make, name, cause) in LISTED.items():
                with self.subTest(group=group, error=error):
                    installed = make(adapter())
                    self.entries[group] = installed
                    with self.assertRaises(ValueError) as caught:
                        registry()
                    self.assertEqual(str(caught.exception),
                                     "Adapter entry point %r (%r) in group %r: %s" % (name, VALUE, group, cause))
                    # Nothing is imported and nothing is published.
                    for item in installed:
                        item.load.assert_not_called()
                    self.assertFalse(module._loaded)
            with self.subTest(group=group, error='corrected'):
                # The corrected entry point loads on the next use.
                self.entries[group] = [entry('fixture-extension', adapter())]
                registry()
                self.assertTrue(module._loaded)

    def test_channel_listing_errors_keep_their_text(self):
        self.entries[channel_plugins.ENTRYPOINT_GROUP] = [entry('Bad Name', None)]
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError) as caught:
            channel_plugins.available_channel_adapters(Path(directory) / 'approvals.json')
        self.assertEqual(str(caught.exception), 'Invalid adapter ID')


class ListingErrorRequests(unittest.TestCase):
    setUp = requests.ExtensionRequests.setUp
    post = requests.ExtensionRequests.post

    def test_failed_request_names_the_group_and_entry_point(self):
        self.entries['flashalign.model_adapters'] = [entry('Bad Name', requests.valid())]
        with self.assertLogs('owner_mode', level='WARNING') as logs:
            response = self.post('head')
        self.assertEqual(response.status_code, 502)
        self.assertIn('provider wrapper unavailable', response.text)
        self.assertEqual([record.getMessage() for record in logs.records], [
            "OWNER_MODE_EXTENSION route=head: Adapter entry point 'Bad Name' ('fixture_extension:adapter') in group "
            "'flashalign.model_adapters' could not be loaded or registered; answered HTTP 502"])
        with self.assertNoLogs('owner_mode', level='WARNING'):
            self.assertEqual(self.post('plain').status_code, 200)


if __name__ == '__main__':
    unittest.main()
