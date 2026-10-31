# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import json
from pathlib import Path
import tempfile
import unittest

import test_setup
from owner_mode.backend_setup import _native_profile_layers
from owner_mode.setup_runtime import SetupError


class NativeMTPInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.kit = Path(self.temp.name)/'kit'
        self.kit.mkdir()
        self.sites = [{'layer': 2, 'site': 'attn_out', 'tensor': 'model.layers.2.out'},
                      {'layer': 3, 'site': 'attn_out', 'tensor': 'model.layers.3.out'}]
        self.metadata = {'profile_native_mtp_layers': [3],
                         'required_features': ['native_mtp_writer_groups'],
                         'profile_site_inventory': 'sites.json'}
        self.model = {'num_hidden_layers': 3, 'num_nextn_predict_layers': 1}
        self.write(self.sites)

    def write(self, rows):
        (self.kit/'sites.json').write_text(json.dumps(rows))

    def check(self, **updates):
        return _native_profile_layers(self.kit, {**self.metadata, **updates}, self.model, self.sites)

    def test_explicit_appended_mtp_layer_is_accepted(self):
        self.assertEqual(self.check(), [3])

    def test_existing_models_without_declarations_are_unchanged(self):
        self.assertEqual(_native_profile_layers(self.kit, {}, self.model, self.sites), [])

    def test_native_layer_needs_checkpoint_range_and_engine_capability(self):
        for update in ({'profile_native_mtp_layers': [2]}, {'profile_native_mtp_layers': [4]},
                       {'profile_native_mtp_layers': [True]}, {'profile_native_mtp_layers': [3, 3]},
                       {'profile_native_mtp_layers': None}, {'required_features': []}):
            with self.subTest(update=update), self.assertRaises(SetupError):
                self.check(**update)

    def test_missing_extra_or_duplicate_profile_sites_are_rejected(self):
        for rows in (self.sites[:1], self.sites + self.sites[:1],
                     self.sites + [{'layer': 0, 'site': 'attn_out', 'tensor': 'other'}]):
            with self.subTest(rows=rows), self.assertRaises(SetupError):
                _native_profile_layers(self.kit, self.metadata, self.model, rows)

    def test_bad_or_duplicate_inventory_is_rejected(self):
        for rows in ([], self.sites + self.sites[:1], [{'layer': True, 'site': 'x', 'tensor': 'y'}],
                     [{'layer': 3, 'site': '', 'tensor': 'y'}]):
            self.write(rows)
            with self.subTest(rows=rows), self.assertRaises(SetupError):
                self.check()

    def test_inventory_must_remain_in_the_engine_kit(self):
        for name in ('', '.', '..', '../sites.json', 'other/sites.json', 'other\\sites.json', '\x00'):
            with self.subTest(name=repr(name)), self.assertRaises(SetupError):
                self.check(profile_site_inventory=name)
        outside = Path(self.temp.name)/'external.json'
        outside.write_text(json.dumps(self.sites))
        (self.kit/'escape.json').symlink_to(outside)
        with self.assertRaises(SetupError):
            self.check(profile_site_inventory='escape.json')


if __name__ == '__main__':
    unittest.main()
