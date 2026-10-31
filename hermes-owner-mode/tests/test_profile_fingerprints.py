# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Whole-file configuration pins also protect quantization metadata."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_setup import wizard
from owner_mode.engine_setup import check_profile_checkpoint
from owner_mode.profiles import _compatible_models, ProfileError
from owner_mode.setup_runtime import SetupError


class ProfileFingerprints(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model = Path(temporary.name)
        (self.model/'config.json').write_text(json.dumps({
            'model_type':'qwen3_5', 'quantization_config':{'bits':2.0}}))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        digest = lambda name: hashlib.sha256((self.model/name).read_bytes()).hexdigest()
        self.binding = {'id':'example/stock-exl3','revision':'a'*40,'model_type':'qwen3_5',
                        'config_sha256':digest('config.json'),
                        'index_sha256':digest('model.safetensors.index.json')}
        self.profile = {'model':{'id':'example/base'},'compatible_models':[self.binding]}

    def test_complete_config_pin_does_not_need_a_second_quantization_digest(self):
        _compatible_models([self.binding])
        check_profile_checkpoint(self.model,self.profile)

    def test_quantization_change_still_fails_the_complete_config_pin(self):
        (self.model/'config.json').write_text(json.dumps({
            'model_type':'qwen3_5','quantization_config':{'bits':2.5}}))
        with self.assertRaisesRegex(SetupError,'differs'):
            check_profile_checkpoint(self.model,self.profile)

    def test_optional_quantization_digest_is_checked_when_present(self):
        self.binding['quantization_config_sha256']='0'*64
        with self.assertRaisesRegex(SetupError,'differs'):
            check_profile_checkpoint(self.model,self.profile)
        self.binding['quantization_config_sha256']='malformed'
        with self.assertRaises(ProfileError):
            _compatible_models([self.binding])

    def test_complete_config_pin_remains_required(self):
        del self.binding['config_sha256']
        with self.assertRaises(ProfileError):
            _compatible_models([self.binding])


if __name__ == '__main__':
    unittest.main()
