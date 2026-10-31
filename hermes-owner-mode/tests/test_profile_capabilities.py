# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Declared backend capabilities must match setup's profile checks."""
import json
from pathlib import Path
import unittest

from test_setup import wizard
from owner_mode.backend_setup import KIT_ROOT, _check_profile_ranks, backend_catalog
from owner_mode.setup_runtime import SetupError


class ProfileCapabilities(unittest.TestCase):
    def test_shared_grouped_contract_copies_match(self):
        root = KIT_ROOT.parent
        canonical = (KIT_ROOT/'common/deepseek_grouped.py').read_bytes()
        self.assertEqual((root/'hermes-owner-mode/deepseek_grouped.py').read_bytes(), canonical)
        self.assertEqual((KIT_ROOT/'vllm-deepseek/flashalign_deepseek/deepseek_grouped.py').read_bytes(), canonical)

    def test_all_packaged_backends_declare_their_rank_limit(self):
        expected = {'sglang-glm53-flash': 1, 'sglang-qwen4': 1}
        for path in KIT_ROOT.glob('*/backend.json'):
            with self.subTest(backend=path.parent.name):
                self.assertEqual(json.loads(path.read_text())['maximum_rank'], expected.get(path.parent.name, 128))
        self.assertTrue(all('maximum_rank' in row for row in backend_catalog()))

    def test_standard_grouped_and_writer_group_ranks(self):
        forms = [
            {'sites': [{'rank': 2}]},
            {'schema': 'flashalign.grouped_projection.v1', 'operators': {'a': {'rank': 2}}},
            {'transform': 'residual_writer_group_projection_v1', 'rank': 2},
        ]
        for manifest in forms:
            with self.subTest(manifest=manifest):
                _check_profile_ranks({'maximum_rank': 128}, manifest)
                with self.assertRaisesRegex(SetupError, 'ranks from 1 to 1'):
                    _check_profile_ranks({'maximum_rank': 1}, manifest)

    def test_every_grouped_operator_is_checked(self):
        manifest = {'schema': 'flashalign.grouped_projection.v1',
                    'operators': {'a': {'rank': 1}, 'b': {'rank': 2}}}
        with self.assertRaisesRegex(SetupError, 'ranks from 1 to 1'):
            _check_profile_ranks({'maximum_rank': 1}, manifest)

    def test_invalid_limits_and_ranks_fail_closed(self):
        for limit in (True, 0, 129, '128'):
            with self.subTest(limit=limit), self.assertRaisesRegex(SetupError, 'declaration'):
                _check_profile_ranks({'maximum_rank': limit}, {'sites': [{'rank': 1}]})
        for rank in (True, 0, 129, None):
            with self.subTest(rank=rank), self.assertRaisesRegex(SetupError, 'ranks from'):
                _check_profile_ranks({'maximum_rank': 128}, {'sites': [{'rank': rank}]})


if __name__ == '__main__':
    unittest.main()
