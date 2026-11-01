# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Declared backend capabilities must match setup's profile checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_setup import wizard
from owner_mode.backend_setup import KIT_ROOT, _check_profile_ranks, backend_catalog
from owner_mode.setup_runtime import SetupError


class ProfileCapabilities(unittest.TestCase):
    def test_provisioned_packages_import_without_the_source_tree(self):
        script = '''
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

installed = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
home = Path(sys.argv[3]).resolve()
assert Path.home() == home
assert all(Path(os.environ[name]).is_relative_to(home)
           for name in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME',
                        'XDG_STATE_HOME', 'HERMES_HOME', 'OPENCLAW_STATE_DIR'))
assert not any(Path(path).resolve().is_relative_to(source) for path in sys.path if path)
spec = importlib.util.spec_from_file_location(
    'owner_mode', installed/'__init__.py', submodule_search_locations=[str(installed)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
names = ('model_metadata', 'profiles', 'proxy', 'plugin', 'serve')
for name in names:
    importlib.import_module('owner_mode.'+name)
for name, module in sys.modules.items():
    if name == 'owner_mode' or name.startswith('owner_mode.'):
        assert Path(module.__file__).resolve().is_relative_to(installed), name
print(json.dumps(list(names)))
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            home = root/'home'
            home.mkdir()
            wizard.provision(home, ['hermes', 'openclaw'], 'http://127.0.0.1:1/v1',
                             'fixture-model', 'fixture-profile', [], False)
            config = wizard.load_config(home/'.openclaw/openclaw.json')
            release = Path(config['plugins']['entries']['owner-mode']['config']['corePath'])
            packages = {'hermes': home/'.hermes/plugins/owner-mode', 'openclaw': release}
            environment = os.environ.copy()
            for name in ('PYTHONPATH', 'PYTHONHOME', 'FLASHALIGN_API_KEY',
                         'FLASHALIGN_PROFILE_PATH', 'FLASHALIGN_BRIDGE_INSTANCE'):
                environment.pop(name, None)
            environment.update({'HOME': str(home), 'XDG_CONFIG_HOME': str(home/'.config'),
                                'XDG_DATA_HOME': str(home/'.local/share'),
                                'XDG_CACHE_HOME': str(home/'.cache'),
                                'XDG_STATE_HOME': str(home/'.local/state'),
                                'HERMES_HOME': str(home/'.hermes'),
                                'OPENCLAW_STATE_DIR': str(home/'.openclaw')})
            for harness, installed in packages.items():
                with self.subTest(harness=harness):
                    self.assertTrue(set(wizard.MODULES) <= {path.name for path in installed.iterdir()})
                    result = subprocess.run(
                        [sys.executable, '-I', '-c', script, str(installed), str(KIT_ROOT.parent), str(home)],
                        cwd=root, env=environment, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout),
                                     ['model_metadata', 'profiles', 'proxy', 'plugin', 'serve'])

    def test_shared_grouped_contract_copies_match(self):
        root = KIT_ROOT.parent
        canonical = (KIT_ROOT/'common/deepseek_grouped.py').read_bytes()
        self.assertEqual((root/'hermes-owner-mode/deepseek_grouped.py').read_bytes(), canonical)
        self.assertEqual((KIT_ROOT/'vllm-deepseek/flashalign_deepseek/deepseek_grouped.py').read_bytes(), canonical)

    def test_all_packaged_backends_declare_their_rank_limit(self):
        expected = {'sglang-glm53-flash': 1, 'sglang-qwen4': 1}
        for path in KIT_ROOT.glob('*/backend.json'):
            with self.subTest(backend=path.parent.name):
                metadata = json.loads(path.read_text())
                if metadata.get('profile_schemas') == ['flashalign.native_wo_b.v1']:
                    self.assertNotIn('maximum_rank', metadata)
                else:
                    self.assertEqual(metadata['maximum_rank'], expected.get(path.parent.name, 128))
        self.assertTrue(all('maximum_rank' in row or row.get('variant') == 'deepseek-v41-native'
                            for row in backend_catalog()))

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
