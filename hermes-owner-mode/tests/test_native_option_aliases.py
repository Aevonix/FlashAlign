# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Exercise real launcher parsers with synthetic loaders, never a model or GPU."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]/'engines'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def native_cli(kind):
    package_name = '_native_alias_'+kind.replace('-', '_')
    package_dir = ROOT/kind/('flashalign_deepseek' if kind == 'vllm-deepseek' else 'flashalign_vllm')
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    core = types.ModuleType(package_name+'.core')
    core.load_key = lambda _: b''
    core.load_profiles = lambda *args, **kwargs: {'test': types.SimpleNamespace(meta={'model': {'id': 'model'}})}
    plugin = types.ModuleType(package_name+'.plugin')
    registered, calls = [], []
    plugin.register = lambda: registered.append(True)
    backend = types.ModuleType('vllm.entrypoints.cli.main')
    backend.main = lambda: calls.append(sys.argv.copy())
    with mock.patch.dict(sys.modules, {package_name: package, package_name+'.core': core,
                         package_name+'.plugin': plugin, 'vllm.entrypoints.cli.main': backend}), \
         mock.patch.dict(os.environ), mock.patch.object(sys, 'argv', ['native-test']):
        cli = load_file(package_name+'.cli', package_dir/'cli.py')
        if kind == 'vllm-deepseek':
            cli.validate_model = lambda *args: 'model'
        yield cli, calls, registered


class NativeOptionAliases(unittest.TestCase):
    def arguments(self, kind):
        args = ['--model', '/models/source_config', '--profile', '/profiles/test',
                '--key_file', '/synthetic/key']
        if kind == 'vllm-deepseek':
            args += ['--allow_research']
        return args

    def test_direct_clis_reject_underscore_and_nested_protected_overrides(self):
        for kind in ('vllm', 'vllm-gemma4', 'vllm-deepseek'):
            for extra in (['--tensor_parallel_size=2'], ['--middleware', 'foreign'],
                          ['--no_enforce_eager'], ['--compilation_config.mode=3'],
                          ['--speculative_config.method=foreign'], ['--po', '9001'],
                          ['--tensor-parallel-s=2'], ['--no-enforce-e'], ['--max-model-l', '8192']):
                with self.subTest(kind=kind, extra=extra), native_cli(kind) as (cli, calls, registered):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        cli.main(self.arguments(kind)+extra)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertEqual(calls, [])
                    self.assertEqual(registered, [])

    def test_direct_clis_preserve_supported_context_values_and_speculation(self):
        for kind in ('vllm', 'vllm-gemma4', 'vllm-deepseek'):
            config = {'method': 'dspark' if kind == 'vllm-deepseek' else 'ngram',
                      'num_speculative_tokens': 5 if kind == 'vllm-deepseek' else 3}
            if kind == 'vllm-deepseek': config['model'] = '/models/source_config'
            with self.subTest(kind=kind), native_cli(kind) as (cli, calls, registered):
                cli.main(self.arguments(kind)+['--speculative_config='+json.dumps(config),
                    '--max_model_len=131072', '--generation_config', '/models/generation_config'])
                self.assertEqual(len(calls), 1)
                self.assertEqual(registered, [True])
                argv = calls[0]
                self.assertIn('--max-model-len=131072', argv)
                self.assertEqual(argv[argv.index('--generation-config')+1], '/models/generation_config')
                self.assertEqual(json.loads(argv[argv.index('--speculative-config')+1]), config)
                self.assertIn('--enforce-eager', argv)
                self.assertNotIn('--no-enforce-eager', argv)

    def test_deepseek_inner_cli_protects_the_outer_metadata_and_graph_contract(self):
        for extra in (['--hf_overrides={"model_type":"other"}'],
                      ['--hf_overrides.model_type=other'], ['--quantization', 'other'],
                      ['--enforce_eager=false']):
            with self.subTest(extra=extra), native_cli('vllm-deepseek') as (cli, calls, _):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(self.arguments('vllm-deepseek')+extra)
                self.assertEqual(calls, [])

    def test_deepseek_supervisor_parser_and_builder_share_normalization(self):
        launcher = load_file('_native_alias_deepseek_launcher', ROOT/'vllm-deepseek/launch.py')
        args = types.SimpleNamespace(model=Path('/models/stock'), model_receipt=Path('/state/receipt'),
            key_file=Path('/state/key'), replay_file=Path('/state/replay'), profile=[Path('/profile')],
            host='127.0.0.1', port=8041, speculative_config=None)
        metadata = json.loads((ROOT/'vllm-deepseek/backend.json').read_text())
        for extra in (['--hf_overrides.model_type=other'], ['--no_enforce_eager'],
                      ['--tensor_parallel_size=2'], ['--replay_file=/foreign']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'cannot override'):
                launcher.build_command(args, extra, metadata)
        for extra in (['--po', '9001'], ['--max-model-l=8192'], ['--no-enforce-e'], ['--tensor-parallel-s=2']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'complete vLLM option'):
                launcher.build_command(args, extra, metadata)
        argv = ['direct', '--model', '/models/stock', '--profile', '/profile',
                '--model_receipt', '/state/receipt', '--key_file', '/state/key',
                '--replay_file', '/state/replay', '--max_model_len=131072', '--dry_run']
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', argv), redirect_stdout(output):
            launcher.main()
        command = json.loads(output.getvalue())['command']
        self.assertIn('--max-model-len=131072', command)
        self.assertEqual(json.loads(command[command.index('--speculative-config')+1])['num_speculative_tokens'], 5)

    def test_gemma_outer_parser_rejects_metadata_aliases_and_retains_valid_options(self):
        launcher = load_file('_native_alias_gemma_launcher', ROOT/'vllm-gemma4/launch.py')
        base = ['direct', '--model', '/model', '--profile', '/profile', '--key_file', '/state/key']
        for extra in (['--hf_overrides={"model_type":"other"}'],
                      ['--hf_overrides.model_type=other'], ['--hf_config_path=/foreign'],
                      ['--po', '9001'], ['--hf-override={}'], ['--max-model-l=8192']):
            with self.subTest(extra=extra), mock.patch.object(sys, 'argv', base+extra):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): launcher.main()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);profile = root/'profile';profile.mkdir()
            identity = {'id': 'model', 'revision': 'pin'}
            (profile/'profile.json').write_text(json.dumps({'model': identity, 'sites': []}))
            (root/'backend.json').write_text(json.dumps({'compatible_models': [identity],
                'site_tensor_patterns': {}, 'default_speculative_config': {'method': 'ngram'},
                'tested_speculative_methods': ['ngram'], 'default_arguments': ['--max-model-len', '65536']}))
            argv = ['direct', '--model', '/model', '--profile', str(profile), '--key_file', '/state/key',
                    '--max_model_len=131072', '--dry_run']
            output = io.StringIO()
            with mock.patch.object(launcher, 'ROOT', root), mock.patch.object(launcher, 'fingerprint', return_value=True), \
                 mock.patch.object(sys, 'argv', argv), redirect_stdout(output):
                launcher.main()
            command = json.loads(output.getvalue())['command']
            self.assertIn('--max-model-len=131072', command)
            self.assertEqual(json.loads(command[command.index('--speculative-config')+1]), {'method': 'ngram'})

    def test_runtime_normalizers_match_and_preserve_nested_values(self):
        paths = [ROOT/'vllm/flashalign_vllm/arguments.py',
                 ROOT/'vllm-gemma4/flashalign_vllm/arguments.py',
                 ROOT/'vllm-deepseek/flashalign_deepseek/arguments.py']
        self.assertEqual(len({path.read_bytes() for path in paths}), 1)
        inputs = ['--hf_overrides.rope_scaling={"rope_type":"linear"}', '/with_underscore',
                  '--max_model_len', '65536', '--generation_config=/with_underscore']
        expected = ['--hf-overrides.rope_scaling={"rope_type":"linear"}', '/with_underscore',
                    '--max-model-len', '65536', '--generation-config=/with_underscore']
        for index, path in enumerate(paths):
            module = load_file('_native_arguments_'+str(index), path)
            self.assertEqual(module.normalize_arguments(inputs), expected)
            self.assertEqual(module.option_name(expected[0]), '--hf-overrides')


if __name__ == '__main__': unittest.main()
