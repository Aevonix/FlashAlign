# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Local engine planning: format checks, speculative continuity and private staging."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
from unittest import mock

from test_setup import make_profile, wizard
from owner_mode.backend_setup import KIT_ROOT, _physical_devices_by_process, _prepare_runtime, active_speculation, check_backend, prepare_backend
from owner_mode.model_metadata import inspect_model
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError
from owner_mode.quickstart import make_default_plan


def gguf(path, architecture='qwen35', width=2):
    fields = [('general.architecture', 8, architecture),
              (architecture+'.embedding_length', 4, width),
              ('tokenizer.ggml.tokens', 9, ['a', 'b', 'c'])]
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data))+data
    with path.open('wb') as stream:
        stream.write(b'GGUF'+struct.pack('<IQQ', 3, 0, len(fields)))
        for name, kind, value in fields:
            stream.write(string(name)+struct.pack('<I', kind))
            if kind == 8:
                stream.write(string(value))
            elif kind == 9:
                stream.write(struct.pack('<IQ', 8, len(value)))
                for item in value:
                    stream.write(string(item))
            else:
                stream.write(struct.pack('<I', value))


class BackendPlanning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.profile = make_profile(self.root/'profile')
        metadata = json.loads((self.profile/'profile.json').read_text())
        metadata['sites'][0].update(scale=1, tensor='model.language_model.layers.0.self_attn.o_proj.weight')
        (self.profile/'profile.json').write_text(json.dumps(metadata))
        self.model = self.root/'model'
        self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'hidden_size': 2}))
        self.spec = {'kind': 'vllm', 'model_path': str(self.model), 'devices': ['0']}
        self.proc = self.root/'proc'
        self.proc.mkdir()

    def check(self, spec=None):
        return check_backend(self.home, spec or self.spec, validate_profile(self.profile),
                             model_id='bench', proc_root=self.proc)

    def process(self, argv):
        process = self.proc/'123'
        process.mkdir()
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in argv)+b'\0')
        (process/'cwd').symlink_to(self.root)

    def test_gguf_metadata_skips_token_array_without_importing_model_library(self):
        path = self.root/'model.gguf'
        gguf(path)
        value = inspect_model(path)
        self.assertEqual((value['format'], value['model_type'], value['hidden_size']), ('GGUF', 'qwen3_5', 2))
        self.assertNotIn('tokenizer.ggml.tokens', value['metadata'])

    def test_truncated_gguf_is_rejected(self):
        path = self.root/'broken.gguf'
        path.write_bytes(b'GGUF'+struct.pack('<IQQ', 3, 0, 1))
        with self.assertRaisesRegex(SetupError, 'metadata'):
            inspect_model(path)

    def test_check_is_read_only_and_selects_tested_ngram_default(self):
        value = self.check()
        self.assertEqual(value['drafting']['speculative_config']['method'], 'ngram')
        self.assertEqual(list(self.home.iterdir()), [])

    def test_unknown_architecture_does_not_get_a_qwen_profile(self):
        (self.model/'config.json').write_text(json.dumps({'model_type': 'other', 'hidden_size': 2}))
        with self.assertRaisesRegex(SetupError, 'model family'):
            self.check()

    def test_unqualified_quantization_is_reported_before_staging(self):
        (self.model/'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'hidden_size': 2,
                                                        'quantization_config': {'quant_method': 'fp8'}}))
        with self.assertRaisesRegex(SetupError, 'BF16'):
            self.check()

    def test_parallel_engine_and_listener_overrides_are_rejected(self):
        with self.assertRaisesRegex(SetupError, 'GPU count'):
            self.check({**self.spec, 'devices': ['0', '1']})
        with self.assertRaisesRegex(SetupError, 'cannot override'):
            self.check({**self.spec, 'arguments': ['--model=unrelated']})

    def test_vllm_underscore_context_cannot_bypass_hermes_floor(self):
        found = {'hermes': {'detected': True, 'runtime_compatible': True,
                            'context_requirements': []}}
        for arguments, message in ((['--max_model_len', '8192'], 'runtime minimum context=64000'),
                                   (['--max_model_len=8192'], 'runtime minimum context=64000'),
                                   (['--max-model-l', '8192'], 'complete vLLM option')):
            with self.subTest(arguments=arguments):
                plan = self.root/'plan.json'
                plan.write_text(json.dumps({'model': 'bench', 'profile': str(self.profile),
                    'harnesses': ['hermes'], 'tools_default': False,
                    'contacts': [{'channel': 'sms', 'account': 'default', 'contact': '+15550000101'}],
                    'engine': {**self.spec, 'arguments': arguments}}))
                plan.chmod(0o600)
                provision = mock.Mock()
                with (mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
                      mock.patch('owner_mode.automation.prepare_engine',
                                 side_effect=AssertionError('preparation ran before context check')) as prepare):
                    with self.assertRaisesRegex(SetupError, message):
                        wizard.run_plan(plan, self.home, found, provision, apply=True)
                prepare.assert_not_called()
                provision.assert_not_called()
                self.assertEqual(list(self.home.iterdir()), [])

    def test_vllm_underscore_options_preserve_values_and_launch_context(self):
        from owner_mode.drafting import argument
        spec = {**self.spec, 'arguments': ['--max_model_len=131072',
                                          '--generation_config', '/models/source_config',
                                          '--hf_overrides.rope_scaling', '{"rope_type":"linear"}']}
        checked = self.check(spec)
        self.assertEqual(checked['context_window']['tokens'], 131072)
        self.assertEqual(checked['arguments'], ['--max-model-len=131072',
            '--generation-config', '/models/source_config',
            '--hf-overrides.rope_scaling', '{"rope_type":"linear"}'])
        with (mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python': sys.executable}),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None)):
            deployment = prepare_backend(self.home, spec, self.profile, model_id='bench')
        argv = deployment['commands'][0]['argv']
        self.assertEqual(argument(argv, '--max-model-len'), '131072')
        self.assertEqual(argument(argv, '--generation-config'), '/models/source_config')
        self.assertEqual(json.loads(argument(argv, '--speculative-config')),
                         checked['drafting']['speculative_config'])
        self.assertEqual(spec['arguments'][0], '--max_model_len=131072')

    def test_vllm_reserved_underscore_aliases_fail_before_staging(self):
        for arguments in (['--tensor_parallel_size', '2'], ['--served_model_name=foreign'],
                          ['--speculative_config={"method":"ngram"}'],
                          ['--disable_speculative_decoding'], ['--spec_draft_n_max=0']):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(SetupError, 'cannot override'):
                    self.check({**self.spec, 'arguments': arguments})
                self.assertEqual(list(self.home.iterdir()), [])

    def test_active_vllm_underscore_speculation_is_preserved(self):
        config = {'method': 'ngram', 'num_speculative_tokens': 7,
                  'prompt_lookup_min': 2, 'prompt_lookup_max': 8}
        self.process(['vllm', 'serve', str(self.model), '--speculative_config='+json.dumps(config)])
        self.assertEqual(self.check()['drafting']['speculative_config'], config)
        (self.proc/'123/cmdline').write_bytes(b'\0'.join(value.encode() for value in
            ['vllm', 'serve', str(self.model), '--speculative_config={invalid'])+b'\0')
        with self.assertRaisesRegex(SetupError, 'could not be read'):
            self.check()

    def test_vllm_protected_abbreviations_and_unreadable_active_config_stop(self):
        for arguments in (['--po', '9001'], ['--tensor-parallel-s=2'], ['--speculative-conf={}']):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(SetupError, 'complete vLLM option'):
                self.check({**self.spec, 'arguments': arguments})
        self.process(['vllm', 'serve', str(self.model), '--speculative-conf={"method":"ngram"}'])
        with self.assertRaisesRegex(SetupError, 'could not be read'):
            self.check()
        self.assertEqual(list(self.home.iterdir()), [])

    def test_active_ngram_settings_are_retained_exactly(self):
        config = {'method': 'ngram', 'num_speculative_tokens': 7, 'prompt_lookup_min': 2, 'prompt_lookup_max': 8}
        self.process(['python', '-m', 'vllm', 'serve', str(self.model), '--speculative-config', json.dumps(config)])
        value = self.check()
        self.assertEqual(value['drafting']['speculative_config'], config)
        with self.assertRaisesRegex(SetupError, 'differ'):
            self.check({**self.spec, 'speculative_config': {'method': 'ngram'}})

    def test_malformed_active_settings_cannot_silently_disable_speculation(self):
        self.process(['python', '-m', 'vllm', 'serve', str(self.model), '--speculative-config', '{invalid'])
        with self.assertRaisesRegex(SetupError, 'could not be read'):
            self.check()

    def test_unsupported_active_method_stops_without_changing_it(self):
        self.process(['python', '-m', 'vllm', 'serve', str(self.model), '--speculative-config', '{"method":"mtp"}'])
        with self.assertRaisesRegex(SetupError, 'will not be disabled'):
            self.check()

    def test_other_backend_on_disjoint_gpu_is_not_replaced(self):
        self.process(['python', '-m', 'sglang.launch_server', '--model-path', str(self.model),
                      '--speculative-algorithm', 'DFLASH', '--speculative-draft-model-path', '/draft'])
        (self.proc/'123/environ').write_bytes(b'CUDA_VISIBLE_DEVICES=1\0NVIDIA_VISIBLE_DEVICES=all\0')
        with mock.patch('owner_mode.backend_setup._physical_devices_by_process', return_value={123:{'1'}}):
            value = self.check()
        self.assertEqual(value['drafting']['speculative_config']['method'], 'ngram')
        # Discovery without a selected placement still observes the active lane.
        self.assertEqual(active_speculation(self.model, 'bench', self.proc)['kind'], 'sglang')

    def test_overlapping_or_unknown_visibility_preserves_active_backend_guard(self):
        self.process(['python', '-m', 'sglang.launch_server', '--model-path', str(self.model),
                      '--speculative-algorithm', 'DFLASH', '--speculative-draft-model-path', '/draft'])
        for environment in (b'', b'CUDA_VISIBLE_DEVICES=0,1\0', b'CUDA_VISIBLE_DEVICES=GPU-uuid\0',
                            b'CUDA_VISIBLE_DEVICES=1\0NVIDIA_VISIBLE_DEVICES=GPU-uuid\0',
                            b'CUDA_VISIBLE_DEVICES=1\0CUDA_VISIBLE_DEVICES=2\0'):
            with self.subTest(environment=environment):
                (self.proc/'123/environ').write_bytes(environment)
                with self.assertRaisesRegex(SetupError, 'different speculative backend'):
                    self.check()

    def test_disjoint_same_backend_does_not_copy_its_decoding_configuration(self):
        self.process(['python', '-m', 'vllm', 'serve', str(self.model), '--speculative-config',
                      json.dumps({'method':'ngram','num_speculative_tokens':7})])
        (self.proc/'123/environ').write_bytes(b'CUDA_VISIBLE_DEVICES=1\0')
        with mock.patch('owner_mode.backend_setup._physical_devices_by_process', return_value={123:{'1'}}):
            value = self.check()
        self.assertEqual(value['drafting']['speculative_config']['num_speculative_tokens'], 3)

    def test_unconfirmed_or_remapped_gpu_ownership_keeps_active_guard(self):
        self.process(['python', '-m', 'sglang.launch_server', '--model-path', str(self.model),
                      '--speculative-algorithm', 'DFLASH', '--speculative-draft-model-path', '/draft'])
        (self.proc/'123/environ').write_bytes(b'CUDA_VISIBLE_DEVICES=1\0')
        for placements in ({}, {123:{'0'}}, {123:{'1','2'}}):
            with (self.subTest(placements=placements),
                  mock.patch('owner_mode.backend_setup._physical_devices_by_process', return_value=placements)):
                with self.assertRaisesRegex(SetupError, 'different speculative backend'):
                    self.check()

    def test_driver_ownership_includes_engine_parent_without_its_own_cuda_context(self):
        rows = [SimpleNamespace(stdout='0, GPU-a\n1, GPU-b\n'),
                SimpleNamespace(stdout='42, GPU-b\n')]
        stats = {'/proc/42/stat':'42 (worker (gpu)) S 12', '/proc/12/stat':'12 (engine) S 1'}
        with (mock.patch('owner_mode.backend_setup.subprocess.run', side_effect=rows),
              mock.patch.object(Path, 'read_text', autospec=True, side_effect=lambda path:stats[str(path)])):
            self.assertEqual(_physical_devices_by_process(Path('/proc')), {42:{'1'},12:{'1'}})
        with mock.patch('owner_mode.backend_setup.subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(_physical_devices_by_process(Path('/proc')), {})

    def test_per_rank_tensorfold_visibility_does_not_exclude_a_partial_world(self):
        self.process(['python', '-m', 'tensorfold', 'serve', str(self.model),
                      '--tp', '2', '--rank', '0', '--drafter', '/draft'])
        (self.proc/'123/environ').write_bytes(b'CUDA_VISIBLE_DEVICES=1\0')
        with mock.patch('owner_mode.backend_setup._physical_devices_by_process', return_value={123:{'1'}}):
            with self.assertRaisesRegex(SetupError, 'different speculative backend'):
                self.check()

    def test_duplicate_gpu_identity_rows_are_not_trusted(self):
        for output in ('0, GPU-a\n1, GPU-a\n', '0, GPU-a\n0, GPU-b\n'):
            with self.subTest(output=output), mock.patch('owner_mode.backend_setup.subprocess.run',
                                                       return_value=SimpleNamespace(stdout=output)):
                self.assertEqual(_physical_devices_by_process(Path('/proc')), {})
        self.assertFalse((self.home/'.config').exists())

    def test_other_engine_speculation_is_not_replaced_during_migration(self):
        self.process(['python', '-m', 'sglang.launch_server', '--model-path', str(self.model),
                      '--speculative-algorithm', 'DFLASH', '--speculative-draft-model-path', str(self.root/'draft')])
        with self.assertRaisesRegex(SetupError, 'different speculative backend'):
            self.check()

    def test_sglang_detects_a_local_compatible_draft(self):
        draft = self.root/'draft'
        draft.mkdir()
        (draft/'config.json').write_text('{"architectures":["DFlash2DraftModel"]}')
        value = self.check({**self.spec, 'kind': 'sglang', 'draft_model': str(draft)})
        self.assertEqual(value['drafting']['draft_algorithm'], 'DFLASH')
        self.assertEqual(value['drafting']['draft_model'], str(draft))

    def test_sglang_defaults_are_replaced_without_consuming_flags_or_negative_values(self):
        from owner_mode.backend_setup import _sglang_arguments
        defaults = ['--context-length', '65536', '--offset', '-1',
                    '--enable-metrics', '--enable-cache-report']
        for explicit in (['--context-length', '32768'], ['--context-length=32768']):
            with self.subTest(explicit=explicit):
                result = _sglang_arguments(defaults, [*explicit, '--offset=-2', '--enable-metrics'])
                self.assertEqual(result, ['--enable-cache-report', '--context-length', '32768',
                                          '--offset', '-2', '--enable-metrics'])
        duplicates = ['--context-length', '32768', '--context-length=16384']
        self.assertEqual(_sglang_arguments(defaults, duplicates).count('--context-length'), 2)
        self.assertEqual(_sglang_arguments(['--label=default'], ['--label=--literal']),
                         ['--label=--literal'])

    def test_sglang_saved_command_contains_one_explicit_context_setting(self):
        from owner_mode.drafting import argument
        draft = self.root/'draft'
        draft.mkdir()
        (draft/'config.json').write_text('{"architectures":["DFlash2DraftModel"]}')
        for explicit in (['--context-length', '65536'], ['--context-length=32768']):
            with self.subTest(explicit=explicit):
                spec = {**self.spec, 'kind': 'sglang', 'draft_model': str(draft),
                        'arguments': [*explicit, '--enable-cache-report']}
                with (mock.patch('owner_mode.backend_setup._prepare_runtime',
                                 return_value={'python':sys.executable, 'source':str(self.root/'native')}),
                      mock.patch('owner_mode.backend_setup.active_speculation', return_value=None)):
                    deployment = prepare_backend(self.home, spec, self.profile, model_id='bench')
                argv = deployment['commands'][0]['argv']
                self.assertEqual(argv.count('--context-length'), 1)
                self.assertEqual(argv.count('--enable-cache-report'), 1)
                self.assertEqual(argument(argv, '--context-length'),
                                 '32768' if '=' in explicit[0] else '65536')
                self.assertEqual(argument(argv, '--draft-algorithm'), 'DFLASH')
                self.assertEqual(argument(argv, '--draft-model'), str(draft))

    def test_llamacpp_requires_gguf_and_preserves_external_draft(self):
        model = self.root/'model.gguf'
        draft = self.root/'draft.gguf'
        gguf(model)
        gguf(draft)
        value = self.check({**self.spec, 'kind': 'llamacpp', 'model_path': str(model), 'draft_model': str(draft)})
        self.assertEqual(value['drafting']['draft_algorithm'], 'draft-simple')
        with self.assertRaisesRegex(SetupError, 'model family'):
            self.check({**self.spec, 'kind': 'llamacpp'})

    def test_llamacpp_uses_qualified_64k_default_in_check_and_launch(self):
        from owner_mode.drafting import argument
        model, draft = self.root/'model.gguf', self.root/'draft.gguf'
        gguf(model);gguf(draft)
        spec = {**self.spec, 'kind': 'llamacpp', 'model_path': str(model), 'draft_model': str(draft)}
        self.assertEqual(self.check(spec)['context_window']['tokens'], 65536)
        runtime = {'python': sys.executable, 'binary': '/fixture/llama-server', 'gguf_python': str(self.root)}
        with (mock.patch('owner_mode.backend_setup._prepare_runtime', return_value=runtime),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None)):
            deployment = prepare_backend(self.home, spec, self.profile, model_id='bench')
        native = json.loads((Path(deployment['state'])/'native-arguments.json').read_text())
        self.assertEqual(int(argument(native, '--ctx-size')) // int(argument(native, '--parallel')), 65536)
        self.assertEqual(argument(native, '--parallel'), '2')
        self.assertEqual(argument(native, '--spec-type'), 'draft-simple')
        self.assertEqual(argument(native, '--spec-draft-n-max'), '4')

    def test_vllm_context_default_and_override_match_the_actual_launch(self):
        from owner_mode.drafting import argument
        for arguments, expected in (([], '65536'), (['--max-model-len', '8192'], '8192')):
            with self.subTest(arguments=arguments):
                spec = {**self.spec, 'arguments': arguments}
                self.assertEqual(self.check(spec)['context_window']['tokens'], int(expected))
                with (mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python':sys.executable}),
                      mock.patch('owner_mode.backend_setup.active_speculation', return_value=None)):
                    deployment = prepare_backend(self.home, spec, self.profile, model_id='bench')
                argv = deployment['commands'][0]['argv']
                self.assertEqual(argument(argv, '--max-model-len'), expected)
                self.assertEqual(argument(argv, '--max-num-seqs'), '4')
                self.assertEqual(json.loads(argument(argv, '--speculative-config'))['method'], 'ngram')

    def test_llamacpp_explicit_context_and_parallel_override_match_launch(self):
        from owner_mode.drafting import argument
        model = self.root/'model.gguf';gguf(model)
        spec = {**self.spec, 'kind': 'llamacpp', 'model_path': str(model),
                'arguments': ['--ctx-size', '32768', '--parallel', '1']}
        self.assertEqual(self.check(spec)['context_window']['tokens'], 32768)
        runtime = {'python': sys.executable, 'binary': '/fixture/llama-server', 'gguf_python': str(self.root)}
        with (mock.patch('owner_mode.backend_setup._prepare_runtime', return_value=runtime),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None)):
            deployment = prepare_backend(self.home, spec, self.profile, model_id='bench')
        native = json.loads((Path(deployment['state'])/'native-arguments.json').read_text())
        self.assertEqual(argument(native, '--ctx-size'), '32768')
        self.assertEqual(argument(native, '--parallel'), '1')

    def test_malformed_default_arguments_reject_before_staging(self):
        model = self.root/'model.gguf';gguf(model)
        kit = KIT_ROOT/'llamacpp'
        metadata = json.loads((kit/'backend.json').read_text())
        for value in ('--ctx-size 131072', [131072]):
            with (self.subTest(value=value),
                  mock.patch('owner_mode.backend_setup._kits', return_value=iter([
                      (kit, {**metadata, 'default_arguments': value}, 'llamacpp')]))):
                with self.assertRaisesRegex(SetupError, 'default arguments must be a string array'):
                    self.check({**self.spec, 'kind': 'llamacpp', 'model_path': str(model)})
                self.assertEqual(list(self.home.iterdir()), [])

    def test_staging_copies_runtime_and_creates_unique_private_keys_without_starting(self):
        with mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python': sys.executable}), \
                mock.patch('owner_mode.backend_setup.active_speculation', return_value=None):
            first = prepare_backend(self.home, self.spec, self.profile, model_id='bench')
            again = prepare_backend(self.home, self.spec, self.profile, model_id='bench')
            second = prepare_backend(self.home, {**self.spec, 'port': 18912}, self.profile, model_id='bench')
        first_key = Path(first['key_file'])
        self.assertEqual(first_key.read_bytes(), Path(again['key_file']).read_bytes())
        self.assertNotEqual(first_key.read_bytes(), Path(second['key_file']).read_bytes())
        self.assertEqual(first_key.stat().st_mode & 0o777, 0o600)
        self.assertFalse((Path(first['state'])/'running.json').exists())
        self.assertTrue((Path(first['source'])/'flashalign_vllm/core.py').is_file())
        argv = first['commands'][0]['argv']
        self.assertIn('flashalign_vllm', argv)
        self.assertIn('--speculative-config', argv)
        self.assertNotIn(first_key.read_bytes().hex(), json.dumps(first))

    def test_modified_staged_code_is_never_overwritten(self):
        with mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python': sys.executable}), \
                mock.patch('owner_mode.backend_setup.active_speculation', return_value=None):
            first = prepare_backend(self.home, self.spec, self.profile, model_id='bench')
            path = Path(first['source'])/'flashalign_vllm/core.py'
            path.write_text('changed by operator')
            with self.assertRaisesRegex(SetupError, 'changed'):
                prepare_backend(self.home, self.spec, self.profile, model_id='bench')
            self.assertEqual(path.read_text(), 'changed by operator')

    def test_vllm_variant_uses_its_validating_launcher(self):
        kit = KIT_ROOT/'vllm-gemma4'
        metadata = json.loads((kit/'backend.json').read_text())
        metadata.update(model_types=['qwen3_5'], compatible_models=[])
        # This synthetic launcher test changes the model family, so it must not
        # retain a capability declaration for the original Gemma family.
        metadata.pop('openclaw_compat_by_model_type', None)
        metadata.pop('request_compatibility_by_model_type', None)
        metadata.pop('input_modalities_by_model_type', None)
        with (mock.patch('owner_mode.backend_setup._kits', return_value=iter([(kit, metadata, 'vllm')])),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value={'python': sys.executable}),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None)):
            deployment = prepare_backend(self.home, self.spec, self.profile, model_id='bench')
        argv = deployment['commands'][0]['argv']
        self.assertEqual(Path(argv[1]).name, 'launch.py')
        self.assertIn('--speculative-config', argv)
        self.assertNotIn('flashalign_vllm', argv)

    def test_default_qwen_setup_uses_public_vllm_kit_without_package_download(self):
        (self.model/'config.json').write_text('{"model_type":"qwen3_5","hidden_size":5120}')
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=None, engine_bundle=None,
                               backend='auto', profiles=[], engine_python=None, devices='0',
                               tools_default=None, engine_only=True)
        with (mock.patch('owner_mode.quickstart.download_engine_bundle', side_effect=AssertionError('network')),
              redirect_stdout(io.StringIO())):
            path = make_default_plan(args, {}, wizard.Console(color=False))
        plan = json.loads(path.read_text())
        self.assertEqual(plan['engine']['kind'], 'vllm')
        self.assertTrue(plan['engine_only'])
        self.assertNotIn('bundle', plan['engine'])
        self.assertEqual(plan['harnesses'], [])

    def test_default_setup_retains_active_backend_and_speculative_arguments(self):
        (self.model/'config.json').write_text('{"model_type":"qwen3_5","hidden_size":5120}')
        draft = self.root/'draft'
        draft.mkdir()
        (draft/'config.json').write_text('{"architectures":["DFlash2DraftModel"]}')
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=None, engine_bundle=None,
                               backend='auto', profiles=[], engine_python=None, devices='0',
                               tools_default=None, engine_only=True)
        active = {'kind': 'sglang', 'draft_model': str(draft), 'draft_algorithm': 'DFLASH',
                  'speculative_arguments': ['--speculative-num-draft-tokens', '12']}
        with (mock.patch('owner_mode.backend_setup.active_speculation', return_value=active),
              redirect_stdout(io.StringIO())):
            path = make_default_plan(args, {}, wizard.Console(color=False))
        plan = json.loads(path.read_text())
        self.assertEqual(plan['engine']['kind'], 'sglang')
        self.assertEqual(plan['engine']['speculative_arguments'], active['speculative_arguments'])
        self.assertEqual(plan['engine']['draft_model'], str(draft))

    def test_pip_build_artifacts_do_not_change_the_staged_runtime(self):
        kit, state = self.root/'kit', self.root/'state'
        kit.mkdir();state.mkdir()
        (kit/'requirements.txt').write_text('')
        (kit/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="1"\n')
        (state/'venv/bin').mkdir(parents=True)
        (state/'venv/bin/python').touch()
        def build(command, _log):
            if command[-2] == '--no-deps':
                source = Path(command[-1])
                (source/'build').mkdir()
                (source/'fixture.egg-info').mkdir()
        with (mock.patch('owner_mode.backend_setup._run', side_effect=build),
              mock.patch('owner_mode.backend_setup.prepare_python_headers', return_value={})):
            _prepare_runtime('vllm', {'install_dependencies': True}, kit, state)
        self.assertEqual({p.name for p in kit.iterdir()}, {'requirements.txt', 'pyproject.toml'})
        self.assertFalse(list(state.glob('package-build-*')))


if __name__ == '__main__':
    unittest.main()
