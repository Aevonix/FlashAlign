# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Gemma GGUF qualification and scale-aware setup, without starting an engine."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_setup import make_profile, wizard
from owner_mode.backend_setup import (KIT_ROOT, _check_llamacpp_strengths,
                                      check_backend, prepare_backend)
from owner_mode.model_metadata import inspect_model
from owner_mode.profiles import validate_profile
from owner_mode.quickstart import make_default_plan
from owner_mode.setup_runtime import SetupError, normalize_engine_arguments


def write_gguf(path, architecture='gemma4', *, tensor_kind=0, padding=b'', payload=b'weights'):
    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data
    fields = [('general.architecture', 8, architecture),
              (architecture+'.embedding_length', 4, 2),
              (architecture+'.block_count', 4, 30),
              ('tokenizer.ggml.tokens', 9, ['a', 'b'])]
    header = b'GGUF'+struct.pack('<IQQ', 3, 1, len(fields))
    for name, kind, value in fields:
        header += string(name) + struct.pack('<I', kind)
        if kind == 8:
            header += string(value)
        elif kind == 9:
            header += struct.pack('<IQ', 8, len(value))
            header += b''.join(string(token) for token in value)
        else:
            header += struct.pack('<I', value)
    header += string('blk.5.attn_output.weight') + struct.pack('<IQQIQ', 2, 2, 2, tensor_kind, 0)
    path.write_bytes(header+padding+payload)
    return {'sha256': hashlib.sha256(header).hexdigest(),
            'size': path.stat().st_size, 'header_bytes': len(header)}


class GemmaLlamaSetup(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.home = self.root/'home'
        self.home.mkdir()
        self.proc = self.root/'proc'
        self.proc.mkdir()
        self.model = self.root/'gemma.gguf'
        self.header = write_gguf(self.model)
        self.profile = make_profile(self.root/'profile', model='fixture/gemma', name='gemma-profile')
        manifest = json.loads((self.profile/'profile.json').read_text())
        manifest['model']['revision'] = 'a'*40
        manifest['sites'][0].update(scale=2.997, tensor='model.language_model.layers.0.self_attn.o_proj.weight')
        (self.profile/'profile.json').write_text(json.dumps(manifest))
        self.kit = KIT_ROOT/'llamacpp'
        self.metadata = json.loads((self.kit/'backend.json').read_text())
        self.metadata.update(model_architectures=['qwen35', 'gemma4'],
                             site_scale_mode='positive_finite', maximum_rank=128,
                             speculative_types_by_model_architecture={
                                 'qwen35':['draft-simple'], 'gemma4':['draft-simple']},
                             default_speculative_type_by_model_architecture={},
                             gguf_required_model_architectures=['gemma4'],
                             gguf_models=[{'id': 'fixture/gemma-GGUF', 'revision': 'b'*40,
                                'architecture': 'gemma4', 'profile_name': 'gemma-profile',
                                'header_sha256': self.header['sha256'], 'size': self.header['size'],
                                'profile_model': {'id': 'fixture/gemma', 'revision': 'a'*40}}])
        self.spec = {'kind': 'llamacpp', 'model_path': str(self.model), 'devices': ['0']}

    def check(self, **spec):
        with mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit, self.metadata, 'llamacpp')])):
            return check_backend(self.home, {**self.spec, **spec}, validate_profile(self.profile),
                                 model_id='fixture/gemma', proc_root=self.proc)

    def test_fingerprint_includes_tensor_types_but_excludes_alignment_and_payload(self):
        first = inspect_model(self.model)
        self.assertEqual(first['gguf_header'], self.header)
        self.assertEqual(first['model_type'], 'gemma4')
        self.assertNotIn('tokenizer.ggml.tokens', first['metadata'])
        write_gguf(self.model, padding=b'pad', payload=b'changed payload')
        self.assertEqual(inspect_model(self.model)['gguf_header']['sha256'], self.header['sha256'])
        write_gguf(self.model, tensor_kind=2)
        self.assertNotEqual(inspect_model(self.model)['gguf_header']['sha256'], self.header['sha256'])

    def test_scaled_gemma_profile_matches_pinned_gguf(self):
        result = self.check()
        self.assertEqual(result['gguf_binding']['revision'], 'b'*40)
        self.assertEqual(result['gguf_binding']['profile_model']['revision'], 'a'*40)
        self.assertEqual(result['input_modalities'], ['text'])
        self.assertEqual(result['model']['gguf_architecture'], 'gemma4')
        self.assertEqual(list(self.home.iterdir()), [])

    def test_unqualified_quantization_or_size_stops_before_staging(self):
        for tensor_kind, payload in ((2, b'weights'), (0, b'larger payload')):
            with self.subTest(tensor_kind=tensor_kind, payload=payload):
                write_gguf(self.model, tensor_kind=tensor_kind, payload=payload)
                with self.assertRaisesRegex(SetupError, 'fingerprint'):
                    self.check()
                self.assertEqual(list(self.home.iterdir()), [])

    def test_missing_or_duplicate_binding_stops_before_staging(self):
        for entries in ([], self.metadata['gguf_models']*2):
            with self.subTest(entries=len(entries)):
                with mock.patch.dict(self.metadata, gguf_models=entries):
                    with self.assertRaisesRegex(SetupError, 'fingerprint'):
                        self.check()
                self.assertEqual(list(self.home.iterdir()), [])

    def test_wrong_source_model_or_revision_is_rejected(self):
        pin = self.metadata['gguf_models'][0]
        for field, value, message in (
                ('profile_model', {'id': 'fixture/gemma', 'revision': 'c'*40}, 'profile model or revision'),
                ('profile_model', {'id': 'unrelated/model', 'revision': 'a'*40}, 'profile model or revision')):
            with self.subTest(field=field, value=value), mock.patch.dict(pin, {field: value}):
                with self.assertRaisesRegex(SetupError, message):
                    self.check()

    def test_operator_profile_for_same_pinned_source_is_accepted(self):
        manifest = json.loads((self.profile/'profile.json').read_text())
        manifest['name'] = 'operator-custom-profile'
        (self.profile/'profile.json').write_text(json.dumps(manifest))
        self.assertEqual(self.check()['gguf_binding']['profile_model']['id'], 'fixture/gemma')

    def test_source_base_identity_is_accepted_without_relabeling_profile(self):
        manifest = json.loads((self.profile/'profile.json').read_text())
        manifest['source'] = {'base': {'id': 'fixture/base', 'revision': 'd'*40}}
        (self.profile/'profile.json').write_text(json.dumps(manifest))
        self.metadata['gguf_models'][0]['profile_model'] = {'id': 'fixture/base', 'revision': 'd'*40}
        self.assertEqual(self.check()['gguf_binding']['profile_model']['id'], 'fixture/base')

    def test_wrong_architecture_is_not_selected(self):
        write_gguf(self.model, architecture='llama')
        with self.assertRaisesRegex(SetupError, 'model family'):
            self.check()

    def test_malformed_binding_declaration_is_rejected(self):
        pin = self.metadata['gguf_models'][0]
        for field, value in (('revision', 'unpinned'), ('header_sha256', 'bad'), ('size', True),
                             ('profile_model', {'id':'fixture/gemma','revision':'main'})):
            with self.subTest(field=field), mock.patch.dict(pin, {field: value}):
                with self.assertRaisesRegex(SetupError, 'pinned GGUF checkpoint identity'):
                    self.check()

    def test_strength_capability_cannot_silently_accept_invalid_or_unsupported_settings(self):
        for scale in (0, -1, float('nan'), float('inf'), True, 1e100, 1e-100, 2e38):
            with self.subTest(scale=scale), self.assertRaisesRegex(SetupError, 'finite positive'):
                _check_llamacpp_strengths(self.metadata, [{'scale':scale, 'rank':1}])
        with self.assertRaisesRegex(SetupError, 'rank at most 128'):
            _check_llamacpp_strengths(self.metadata, [{'scale':3, 'rank':129}])
        old = copy.deepcopy(self.metadata)
        del old['site_scale_mode']
        old['site_scale'] = 1.0
        _check_llamacpp_strengths(old, [{'scale':1, 'rank':1}])
        with self.assertRaisesRegex(SetupError, 'site strengths of'):
            _check_llamacpp_strengths(old, [{'scale':3, 'rank':1}])
        with self.assertRaisesRegex(SetupError, 'Unsupported llama.cpp site strength'):
            _check_llamacpp_strengths({**self.metadata, 'site_scale_mode':'unchecked'}, [{'scale':3, 'rank':1}])

    def test_qwen_legacy_binding_behavior_remains_available(self):
        write_gguf(self.model, architecture='qwen35')
        self.assertNotIn('gguf_binding', self.check())

    def test_preparation_preserves_active_speculation_and_writes_matched_model_receipt(self):
        draft = self.root/'draft.gguf'
        write_gguf(draft)
        active = {'kind':'llamacpp', 'draft_model':str(draft), 'draft_algorithm':'draft-simple',
                  'speculative_arguments':['--spec-draft-n-max','7','--spec-draft-ngl','99']}
        runtime = {'python':sys.executable, 'binary':'/fixture/llama-server', 'gguf_python':str(self.root)}
        with (mock.patch('owner_mode.backend_setup._kits', side_effect=lambda:iter([(self.kit, self.metadata, 'llamacpp')])),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=active),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value=runtime)):
            result = prepare_backend(self.home, self.spec, self.profile, model_id='fixture/gemma')
        self.assertEqual(result['gguf_model']['header_sha256'], self.header['sha256'])
        self.assertEqual(result['drafting']['speculative_arguments'], active['speculative_arguments'])
        native = json.loads((Path(result['state'])/'native-arguments.json').read_text())
        self.assertEqual(native[native.index('--spec-draft-n-max')+1], '7')
        self.assertEqual(native[native.index('--spec-type')+1], 'draft-simple')
        self.assertFalse((Path(result['state'])/'running.json').exists())

    def test_default_setup_keeps_gguf_and_active_draft_without_prompts(self):
        draft = self.root/'draft.gguf'
        write_gguf(draft)
        active = {'kind':'llamacpp', 'draft_model':str(draft), 'draft_algorithm':'draft-simple',
                  'speculative_arguments':['--spec-draft-n-max','7']}
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=None, engine_bundle=None,
                               backend='auto', profiles=[], engine_python=None, devices='0',
                               tools_default=None, engine_only=True)
        with (mock.patch('owner_mode.quickstart.bundled_profile_for', return_value=validate_profile(self.profile)),
              mock.patch('owner_mode.backend_setup._kits', side_effect=lambda:iter([(self.kit, self.metadata, 'llamacpp')])),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=active),
              mock.patch('owner_mode.quickstart.download_engine_bundle', side_effect=AssertionError('network')),
              mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')),
              mock.patch('sys.stdout', new_callable=io.StringIO)):
            plan_path = make_default_plan(args, {}, wizard.Console(color=False))
        plan = json.loads(plan_path.read_text())
        self.assertEqual(plan['model'], 'fixture/gemma')
        self.assertEqual(plan['engine']['kind'], 'llamacpp')
        self.assertEqual(plan['engine']['model_path'], str(self.model))
        self.assertEqual(plan['engine']['draft_model'], str(draft))
        self.assertEqual(plan['engine']['speculative_arguments'], active['speculative_arguments'])

    def enable_gemma_methods(self):
        self.metadata['speculative_types_by_model_architecture'] = {
            'qwen35':['draft-simple'], 'gemma4':['draft-simple', 'draft-mtp', 'ngram-simple']}
        self.metadata['draftless_speculative_types'] = ['ngram-simple']

    def test_gemma_mtp_is_explicit_and_keeps_its_draft_file(self):
        self.enable_gemma_methods()
        draft = self.root/'gemma-mtp.gguf'
        write_gguf(draft)
        result = self.check(draft_algorithm='draft-mtp', draft_model=str(draft),
                            speculative_arguments=['--spec-draft-n-max', '6'])
        self.assertEqual(result['drafting'], {'draft_algorithm':'draft-mtp', 'draft_model':str(draft),
                         'speculative_arguments':['--spec-draft-n-max','6']})
        with self.assertRaisesRegex(SetupError, 'requires a local GGUF draft'):
            self.check(draft_algorithm='draft-mtp')

    def test_draft_file_selects_the_declared_model_specific_method(self):
        self.enable_gemma_methods()
        self.metadata['speculative_types_by_model_architecture']['gemma4'] = ['draft-mtp', 'ngram-simple']
        self.metadata['default_speculative_type_by_model_architecture'] = {'gemma4':'ngram-simple'}
        draft = self.root/'gemma-mtp.gguf'
        write_gguf(draft)
        self.assertEqual(self.check(draft_model=str(draft))['drafting']['draft_algorithm'], 'draft-mtp')

    def test_ngram_can_be_selected_without_a_draft_and_cannot_leak_into_qwen(self):
        self.enable_gemma_methods()
        result = self.check(draft_algorithm='ngram-simple',
                            speculative_arguments=['--spec-ngram-size-n', '10'])
        self.assertEqual(result['drafting'], {'draft_algorithm':'ngram-simple',
                         'speculative_arguments':['--spec-ngram-simple-size-n','10']})
        draft = self.root/'draft.gguf'
        write_gguf(draft)
        with self.assertRaisesRegex(SetupError, 'does not use an external draft'):
            self.check(draft_algorithm='ngram-simple', draft_model=str(draft))
        write_gguf(self.model, architecture='qwen35')
        with self.assertRaisesRegex(SetupError, 'unsupported.*will not be disabled'):
            self.check(draft_algorithm='ngram-simple')

    def test_default_ngram_and_active_mtp_stay_enabled_in_generated_launch(self):
        self.enable_gemma_methods()
        self.metadata['default_speculative_type_by_model_architecture'] = {'gemma4':'ngram-simple'}
        draft = self.root/'draft.gguf'
        write_gguf(draft)
        runtime = {'python':sys.executable, 'binary':'/fixture/llama-server', 'gguf_python':str(self.root)}
        for active, method in ((None,'ngram-simple'),
                ({'kind':'llamacpp', 'draft_algorithm':'draft-mtp', 'draft_model':str(draft),
                  'speculative_arguments':['--spec-draft-n-max','6']}, 'draft-mtp')):
            with self.subTest(method=method):
                with (mock.patch('owner_mode.backend_setup._kits', side_effect=lambda:iter([(self.kit,self.metadata,'llamacpp')])),
                      mock.patch('owner_mode.backend_setup.active_speculation', return_value=active),
                      mock.patch('owner_mode.backend_setup._prepare_runtime', return_value=runtime)):
                    result = prepare_backend(self.home, {**self.spec, 'port':18921 if active else 18920},
                                             self.profile, model_id='fixture/gemma')
                native = json.loads((Path(result['state'])/'native-arguments.json').read_text())
                self.assertEqual(native[native.index('--spec-type')+1], method)
                self.assertEqual(native.count('--spec-type'), 1)

    def test_active_unsupported_method_and_extra_method_override_stop(self):
        self.enable_gemma_methods()
        with mock.patch('owner_mode.backend_setup.active_speculation',
                        return_value={'kind':'llamacpp','draft_algorithm':'ngram-cache'}):
            with self.assertRaisesRegex(SetupError, 'unsupported.*will not be disabled'):
                self.check()
        with self.assertRaisesRegex(SetupError, 'cannot override'):
            self.check(draft_algorithm='ngram-simple', speculative_arguments=['--spec-type=draft-mtp'])

    def test_current_draft_model_flag_is_preserved_and_cannot_override_managed_path(self):
        self.enable_gemma_methods()
        draft = self.root/'gemma-mtp.gguf'
        write_gguf(draft)
        process = self.proc/'123'
        process.mkdir()
        (process/'cwd').symlink_to(self.root)
        argv = ['llama-server', '--model', str(self.model), '--spec-type', 'draft-mtp',
                '--spec-draft-model', str(draft), '--spec-draft-n-max', '6']
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in argv)+b'\0')
        result = self.check()['drafting']
        self.assertEqual(result['draft_model'], str(draft))
        self.assertEqual(result['speculative_arguments'], ['--spec-draft-n-max', '6'])
        (process/'cmdline').unlink()
        for arguments in ({'arguments':['--spec-draft-model', str(draft)]},
                          {'speculative_arguments':['--spec-draft-model',str(draft)]}):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(SetupError, 'cannot override'):
                self.check(**arguments)

    def test_removed_draft_count_aliases_preserve_values_for_draft_methods(self):
        for alias in ('--draft', '--draft-n', '--draft-max'):
            for option in ([alias, '007'], [alias+'=007']):
                original = ['--spec-type', 'draft-mtp', *option]
                with self.subTest(option=option):
                    normalized = normalize_engine_arguments('llamacpp', original)
                    expected = (['--spec-type','draft-mtp','--spec-draft-n-max','007']
                                if len(option) == 2 else ['--spec-type','draft-mtp','--spec-draft-n-max=007'])
                    self.assertEqual(normalized, expected)
                    self.assertEqual(original[-len(option):], option)

    def test_removed_ngram_options_require_and_preserve_the_selected_method(self):
        for method, expected in (('ngram-simple', '--spec-ngram-simple-size-n'),
                                  ('ngram-mod', '--spec-ngram-mod-n-match')):
            self.assertEqual(normalize_engine_arguments('llamacpp', ['--spec-ngram-size-n','12'],
                             llamacpp_method=method), [expected,'12'])
        self.assertEqual(normalize_engine_arguments('llamacpp', ['--draft-max=9','--draft-min','2'],
                         llamacpp_method='ngram-mod'), ['--spec-ngram-mod-n-max=9','--spec-ngram-mod-n-min','2'])
        with self.assertRaisesRegex(SetupError, 'ambiguous'):
            normalize_engine_arguments('llamacpp', ['--draft-max','9'], llamacpp_method='ngram-simple')
        with self.assertRaisesRegex(SetupError, 'ambiguous'):
            normalize_engine_arguments('llamacpp', ['--draft-max','9'])

    def test_conflicting_or_incomplete_legacy_counts_stop_before_startup(self):
        for arguments, message in ((['--draft-max','4','--spec-draft-n-max','5'], 'Conflicting'),
                                   (['--draft-max'], 'integer'),
                                   (['--spec-type','draft-mtp,ngram-mod','--draft-max','4'], 'ambiguous')):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(SetupError, message):
                normalize_engine_arguments('llamacpp', arguments, llamacpp_method=None if message == 'ambiguous' else 'draft-mtp')

    def test_active_legacy_draft_count_survives_discovery_and_explicit_plan(self):
        self.enable_gemma_methods()
        draft = self.root/'gemma-mtp.gguf'
        write_gguf(draft)
        process = self.proc/'123'
        process.mkdir()
        (process/'cwd').symlink_to(self.root)
        argv = ['llama-server', '--model', str(self.model), '--spec-type', 'draft-mtp',
                '--model-draft', str(draft), '--draft-max', '6']
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in argv)+b'\0')
        self.assertEqual(self.check()['drafting']['speculative_arguments'], ['--spec-draft-n-max','6'])
        self.assertEqual(self.check(speculative_arguments=['--draft','6'])['drafting']['speculative_arguments'],
                         ['--spec-draft-n-max','6'])
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in
            ['llama-server','--model',str(self.model),'--draft-max','6'])+b'\0')
        with self.assertRaisesRegex(SetupError, 'ambiguous'):
            self.check()


if __name__ == '__main__':
    unittest.main()
