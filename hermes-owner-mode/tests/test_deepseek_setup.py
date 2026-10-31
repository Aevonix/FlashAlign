# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""DeepSeek setup contracts using tiny synthetic files; no model is loaded."""
from contextlib import redirect_stdout
import hashlib
import hmac
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_setup import wizard, write_tensors
from owner_mode.backend_setup import (KIT_ROOT, _prepare_runtime, active_speculation,
                                      check_backend, prepare_backend, select_backend_python)
from owner_mode.profiles import (DEEPSEEK_MODEL, ProfileError, stage_profile,
                                 validate_profile)
from owner_mode.quickstart import make_default_plan
from owner_mode.model_metadata import compatible_request, configure_hermes_reasoning
from owner_mode.policy import Policy, VerifiedTurn
from owner_mode.proxy import ProviderProxy
from owner_mode.setup_runtime import SetupError

KIT = KIT_ROOT/'vllm-deepseek'
PUBLIC = KIT_ROOT.parent/'profiles/deepseek41-flash1017-v1'
spec = importlib.util.spec_from_file_location('deepseek_launch_fixture', KIT/'launch.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_profile(path):
    path.mkdir()
    rows = [
        {'layer': 0, 'site': 'attn_out', 'tensor': 'layers.0.attn.wo_b.weight', 'operator': 'text'},
        {'layer': 0, 'site': 'moe_out', 'tensor_pattern': 'layers.0.ffn.*.w2.weight',
         'components': ['routed_experts', 'shared_expert'],
         'application': 'combined_moe_output_before_mhc_residual', 'operator': 'text'},
        {'layer': 1, 'site': 'engram_value_out', 'tensor': 'layers.1.engram.wkv.weight',
         'output_slice': [20480, 25600], 'operator': 'text'},
        {'layer': 0, 'site': 'embedding_out', 'tensor': 'embed.weight', 'operator': 'text'},
        {'layer': 0, 'site': 'vision_aligner_out', 'tensor': 'aligner.w2.weight', 'bias_tensor': 'aligner.w2.bias',
         'application': 'before_unchanged_bias', 'operator': 'text'},
        {'layer': 0, 'site': 'image_marker_out', 'tensors': ['image_start', 'image_end', 'image_newline'],
         'application': 'typed_image_span_delimiters_without_aligner_bias', 'operator': 'text'}]
    doc = {'schema': 'flashalign.grouped_projection.v1', 'name': 'synthetic-native-test',
           'model': DEEPSEEK_MODEL, 'hidden_size': 5120, 'transform': 'projection',
           'requires': ['grouped_basis', 'selected_tp_rows', 'combined_moe_output',
                        'engram_value_slice', 'vision_pre_bias', 'typed_image_span'],
           'operators': {'text': {'basis': 'basis', 'rank': 1, 'scale': 1}}, 'sites': rows}
    (path/'profile.json').write_text(json.dumps(doc))
    header = json.dumps({'basis': {'dtype': 'F32', 'shape': [1, 5120],
                                   'data_offsets': [0, 5120*4]}}).encode()
    (path/'profile.safetensors').write_bytes(struct.pack('<Q', len(header))+header+
                                            struct.pack('<5120f', 1., *([0.]*5119)))
    return doc


class GroupedProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'profile'
        self.doc = group_profile(self.path)

    def changed(self):
        (self.path/'profile.json').write_text(json.dumps(self.doc))
        return validate_profile(self.path)

    def test_typed_group_validates_and_stages_a_local_checksum(self):
        checked = self.changed()
        self.assertEqual(checked['schema'], 'flashalign.grouped_projection.v1')
        staged = self.root/'staged'
        stage_profile(self.path, staged)
        self.assertEqual(validate_profile(staged)['hashes'], checked['hashes'])
        self.assertEqual(json.loads((staged/'sha256.json').read_text()), checked['hashes'])

    def test_unsupported_schema_fails(self):
        self.doc['schema'] = 'unknown'
        with self.assertRaisesRegex(ProfileError, 'schema'): self.changed()

    def test_model_revision_is_exact(self):
        self.doc['model'] = {**DEEPSEEK_MODEL, 'revision': '0'*40}
        with self.assertRaisesRegex(ProfileError, 'pinned'): self.changed()

    def test_required_feature_cannot_be_silently_dropped(self):
        self.doc['requires'].remove('typed_image_span')
        with self.assertRaisesRegex(ProfileError, 'requirements'): self.changed()

    def test_duplicate_site_is_rejected(self):
        self.doc['sites'].append(self.doc['sites'][0])
        with self.assertRaisesRegex(ProfileError, 'Duplicate'): self.changed()

    def test_engram_key_rows_cannot_be_transformed(self):
        self.doc['sites'][2]['output_slice'] = [0, 5120]
        with self.assertRaisesRegex(ProfileError, 'native output'): self.changed()

    def test_vision_bias_contract_cannot_be_changed(self):
        self.doc['sites'][4]['application'] = 'after_bias'
        with self.assertRaisesRegex(ProfileError, 'native output'): self.changed()

    def test_unused_operator_is_rejected(self):
        self.doc['operators']['unused'] = dict(self.doc['operators']['text'])
        with self.assertRaisesRegex(ProfileError, 'unused'): self.changed()

    def test_every_declared_basis_is_checked_even_without_Q_suffix(self):
        payload = self.path/'profile.safetensors'
        raw = bytearray(payload.read_bytes())
        raw[-5120*4:-5120*4+4] = struct.pack('<f', 2.)
        payload.write_bytes(raw)
        with self.assertRaisesRegex(ProfileError, 'orthonormal'): self.changed()

    def test_public_85_site_profile_remains_byte_identical(self):
        before = {p.name: digest(p) for p in PUBLIC.iterdir() if p.is_file()}
        checked = validate_profile(PUBLIC)
        staged = self.root/'standard'
        stage_profile(PUBLIC, staged)
        self.assertEqual(checked['name'], 'deepseek41-flash1017-v1')
        self.assertEqual(len(json.loads((staged/'profile.json').read_text())['sites']), 85)
        self.assertEqual(before, {p.name: digest(p) for p in PUBLIC.iterdir() if p.is_file()})
        self.assertEqual(digest(staged/'profile.safetensors'), before['profile.safetensors'])
        self.assertEqual(digest(staged/'profile.json'), before['profile.json'])


class DeepSeekPlanning(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root/'home';self.home.mkdir()
        self.model = self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type': 'deepseek_v41', 'hidden_size': 5120,
                                                        'num_hidden_layers': 40, 'quantization_config': {'quant_method': 'fp8'}}))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        self.metadata = json.loads((KIT/'backend.json').read_text())
        # Only fixture fingerprints change. Production metadata remains exact and immutable.
        entry = self.metadata['compatible_models'][0]
        entry.update(config_sha256=digest(self.model/'config.json'),
                     index_sha256=digest(self.model/'model.safetensors.index.json'),
                     quantization_config_sha256=hashlib.sha256(b'{"quant_method":"fp8"}').hexdigest())
        self.spec = {'kind': 'vllm', 'model_path': str(self.model), 'devices': ['0', '1', '2', '3']}
        self.profile = validate_profile(PUBLIC)

    def check(self, spec=None, profile=None):
        with mock.patch('owner_mode.backend_setup._kits', return_value=iter([(KIT, self.metadata, 'vllm')])):
            return check_backend(self.home, spec or self.spec, profile or self.profile,
                                 model_id=DEEPSEEK_MODEL['id'], proc_root=self.root/'no-processes')

    def test_primary_checkpoint_remains_valid_alongside_native_aliases(self):
        self.assertTrue(self.profile['compatible_models'])
        self.assertTrue(self.check())
        for field,value in [('id','other/model'),('revision','0'*40),('config_sha256','0'*64),
                            ('index_sha256','0'*64),('quantization_config_sha256','0'*64)]:
            with self.subTest(field=field):
                changed={**self.profile,'model':{**self.profile['model'],field:value}}
                with self.assertRaises(SetupError):
                    self.check(profile=changed)

    def test_public_profile_and_native_draft_selected_without_weight_load(self):
        result = self.check()
        self.assertEqual(result['drafting']['speculative_config'],
                         {'method': 'dspark', 'model': str(self.model), 'num_speculative_tokens': 5})
        self.assertEqual(list(self.home.iterdir()), [])

    def test_grouped_operator_supplied_profile_is_accepted(self):
        path = self.root/'profile';group_profile(path)
        self.assertEqual(self.check(profile=validate_profile(path))['metadata']['variant'], 'deepseek-v41')

    def test_wrong_checkpoint_fingerprint_fails(self):
        (self.model/'config.json').write_text((self.model/'config.json').read_text()+' ')
        with self.assertRaisesRegex(SetupError, 'fingerprints'): self.check()

    def test_mismatched_standard_site_layer_fails(self):
        path = self.root/'profile';path.mkdir()
        doc = {'name': 'synthetic-standard', 'model': DEEPSEEK_MODEL, 'hidden_size': 5120,
               'sites': [{'layer': 0, 'site': 'attn_out', 'rank': 1, 'scale': 1,
                          'tensor': 'layers.0.attn.wo_b.weight'}]}
        (path/'profile.json').write_text(json.dumps(doc))
        write_tensors(path, {'L0.attn_out.Q': ([1, 5120], [1., *([0.]*5119)])})
        self.check(profile=validate_profile(path))
        doc['sites'][0]['tensor'] = 'layers.3.attn.wo_b.weight'
        (path/'profile.json').write_text(json.dumps(doc))
        with self.assertRaisesRegex(SetupError, 'target tensor'): self.check(profile=validate_profile(path))

    def test_duplicate_or_missing_gpu_fails(self):
        for devices in (['0'], ['0', '0', '1', '2']):
            with self.assertRaisesRegex(SetupError, 'GPU count'): self.check({**self.spec, 'devices': devices})

    def test_ngram_cannot_replace_native_dspark(self):
        with self.assertRaisesRegex(SetupError, 'DSpark5'):
            self.check({**self.spec, 'speculative_config': {'method': 'ngram'}})

    def test_native_process_discovery_preserves_exact_speculation(self):
        proc = self.root/'proc';process = proc/'123';process.mkdir(parents=True)
        config = {'method': 'dspark', 'model': str(self.model), 'num_speculative_tokens': 5}
        command = [sys.executable, '-m', 'flashalign_deepseek', '--model', str(self.model),
                   '--speculative-config', json.dumps(config)]
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in command)+b'\0')
        (process/'cwd').symlink_to(self.root)
        self.assertEqual(active_speculation(self.model, DEEPSEEK_MODEL['id'], proc)['speculative_config'], config)
        (process/'cmdline').write_bytes(b'\0'.join(value.encode() for value in command[:-2])+b'\0')
        with self.assertRaisesRegex(SetupError, 'could not be read'):
            active_speculation(self.model, DEEPSEEK_MODEL['id'], proc)

    def test_external_draft_cannot_replace_native_stock(self):
        with self.assertRaisesRegex(SetupError, 'external'):
            self.check({**self.spec, 'draft_model': str(self.root/'other')})

    def test_resource_override_is_explicit_typed_field(self):
        self.assertTrue(self.check({**self.spec, 'allow_untested_memory': True}))
        with self.assertRaisesRegex(SetupError, 'boolean'):
            self.check({**self.spec, 'allow_untested_memory': 'yes'})
        with self.assertRaisesRegex(SetupError, 'managed plan'):
            self.check({**self.spec, 'arguments': ['--allow-untested-memory']})

    def test_guarded_vllm_underscore_aliases_require_managed_fields(self):
        for arguments in (['--allow_untested_memory'], ['--replay_file=/tmp/other'],
                          ['--model_receipt', '/tmp/other'], ['--no_enforce_eager'],
                          ['--hf_overrides={"model_type":"other"}'],
                          ['--hf_overrides.model_type=other']):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(SetupError, 'managed plan'):
                    self.check({**self.spec, 'arguments': arguments})
                self.assertEqual(list(self.home.iterdir()), [])

    def test_default_one_command_plan_needs_no_private_download_or_draft_prompt(self):
        operator_python = '/operator/python3.12'
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=None, engine_bundle=None,
                               backend='auto', profiles=[], engine_python=operator_python,
                               devices='0,1,2,3', tools_default=None, engine_only=True)
        with (mock.patch('owner_mode.quickstart.download_engine_bundle', side_effect=AssertionError('network')),
              mock.patch('owner_mode.backend_setup._kits', side_effect=lambda: iter([(KIT, self.metadata, 'vllm')])),
              mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 12, 9)),
              mock.patch('builtins.input', side_effect=AssertionError('unexpected prompt')),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              redirect_stdout(io.StringIO())):
            path = make_default_plan(args, {}, wizard.Console(color=False))
        plan = json.loads(path.read_text())
        self.assertEqual(plan['engine']['kind'], 'vllm')
        self.assertEqual(plan['engine']['python'], operator_python)
        self.assertEqual(plan['profile'], str(PUBLIC.resolve()))
        self.assertEqual(plan['engine']['speculative_config']['method'], 'dspark')
        self.assertNotIn('draft_model', plan['engine'])

    def test_incompatible_explicit_python_fails_before_any_staging(self):
        self.spec['python'] = '/operator/python3.10'
        with (mock.patch('owner_mode.backend_setup._kits', return_value=iter([(KIT, self.metadata, 'vllm')])),
              mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 10, 16))):
            with self.assertRaisesRegex(SetupError, 'Python 3.12.*selected engine.python'):
                prepare_backend(self.home, self.spec, PUBLIC, model_id=DEEPSEEK_MODEL['id'])
        self.assertEqual(list(self.home.iterdir()), [])

    def test_default_plan_rejects_old_python_before_writing_config(self):
        args = SimpleNamespace(home=self.home, model_path=self.model, profile=None, engine_bundle=None,
                               backend='auto', profiles=[], engine_python='/operator/python3.10',
                               devices='0,1,2,3', tools_default=None, engine_only=True)
        with (mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 10, 16)),
              mock.patch('owner_mode.backend_setup._kits', side_effect=lambda: iter([(KIT, self.metadata, 'vllm')])),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(SetupError, 'Python 3.12'):
                make_default_plan(args, {}, wizard.Console(color=False))
        self.assertEqual(list(self.home.iterdir()), [])

    def test_existing_large_hermes_context_stops_before_engine_preparation(self):
        config = self.home/'.hermes/config.yaml';config.parent.mkdir()
        original = json.dumps({'model': {'default': DEEPSEEK_MODEL['id'],
                                         'context_length': 737280, 'max_tokens': 32000},
                               'agent': {'reasoning_effort': 'low'}})
        config.write_text(original)
        found = wizard.discover(self.home, which=lambda _: None)
        plan = self.root/'plan.json'
        plan.write_text(json.dumps({'harnesses': ['hermes'], 'model': DEEPSEEK_MODEL['id'],
            'profile': str(PUBLIC), 'contacts': [{'channel': 'sms', 'account': 'default', 'contact': '+15550000101'}],
            'tools_default': False, 'engine': self.spec}))
        plan.chmod(0o600)
        with (mock.patch.object(wizard, 'discover', return_value=found),
              mock.patch('owner_mode.backend_setup._kits', return_value=iter([(KIT, self.metadata, 'vllm')])),
              mock.patch('owner_mode.automation.prepare_engine') as prepare,
              redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(SetupError, '737280.*65536'):
                wizard.main(['--home', str(self.home), '--config', str(plan), '--apply'])
        prepare.assert_not_called()
        self.assertEqual(config.read_text(), original)
        self.assertFalse((self.home/'.config').exists())


class EnginePythonSelection(unittest.TestCase):
    metadata = {'python_min_version': [3, 12]}

    def test_implicit_old_python_selects_existing_compatible_interpreter(self):
        versions = {'/system/python': (3, 10, 16), '/local/python3.12': (3, 12, 10)}
        with (mock.patch('owner_mode.backend_setup.sys._base_executable', '/system/python'),
              mock.patch('owner_mode.backend_setup.sys.executable', '/system/python'),
              mock.patch('owner_mode.backend_setup.shutil.which', side_effect=lambda name: '/local/'+name if name == 'python3.12' else None),
              mock.patch('owner_mode.backend_setup._python_version', side_effect=versions.get)):
            result = select_backend_python({}, self.metadata)
        self.assertEqual(result['executable'], '/local/python3.12')
        self.assertTrue(result['automatic'])
        self.assertEqual(result['version'], [3, 12, 10])

    def test_explicit_compatible_venv_path_is_preserved(self):
        with mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 12, 9)):
            result = select_backend_python({'python': '/operator/venv/bin/python'}, self.metadata)
        self.assertEqual(result['executable'], '/operator/venv/bin/python')
        self.assertFalse(result['automatic'])

    def test_no_compatible_interpreter_reports_action_without_installing(self):
        with (mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 10, 16)),
              mock.patch('owner_mode.backend_setup.shutil.which', return_value=None)):
            with self.assertRaisesRegex(SetupError, 'install a compatible local interpreter'):
                select_backend_python({}, self.metadata)

    def test_explicit_choice_is_not_silently_replaced(self):
        with (mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 10, 16)),
              mock.patch('owner_mode.backend_setup.shutil.which') as which):
            with self.assertRaisesRegex(SetupError, 'selected engine.python'):
                select_backend_python({'python': '/operator/python3.10'}, self.metadata)
        which.assert_not_called()

    def test_runtime_rechecks_before_install_or_log_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            with (mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 10, 16)),
                  mock.patch('owner_mode.backend_setup._run') as run):
                with self.assertRaisesRegex(SetupError, 'Python 3.12'):
                    _prepare_runtime('vllm', {'python': '/operator/python3.10'}, KIT, state)
            run.assert_not_called()
            self.assertEqual(list(state.iterdir()), [])


class DeepSeekResources(unittest.TestCase):
    def setUp(self):
        self.metadata = json.loads((KIT/'backend.json').read_text())
        self.devices = [{'index': str(i), 'capability': '12.0', 'total_bytes': 95*launcher.GIB,
                         'free_bytes': 94*launcher.GIB} for i in range(4)]
        self.memory = {'host_available_bytes': 370*launcher.GIB, 'cgroup_available_bytes': 400*launcher.GIB,
                       'effective_available_bytes': 370*launcher.GIB}

    def test_tested_hardware_policy_passes(self):
        self.assertFalse(launcher.check_resources(self.metadata, devices=self.devices,
                                                 memory=self.memory)['below_startup_policy'])

    def test_hardware_and_cgroup_limits_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'four'):
            launcher.check_resources(self.metadata, devices=self.devices[:3], memory=self.memory)
        self.devices[0]['capability'] = '9.0'
        with self.assertRaisesRegex(ValueError, 'SM120'):
            launcher.check_resources(self.metadata, devices=self.devices, memory=self.memory)

    def test_lower_memory_needs_explicit_experimental_override(self):
        self.memory['effective_available_bytes'] = 250*launcher.GIB
        with self.assertRaisesRegex(ValueError, 'unqualified'):
            launcher.check_resources(self.metadata, devices=self.devices, memory=self.memory)
        result = launcher.check_resources(self.metadata, devices=self.devices, memory=self.memory,
                                          allow_untested_memory=True)
        self.assertTrue(result['below_startup_policy'])

    def test_runtime_floor_cannot_be_overridden(self):
        self.memory['effective_available_bytes'] = 63*launcher.GIB
        with self.assertRaisesRegex(ValueError, 'non-overridable'):
            launcher.check_resources(self.metadata, devices=self.devices, memory=self.memory,
                                     allow_untested_memory=True)

    def test_launch_keeps_native_dspark_and_explicit_replay(self):
        args = SimpleNamespace(model=Path('/models/stock'), model_receipt=Path('/state/hash.json'),
                               key_file=Path('/state/key'), replay_file=Path('/state/replay.sqlite3'),
                               profile=[Path('/state/profile')], host='127.0.0.1', port=8041,
                               speculative_config=None)
        command = launcher.build_command(args, [], self.metadata)
        self.assertEqual(command[command.index('--max-model-len')+1], '65536')
        self.assertIn('--replay-file', command)
        self.assertEqual(json.loads(command[command.index('--speculative-config')+1])['method'], 'dspark')
        with self.assertRaisesRegex(ValueError, 'managed'):
            launcher.build_command(args, ['--dry-run'], self.metadata)
        args.speculative_config = '{"method":"ngram"}'
        with self.assertRaisesRegex(ValueError, 'DSpark5'):
            launcher.build_command(args, [], self.metadata)

    def test_parent_cgroup_limit_is_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);proc = root/'proc';cg = root/'cgroup'
            (proc/'self').mkdir(parents=True);(cg/'parent/child').mkdir(parents=True)
            (proc/'meminfo').write_text('MemAvailable: 400000000 kB\n')
            (proc/'self/cgroup').write_text('0::/parent/child\n')
            (cg/'parent/child/memory.max').write_text('max\n')
            (cg/'parent/child/memory.current').write_text('0\n')
            (cg/'parent/memory.max').write_text(str(200*launcher.GIB))
            (cg/'parent/memory.current').write_text(str(50*launcher.GIB))
            result = launcher.memory_headroom(proc, cg)
            self.assertEqual(result['effective_available_bytes'], 150*launcher.GIB)

    def test_unsupported_cgroup_layout_does_not_silently_use_host_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);proc = root/'proc';(proc/'self').mkdir(parents=True)
            (proc/'meminfo').write_text('MemAvailable: 400000000 kB\n')
            (proc/'self/cgroup').write_text('3:memory:/fixture\n')
            with self.assertRaisesRegex(ValueError, 'cgroup v2'):
                launcher.memory_headroom(proc, root/'cg')

    def test_inactive_checkpoint_cache_has_bounded_working_set_credit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);proc = root/'proc';cg = root/'cgroup'
            (proc/'self').mkdir(parents=True);cg.mkdir()
            (proc/'meminfo').write_text('MemAvailable: 400000000 kB\n')
            (proc/'self/cgroup').write_text('0::/\n')
            (cg/'memory.max').write_text(str(512*launcher.GIB))
            (cg/'memory.current').write_text(str(480*launcher.GIB))
            fields = {'file': 350, 'shmem': 100, 'inactive_file': 300,
                      'unevictable': 10, 'file_dirty': 20, 'file_writeback': 10}
            (cg/'memory.stat').write_text('\n'.join(f'{key} {value*launcher.GIB}' for key, value in fields.items()))
            result = launcher.memory_headroom(proc, cg)
            row = result['cgroups'][0]
            self.assertEqual(row['raw_available_bytes'], 32*launcher.GIB)
            self.assertEqual(row['ordinary_inactive_file_credit_bytes'], 210*launcher.GIB)
            self.assertEqual(result['effective_available_bytes'], 242*launcher.GIB)
            self.assertEqual(row['shmem_bytes'], 100*launcher.GIB)
            self.assertEqual(row['unevictable_bytes'], 10*launcher.GIB)

    def test_shmem_and_pinned_pages_never_receive_file_cache_credit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);proc = root/'proc';cg = root/'cgroup'
            (proc/'self').mkdir(parents=True);cg.mkdir()
            (proc/'meminfo').write_text('MemAvailable: 400000000 kB\n')
            (proc/'self/cgroup').write_text('0::/\n')
            (cg/'memory.max').write_text(str(512*launcher.GIB))
            (cg/'memory.current').write_text(str(480*launcher.GIB))
            (cg/'memory.stat').write_text(f'file {320*launcher.GIB}\nshmem {300*launcher.GIB}\ninactive_file {300*launcher.GIB}\nunevictable {20*launcher.GIB}\n')
            result = launcher.memory_headroom(proc, cg)
            self.assertEqual(result['cgroups'][0]['ordinary_inactive_file_credit_bytes'], 0)
            self.assertEqual(result['effective_available_bytes'], 32*launcher.GIB)

    def test_supervisor_stops_its_actual_child_at_memory_floor(self):
        with mock.patch.object(launcher, 'memory_headroom', return_value={'effective_available_bytes': 0}):
            result = launcher.supervise([sys.executable, '-c', 'import time; time.sleep(30)'], 1)
        self.assertEqual(result, 1)

    def test_supervisor_retains_successful_child_exit(self):
        with mock.patch.object(launcher, 'memory_headroom', return_value={'effective_available_bytes': 10}):
            self.assertEqual(launcher.supervise([sys.executable, '-c', 'pass'], 1), 0)

    def test_stale_root_identity_never_signals_a_reused_process(self):
        with (mock.patch.object(launcher, '_process_start', return_value='new-process'),
              mock.patch.object(launcher.os, 'kill') as kill):
            self.assertFalse(launcher._signal_tree(SimpleNamespace(pid=12345), signal.SIGTERM, 'original-process'))
            kill.assert_not_called()

    def test_missing_supervisor_identity_refuses_to_start_a_child(self):
        with (mock.patch.object(launcher.subprocess, 'Popen') as start,
              mock.patch.object(launcher, '_process_start', return_value=None)):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                launcher.supervise([sys.executable, '-c', 'import time; time.sleep(30)'], 1)
        start.assert_not_called()


class TinyFilePreparation(unittest.TestCase):
    """Exercise real file hashing/staging; dependency and GPU startup are excluded."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root/'home'; self.home.mkdir()
        self.model = self.root/'model'; self.model.mkdir()
        config = {'model_type': 'deepseek_v41', 'hidden_size': 5120,
                  'num_hidden_layers': 40, 'quantization_config': {'quant_method': 'fp8'}}
        (self.model/'config.json').write_text(json.dumps(config))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{"fixture":"part.safetensors"}}')
        (self.model/'part.safetensors').write_bytes(b'tiny synthetic checkpoint, not a real model')
        self.kit = self.root/'synthetic-kit'
        shutil.copytree(KIT, self.kit, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        self.metadata = json.loads((self.kit/'backend.json').read_text())
        self.metadata['compatible_models'][0].update(
            config_sha256=digest(self.model/'config.json'),
            index_sha256=digest(self.model/'model.safetensors.index.json'),
            quantization_config_sha256=hashlib.sha256(b'{"quant_method":"fp8"}').hexdigest())
        (self.kit/'backend.json').write_text(json.dumps(self.metadata))
        pins = {'model': DEEPSEEK_MODEL['id'], 'revision': DEEPSEEK_MODEL['revision'],
                'metadata_sha256': {name: digest(self.model/name) for name in ('config.json', 'model.safetensors.index.json')},
                'shards': {'part.safetensors': {'bytes': (self.model/'part.safetensors').stat().st_size,
                                               'sha256': digest(self.model/'part.safetensors')}}}
        (self.kit/'flashalign_deepseek/model-pins.json').write_text(json.dumps(pins))
        for path in self.model.iterdir(): path.chmod(0o444)
        self.original = {path.name: digest(path) for path in self.model.iterdir()}
        self.engine = {'kind': 'vllm', 'model_path': str(self.model), 'devices': ['0', '1', '2', '3']}

    def prepare(self, provided_pythonpath=None):
        runtime = {'python': sys.executable}
        if provided_pythonpath:
            runtime['provided_pythonpath'] = str(provided_pythonpath)
        with (mock.patch('owner_mode.backend_setup._kits', return_value=iter([(self.kit, self.metadata, 'vllm')])),
              mock.patch('owner_mode.backend_setup.active_speculation', return_value=None),
              mock.patch('owner_mode.backend_setup._prepare_runtime', return_value=runtime)):
            return prepare_backend(self.home, self.engine, PUBLIC, model_id=DEEPSEEK_MODEL['id'])

    def test_real_tiny_payload_verification_prepares_without_sidecars_or_model_writes(self):
        result = self.prepare()
        receipt = Path(result['model_receipt'])
        self.assertTrue(json.loads(receipt.read_text())['complete'])
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
        self.assertTrue(receipt.is_relative_to(self.home))
        self.assertEqual(self.original, {path.name: digest(path) for path in self.model.iterdir()})
        command = result['commands'][0]['argv']
        self.assertIn('--model-receipt', command)
        self.assertIn('--replay-file', command)
        self.assertFalse(Path(result['replay_file']).exists())
        self.assertFalse((Path(result['state'])/'running.json').exists())
        self.assertNotIn('draft_model', result['drafting'])
        self.assertEqual(result['hardware_activation_check'], 'pending')

    def test_repeat_verification_reuses_key_and_resumable_receipt(self):
        first = self.prepare()
        key = Path(first['key_file']).read_bytes()
        again = self.prepare()
        self.assertEqual(first['state'], again['state'])
        self.assertEqual(Path(again['key_file']).read_bytes(), key)
        self.assertTrue(json.loads(Path(again['model_receipt']).read_text())['complete'])

    def test_tampered_payload_prevents_new_preparation(self):
        result = self.prepare()
        shard = self.model/'part.safetensors'
        shard.chmod(0o644)
        shard.write_bytes(b'x'*shard.stat().st_size)
        with self.assertRaisesRegex(SetupError, 'preparation failed'):
            self.prepare()
        self.assertFalse((Path(result['state'])/'running.json').exists())

    def test_staged_native_module_precedes_an_old_provided_plugin(self):
        old = self.root/'old';(old/'flashalign_deepseek').mkdir(parents=True)
        (old/'flashalign_deepseek/__init__.py').write_text("raise AssertionError('stale plugin imported')\n")
        result = self.prepare(old)
        environment = {**os.environ, **result['commands'][0]['env']}
        self.assertEqual(environment['PYTHONPATH'].split(os.pathsep)[0], result['source'])
        command = [sys.executable, '-c', 'from pathlib import Path; import flashalign_deepseek; '
                   'assert Path(flashalign_deepseek.__file__).resolve().is_relative_to(Path('+repr(result['source'])+').resolve())']
        subprocess.run(command, env=environment, cwd=self.root, check=True, capture_output=True)


class HermesModelReasoning(unittest.TestCase):
    def test_incompatible_default_becomes_a_model_specific_override(self):
        config = {'agent': {'reasoning_effort': 'medium',
                            'reasoning_overrides': {'some-other-model': 'max'}}}
        result = configure_hermes_reasoning(config, DEEPSEEK_MODEL['id'])
        self.assertEqual(result['status'], 'adjusted')
        self.assertEqual(config['agent']['reasoning_effort'], 'medium')
        self.assertEqual(config['agent']['reasoning_overrides'],
                         {'some-other-model': 'max', DEEPSEEK_MODEL['id']: 'low'})

    def test_supported_user_choice_is_preserved(self):
        for effort in ['low', 'high', 'xhigh', 'max']:
            config = {'agent': {'reasoning_effort': 'medium',
                                'reasoning_overrides': {DEEPSEEK_MODEL['id']: effort}}}
            before = json.dumps(config)
            self.assertEqual(configure_hermes_reasoning(config, DEEPSEEK_MODEL['id'])['status'], 'preserved')
            self.assertEqual(json.dumps(config), before)

    def test_unrelated_models_are_unchanged(self):
        config = {'agent': {'reasoning_effort': 'medium'}}
        self.assertIsNone(configure_hermes_reasoning(config, 'Qwen/Qwen3.8-27B'))
        self.assertEqual(config, {'agent': {'reasoning_effort': 'medium'}})

    def test_actual_configuration_writer_reports_the_model_adjustment(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary);path = home/'.hermes/config.yaml';path.parent.mkdir()
            path.write_text(json.dumps({'agent': {'reasoning_effort': 'medium'},
                                        'model': {'default': 'previous-model'}}))
            result = wizard.provision(home, ['hermes'], 'http://127.0.0.1:12345/v1',
                                      DEEPSEEK_MODEL['id'], 'fixture-profile', [], False)
            config = json.loads(path.read_text())
            self.assertEqual(config['agent']['reasoning_overrides'][DEEPSEEK_MODEL['id']], 'low')
            self.assertEqual(config['agent']['reasoning_effort'], 'medium')
            self.assertEqual(result['hermes_reasoning']['status'], 'adjusted')


class ExistingNativeEnvironment(unittest.TestCase):
    def test_matching_operator_environment_uses_no_dependency_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            calls = []
            def run(command, log, **kwargs):
                calls.append((command, kwargs))
            environment = {'PYTHONPATH': '/operator/flashinfer-overlay'}
            # This test supplies a simulated engine environment independently
            # of the Python version running the CPU-only test suite.
            operator_python = '/operator/python3.12'
            with (mock.patch('owner_mode.backend_setup._run', side_effect=run),
                  mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 12, 9)),
                  mock.patch('owner_mode.backend_setup.prepare_python_headers', return_value={})):
                runtime = _prepare_runtime('vllm', {'python': operator_python, 'install_dependencies': False,
                                                   'environment': environment}, KIT, state)
            self.assertEqual(runtime['python'], operator_python)
            self.assertEqual(runtime['provided_pythonpath'], environment['PYTHONPATH'])
            self.assertFalse(any('pip' in call[0] for call in calls))
            self.assertEqual(calls[0][1]['env']['PYTHONPATH'], environment['PYTHONPATH'])
            self.assertEqual(calls[-1][1]['env']['PYTHONPATH'], environment['PYTHONPATH'])
            self.assertIn('flashinfer-python', calls[-1][0][-1])
            self.assertIn('0.7.1rc5', calls[-1][0][-1])
            self.assertFalse((state/'runtime-overlay').exists())

    def test_failed_overlay_version_probe_stops_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            def run(command, log, **kwargs):
                if 'flashinfer-python' in command[-1]:
                    raise SetupError('Engine preparation failed')
            with (mock.patch('owner_mode.backend_setup._run', side_effect=run),
                  mock.patch('owner_mode.backend_setup._python_version', return_value=(3, 12, 9)),
                  mock.patch('owner_mode.backend_setup.prepare_python_headers', return_value={})):
                with self.assertRaisesRegex(SetupError, 'Engine preparation failed'):
                    _prepare_runtime('vllm', {'python': '/operator/python3.12'}, KIT, Path(temporary))


class ReasoningTransport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.key = Path(self.tmp.name)/'test-key'
        self.key.write_bytes(b'k'*32);self.key.chmod(0o600)
        class Authority:
            def verify_turn(self, session_id, turn_id, user_message, **_):
                return VerifiedTurn(session_id, turn_id, 'sms', user_message)
        self.policy = Policy(authority=Authority(), allowed_channels=['sms'])
        self.policy.begin(session_id='session', turn_id='turn', user_message='Hello', platform='sms')
        self.body = {'model': DEEPSEEK_MODEL['id'], 'messages': [{'role': 'user', 'content': 'Hello'}],
                     'reasoning_effort': 'none', 'chat_template_kwargs': {'reasoning_effort': 'low'}}
        self.proxy = ProviderProxy(self.policy, {'head': {'upstream': 'http://127.0.0.1:1/v1',
            'profile': 'fixture', 'request_compatibility': 'deepseek_v41'}}, self.key)
        self.addCleanup(self.proxy.close)

    def ticket(self):
        return self.policy.issue(self.body, session_id='session', turn_id='turn', profile='fixture', route='head')

    def test_owner_capability_is_checked_before_translation_and_final_bytes_are_signed(self):
        token = self.ticket();self.assertTrue(token)
        raw, headers, mode = self.proxy.prepare('head', self.body, token)
        self.assertEqual(mode, 'fixture')
        final = json.loads(raw)
        self.assertNotIn('reasoning_effort', final)
        self.assertIs(final['chat_template_kwargs']['enable_thinking'], False)
        self.assertEqual(final['messages'], self.body['messages'])
        self.assertEqual(final['chat_template_kwargs']['reasoning_effort'], 'low')
        _, stamp, nonce, signature = headers['X-Owner-Mode-Auth'].split(':')
        signed = stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
        self.assertTrue(hmac.compare_digest(signature, hmac.new(b'k'*32, signed.encode(), hashlib.sha256).hexdigest()))
        _, _, replay_mode = self.proxy.prepare('head', self.body, token)
        self.assertEqual(replay_mode, 'off')

    def test_native_nested_highest_efforts_keep_owner_authorization(self):
        for effort in ('xhigh', 'max'):
            with self.subTest(effort=effort):
                body = {'model': DEEPSEEK_MODEL['id'],
                        'messages': [{'role': 'user', 'content': 'Hello'}],
                        'reasoning_effort': effort,
                        'chat_template_kwargs': {'reasoning_effort': effort}}
                token, reason = self.policy.decide(body, session_id='session', turn_id='turn',
                                                   profile='fixture', route='head')
                self.assertTrue(token, reason)
                raw, headers, mode = self.proxy.prepare('head', body, token)
                self.assertEqual(mode, 'fixture')
                self.assertEqual(json.loads(raw)['chat_template_kwargs']['reasoning_effort'], effort)
                _, stamp, nonce, signature = headers['X-Owner-Mode-Auth'].split(':')
                signed = stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()
                self.assertTrue(hmac.compare_digest(signature, hmac.new(b'k'*32, signed.encode(), hashlib.sha256).hexdigest()))
                self.assertEqual(self.proxy.prepare('head', body, token)[2], 'off')

    def test_altered_rendered_request_cannot_borrow_original_capability(self):
        token = self.ticket()
        changed = {**self.body, 'reasoning_effort': 'low'}
        _, headers, mode = self.proxy.prepare('head', changed, token)
        self.assertEqual(mode, 'off')
        self.assertNotIn('X-Owner-Mode-Auth', headers)

    def test_normal_requests_preserve_disabled_intent_without_a_signature(self):
        raw, headers, mode = self.proxy.prepare('head', self.body)
        self.assertEqual(mode, 'off')
        self.assertNotIn('X-Owner-Mode-Auth', headers)
        self.assertNotIn('reasoning_effort', json.loads(raw))
        self.assertIs(json.loads(raw)['chat_template_kwargs']['enable_thinking'], False)
        self.assertEqual(self.body['reasoning_effort'], 'none')

    def test_other_models_routes_and_valid_efforts_are_unchanged(self):
        self.assertIs(compatible_request(self.body), self.body)
        other = {**self.body, 'model': 'other'}
        self.assertIs(compatible_request(other, 'deepseek_v41'), other)
        for effort in ['low', 'high', 'xhigh', 'max', 1, 100]:
            body = {**self.body, 'reasoning_effort': effort}
            self.assertIs(compatible_request(body, 'deepseek_v41'), body)


if __name__ == '__main__':
    unittest.main()
