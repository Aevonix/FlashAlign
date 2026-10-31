# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Keep active speculative settings and reject incompatible managed presets."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile
from test_engine_setup import make_bundle
from owner_mode.drafting import active_drafting, draft_settings, MissingDraftModel
from owner_mode.engine_setup import check_bundle, prepare_engine
from owner_mode.profiles import validate_profile
from owner_mode.setup_runtime import SetupError


class DraftPreservation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type': 'glm_moe_dsa', 'hidden_size': 2}))
        self.draft = self.root/'draft';self.draft.mkdir()
        (self.draft/'config.json').write_text('{"speculators_model_type":"dspark"}')
        self.proc = self.root/'proc';self.proc.mkdir()
        self.spec = {'model_path': str(self.model)}
        self.manifest = {'requires_draft_model': True}

    def process(self, *options, model=None, pid='100'):
        path = self.proc/pid;path.mkdir()
        argv = ['python', '-m', 'tensorfold', 'serve', str(model or self.model), *options]
        (path/'cmdline').write_bytes('\0'.join(argv).encode()+b'\0')
        (path/'cwd').symlink_to(self.root, target_is_directory=True)

    def test_active_drafter_parallelism_and_depth_are_preserved(self):
        self.process('--drafter', str(self.draft), '--parallel', '4', '--mtp-drafts', '6')
        settings = draft_settings(self.root, self.spec, self.manifest, 'bench', proc_root=self.proc)
        self.assertEqual(settings, {'draft_model': str(self.draft), 'parallel': 4, 'mtp_drafts': 6})

    def test_active_mtp_cannot_be_silently_replaced_or_disabled(self):
        self.process('--parallel', '1')
        with self.assertRaisesRegex(SetupError, 'will not be disabled or replaced'):
            draft_settings(self.root, {**self.spec, 'draft_model': str(self.draft)}, self.manifest,
                           'bench', proc_root=self.proc)

    def test_missing_draft_stops_and_unrelated_process_is_ignored(self):
        self.process('--drafter', str(self.draft), '--parallel', '4', model=self.root/'other-model')
        self.assertIsNone(active_drafting(self.model, 'bench', self.proc))
        with mock.patch.dict(os.environ, {'HF_HOME': str(self.root/'empty-cache')}):
            with self.assertRaises(MissingDraftModel):
                draft_settings(self.root, self.spec, self.manifest, 'bench', proc_root=self.proc)

    def test_cached_draft_is_detected_without_disabling_speculation(self):
        directory = self.root/'.cache/huggingface/hub/models--example--draft/snapshots/rev'
        directory.mkdir(parents=True)
        (directory/'config.json').write_text('{"speculators_model_type":"dspark"}')
        with mock.patch.dict(os.environ, {'HF_HOME': str(self.root/'.cache/huggingface')}):
            self.assertEqual(draft_settings(self.root, self.spec, self.manifest, 'bench', proc_root=self.proc),
                             {'draft_model': str(directory)})

    def test_active_draft_override_is_rejected(self):
        self.process('--drafter', str(self.draft), '--parallel', '4')
        with self.assertRaisesRegex(SetupError, 'differs from the active engine'):
            draft_settings(self.root, {**self.spec, 'draft_model': str(self.root/'another')},
                           self.manifest, 'bench', proc_root=self.proc)

    def test_prepared_commands_keep_speculation_and_reject_disabling_bundle(self):
        bundle = make_bundle(self.root/'bundle')
        manifest_path = bundle/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(requires_draft_model=True, arguments=['--parallel', '2', '--context', '8192'],
                        speculative_methods=['dflash'], default_speculative_method='dflash',
                        draft_model_architectures=['DFlash2DraftModel'])
        (self.draft/'config.json').write_text('{"architectures":["DFlash2DraftModel"]}')
        manifest_path.write_text(json.dumps(manifest))
        profile = make_profile(self.root/'profile')
        spec = {**self.spec, 'bundle': str(bundle), 'devices': ['0'], 'draft_model': str(self.draft)}
        deployment = prepare_engine(self.root/'home', spec, profile, model_id='bench')
        argv = deployment['commands'][0]['argv']
        self.assertNotIn('--no-drafts', argv)
        self.assertEqual(argv[argv.index('--drafter')+1], str(self.draft))
        manifest['arguments'].append('--no-drafts')
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(SetupError, 'preserve speculative decoding'):
            check_bundle(bundle, {'model_type': 'glm_moe_dsa'}, validate_profile(profile))

    def test_qwen_dflash_and_single_stream_settings_are_preserved(self):
        (self.draft/'config.json').write_text('{"architectures":["DFlash2DraftModel"]}')
        manifest = {**self.manifest, 'draft_model_architectures':['DFlash2DraftModel'],
                    'min_parallel':1, 'tensor_parallel_sizes':[1]}
        self.process('--drafter', str(self.draft), '--parallel', '1', '--tp', '1')
        self.assertEqual(draft_settings(self.root,self.spec,manifest,'bench',proc_root=self.proc),
                         {'draft_model':str(self.draft),'parallel':1})
        (self.proc/'100/cmdline').write_bytes(('\0'.join(['python','-m','tensorfold','serve',str(self.model),
            '--drafter',str(self.draft),'--parallel','1','--tp','2'])+'\0').encode())
        with self.assertRaisesRegex(SetupError,'will not be disabled or replaced'):
            draft_settings(self.root,self.spec,manifest,'bench',proc_root=self.proc)

    def test_gpu_count_respects_package_support(self):
        bundle=make_bundle(self.root/'bundle')
        manifest=json.loads((bundle/'manifest.json').read_text())
        manifest['tensor_parallel_sizes']=[1]
        (bundle/'manifest.json').write_text(json.dumps(manifest))
        profile=make_profile(self.root/'profile')
        spec={**self.spec,'bundle':str(bundle)}
        with mock.patch('owner_mode.native_setup.cuda_inventory',return_value=(['0','1'], {'0':'sm_89','1':'sm_89'})):
            deployment=prepare_engine(self.root/'home',spec,profile,model_id='bench')
        self.assertEqual(len(deployment['commands']),1)
        with self.assertRaisesRegex(SetupError,'rank count'):
            prepare_engine(self.root/'home',{**spec,'devices':['0','1']},profile,model_id='bench')
