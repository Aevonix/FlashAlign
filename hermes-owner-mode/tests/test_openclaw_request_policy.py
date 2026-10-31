# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_setup import wizard
from owner_mode.model_metadata import (checked_openclaw_request_policy,
    declared_openclaw_request_policy, check_openclaw_request_config, check_openclaw_request_file)
from owner_mode.setup_runtime import SetupError
from owner_mode import automation

POLICY={'output_budget':'remaining_context','supports_thinking_off':False}
MODEL='synthetic-family'


class OpenClawRequestPolicy(unittest.TestCase):
    def test_declaration_is_model_scoped_and_strict(self):
        meta={'model_types':['synthetic'],'openclaw_request_policy_by_model_type':{'synthetic':POLICY}}
        self.assertEqual(declared_openclaw_request_policy(meta,'synthetic'),POLICY)
        self.assertIsNone(declared_openclaw_request_policy(meta,'another'))
        for declarations in (None,{'synthetic':None}):
            with self.assertRaises(SetupError):declared_openclaw_request_policy(
                {'model_types':['synthetic'],'openclaw_request_policy_by_model_type':declarations},'synthetic')
        for value in ({},{'output_budget':'8192'},{'supports_thinking_off':0},
                      {'implicit_model_limit':True},{'headers':{}},[] ):
            with self.subTest(value=value),self.assertRaises(SetupError):checked_openclaw_request_policy(value)
        self.assertIsNone(checked_openclaw_request_policy(None))

    def test_absent_preferences_and_supported_levels_are_preserved(self):
        for level in (None,'low','high','max'):
            cfg={'agents':{'defaults':{} if level is None else {'thinkingDefault':level}}}
            before=json.dumps(cfg)
            check_openclaw_request_config(cfg,MODEL,POLICY)
            self.assertEqual(json.dumps(cfg),before)

    def test_saved_off_and_per_model_off_are_rejected(self):
        for cfg in ({'agents':{'defaults':{'thinkingDefault':'off'}}},
                    {'agents':{'defaults':{'thinkingDefault':' OFF '}}},
                    {'agents':{'defaults':{'models':{'owner-mode/'+MODEL:{'params':{'thinking':False}}}}}},
                    {'agents':{'entries':{'child':{'thinkingDefault':'off'}}}}):
            with self.subTest(cfg=cfg),self.assertRaisesRegex(SetupError,'thinking off'):
                check_openclaw_request_config(cfg,MODEL,POLICY)

    def test_model_override_takes_precedence_over_global_off(self):
        cfg={'agents':{'defaults':{'thinkingDefault':'off','models':{
            'owner-mode/'+MODEL:{'params':{'thinking':'high'}}}}}}
        check_openclaw_request_config(cfg,MODEL,POLICY)

    def test_other_agent_model_is_not_rejected(self):
        cfg={'agents':{'defaults':{'thinkingDefault':'low'},
            'entries':{'other':{'model':'another/model','thinkingDefault':'off'}}}}
        check_openclaw_request_config(cfg,MODEL,POLICY)

    def test_required_local_worker_is_rejected_but_unused_or_gateway_is_not(self):
        profiles={'local':{'provider':'device','settings':{'inference':'worker'}}}
        cfg={'cloudWorkers':{'profiles':profiles}}
        check_openclaw_request_config(cfg,MODEL,POLICY)
        cfg['cloudWorkers']['requiredProfile']='local'
        with self.assertRaisesRegex(SetupError,'worker-local'):check_openclaw_request_config(cfg,MODEL,POLICY)
        profiles['local']['settings']['inference']='gateway'
        check_openclaw_request_config(cfg,MODEL,POLICY)
        profiles['local']={'provider':'another','settings':{'inference':'worker'}}
        check_openclaw_request_config(cfg,MODEL,POLICY)

    def provision(self, home, **info):
        with patch.object(wizard,'free_port',return_value=48123):
            wizard.provision(home,['openclaw'],'http://127.0.0.1:48124/v1',MODEL,'synthetic-profile',
                             [],True,model_info={'contextWindow':65536,'openclaw_request_policy':POLICY,**info})
        return json.loads((home/'.openclaw/openclaw.json').read_text())

    def test_writer_records_synthetic_capacity_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=self.provision(Path(temporary))
            plugin=cfg['plugins']['entries']['owner-mode']['config']
            self.assertEqual(plugin['requestPolicy'],{**POLICY,'implicit_model_limit':True})
            self.assertEqual(plugin['model']['maxTokens'],65536)
            self.assertEqual(plugin['model']['contextWindow'],65536)

    def test_real_metadata_limit_is_not_synthetic_even_at_old_default_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=self.provision(Path(temporary),maxTokens=8192)
            plugin=cfg['plugins']['entries']['owner-mode']['config']
            self.assertFalse(plugin['requestPolicy']['implicit_model_limit'])
            self.assertEqual(plugin['model']['maxTokens'],8192)

    def test_harness_default_does_not_become_an_engine_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg=self.provision(Path(temporary),maxTokens=8192,maxTokensSource='harness_output')
            self.assertTrue(cfg['plugins']['entries']['owner-mode']['config']['requestPolicy']['implicit_model_limit'])

    def test_off_fails_before_staging_keys_or_changing_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary);config=home/'.openclaw/openclaw.json';config.parent.mkdir()
            before=json.dumps({'agents':{'defaults':{'thinkingDefault':'off'}}});config.write_text(before)
            with self.assertRaisesRegex(SetupError,'thinking off'):self.provision(home)
            self.assertEqual(config.read_text(),before)
            self.assertFalse((home/'.config/owner-mode/engine.key').exists())

    def test_file_preflight_rejects_off_without_changing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'openclaw.json'
            before='{"agents":{"defaults":{"thinkingDefault":"off"}}}';path.write_text(before)
            with self.assertRaises(SetupError):check_openclaw_request_file(path,MODEL,POLICY)
            self.assertEqual(path.read_text(),before)

    def test_plan_check_rejects_saved_off_before_engine_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary);path=home/'.openclaw/openclaw.json';path.parent.mkdir()
            path.write_text('{"agents":{"defaults":{"thinkingDefault":"off"}}}')
            plan={'harnesses':['openclaw'],'model':MODEL,'engine':{'kind':'sglang'},'profile':'test',
                  'profile_paths':[]}
            found={'openclaw':{'detected':True,'runtime_compatible':True}}
            with patch.object(automation,'pending_setup',return_value=False),\
                 patch.object(automation,'read_plan',return_value=plan),\
                 patch.object(automation,'discover_profiles',return_value=([],[])),\
                 patch.object(automation,'resolve_profile',return_value={'source':'local','manifest':'fixture','name':'test'}),\
                 patch.object(automation,'check_engine_setup',return_value={'openclaw_request_policy':POLICY}),\
                 patch.object(automation,'prepare_engine') as prepare:
                with self.assertRaisesRegex(SetupError,'thinking off'):
                    automation.run_plan(home/'plan.json',home,found,lambda *a,**k:None)
                prepare.assert_not_called()


if __name__=='__main__':unittest.main()
