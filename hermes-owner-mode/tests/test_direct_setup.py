# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Engine-only setup can prepare and activate without any installed harness."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_setup import wizard, make_profile
from test_engine_setup import make_bundle, SERVER, fixture_deployment
from owner_mode.automation import read_plan
from owner_mode.engine_setup import private_json, stop_engine
from owner_mode.quickstart import make_default_plan
from owner_mode.setup_runtime import SetupError


class DirectSetup(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.home=self.root/'home';self.home.mkdir()
        self.model=self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text('{"model_type":"glm_moe_dsa","hidden_size":2}')
        self.profile=make_profile(self.root/'profile')
        self.bundle=make_bundle(self.root/'bundle')
        self.found={name:{'detected':False,'runtime_compatible':False} for name in ['hermes','openclaw']}
        self.path=self.root/'plan.json'
        self.plan={'engine_only':True,'model':'bench','profile':str(self.profile),
                   'engine':{'model_path':str(self.model),'bundle':str(self.bundle),'devices':['0']}}

    def test_preparation_needs_no_harness_and_stops_before_launch(self):
        private_json(self.path,self.plan)
        with mock.patch.object(wizard,'discover',return_value=self.found),redirect_stdout(io.StringIO()):
            checked=wizard.main(['--home',str(self.home),'--config',str(self.path),'--check'])
            self.assertTrue(checked['artifact_validated'])
            self.assertFalse((self.home/'.config').exists())
            result=wizard.main(['--home',str(self.home),'--config',str(self.path),'--apply'])
        self.assertEqual(result['harnesses'],[])
        self.assertEqual(result['status'],'engine_prepared')
        self.assertFalse((Path(result['engine']['state'])/'running.json').exists())
        self.assertFalse((self.home/'.hermes').exists())
        self.assertFalse((self.home/'.openclaw').exists())
        self.assertEqual(Path(result['engine']['key_file']).stat().st_mode&0o777,0o600)

    def test_engine_only_does_not_silently_configure_contacts(self):
        private_json(self.path,{**self.plan,'contacts':[{'channel':'rcs','account':'default','contact':'+15550000011'}]})
        with self.assertRaisesRegex(SetupError,'do not configure'):
            read_plan(self.path)

    def test_activation_writes_client_config_and_never_restarts_a_harness(self):
        import sys
        server=self.root/'server.py';server.write_text(SERVER)
        command=[sys.executable,str(server),'{port}']
        original_fixture = fixture_deployment
        def launch(home, spec, profile, **kwargs):
            return original_fixture(home, {**spec, 'command': command}, profile, **kwargs)
        private_json(self.path,self.plan)
        with (mock.patch.object(wizard,'discover',return_value=self.found),
              mock.patch.object(wizard,'restart_harnesses',side_effect=AssertionError('harness restart')),
              mock.patch('owner_mode.automation.prepare_engine',side_effect=launch),
              redirect_stdout(io.StringIO())):
            result=wizard.main(['--home',str(self.home),'--config',str(self.path),'--activate'])
        try:
            self.assertEqual(result['status'],'engine_ready')
            self.assertEqual(result['restart'],{})
            config=Path(result['client_config'])
            self.assertEqual(config.stat().st_mode&0o777,0o600)
            self.assertEqual(json.loads(config.read_text())['key_file'],result['engine']['key_file'])
            self.assertFalse((self.home/'.config/owner-mode/approvals.json').exists())
        finally:
            stop_engine(result['engine'])

    def test_default_engine_only_plan_has_no_contact_or_tool_prompts(self):
        args=SimpleNamespace(engine_only=True,home=self.home,model_path=self.model,profile=self.profile,
                             engine_bundle=self.bundle,profiles=[],engine_python=None,devices='0',tools_default=None)
        with mock.patch('builtins.input',side_effect=AssertionError('unexpected prompt')),redirect_stdout(io.StringIO()):
            plan=make_default_plan(args,self.found,wizard.Console(color=False))
        value=json.loads(plan.read_text())
        self.assertTrue(value['engine_only'])
        self.assertEqual(value['harnesses'],[])
        self.assertEqual(value['contacts'],[])


if __name__=='__main__':unittest.main()
