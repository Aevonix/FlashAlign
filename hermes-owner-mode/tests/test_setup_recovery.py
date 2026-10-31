# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fault-boundary tests with temporary homes, files and a local protocol fixture."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from test_setup import wizard, make_profile
from test_engine_setup import make_bundle
from support import Engine
from owner_mode import auth, backend_setup, engine_setup, profiles, setup_transaction
from owner_mode.contacts import ApprovalStore
from owner_mode.setup_runtime import SetupError


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.home=self.root/'home';self.home.mkdir()
        self.key=self.root/'engine.key';auth.ensure_key(self.key)
        self.profile=make_profile(self.root/'profile')
        self.paths=[self.home/'.hermes/config.yaml',self.home/'.openclaw/openclaw.json']
        for index,path in enumerate(self.paths):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'before':index})+'\n');path.chmod(0o640)
        self.plugin=self.home/'.hermes/plugins/owner-mode';self.plugin.mkdir(parents=True)
        (self.plugin/'operator.txt').write_text('previous plugin');(self.plugin/'operator.txt').chmod(0o640)
        self.before={str(p):(p.read_bytes(),p.stat().st_mode & 0o777) for p in [*self.paths,self.plugin/'operator.txt',self.key]}
        approvals=self.home/'.config/owner-mode/approvals.json'
        store=ApprovalStore(approvals);store.set_contact('rcs','default','+15550000022',True)
        self.before[str(approvals)]=(approvals.read_bytes(),approvals.stat().st_mode & 0o777)
        token=self.home/'.config/owner-mode/upstream-api.key'
        token.write_text('previous synthetic token');token.chmod(0o600)
        self.before[str(token)]=(token.read_bytes(),0o600)

    def provision(self):
        return wizard.provision(self.home,['hermes','openclaw'],'http://127.0.0.1:9/v1','bench','custom-owner',
            [('rcs','default','+15550000011')],False,self.key,
            profile_details=profiles.validate_profile(self.profile),api_key='new synthetic token')

    def unchanged(self):
        for path,(data,mode) in self.before.items():
            self.assertEqual(Path(path).read_bytes(),data,path)
            self.assertEqual(Path(path).stat().st_mode & 0o777,mode,path)

    def test_interrupted_key_creation_does_not_publish_partial_key(self):
        target=self.root/'new.key'
        with mock.patch.object(auth.secrets,'token_bytes',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):auth.ensure_key(target)
        self.assertFalse(target.exists())
        self.assertTrue(auth.ensure_key(target));saved=digest(target)
        self.assertFalse(auth.ensure_key(target));self.assertEqual(digest(target),saved)
        self.assertEqual(target.stat().st_mode & 0o777,0o600)
        self.assertFalse(list(self.root.glob('.flashalign-key-*')))

    def test_existing_key_and_symlink_are_not_replaced(self):
        saved=digest(self.key)
        link=self.root/'key-link';link.symlink_to(self.key)
        with self.assertRaises(OSError):auth.ensure_key(link)
        self.assertTrue(link.is_symlink());self.assertEqual(digest(self.key),saved)
        self.assertEqual(self.key.stat().st_mode & 0o777,0o600)

    def test_concurrent_key_writer_is_preserved_at_publication(self):
        target=self.root/'concurrent.key'
        existing=b'k'*32
        original_link=os.link
        def publish_after_other_writer(source,destination):
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as stream:stream.write(existing)
            original_link(source,destination)
        with mock.patch.object(auth.os,'link',side_effect=publish_after_other_writer):
            self.assertFalse(auth.ensure_key(target))
        self.assertEqual(auth.read_key(target),existing)
        self.assertFalse(list(self.root.glob('.flashalign-key-*')))

    def test_setup_refuses_files_it_cannot_restore_with_original_ownership(self):
        with mock.patch.object(setup_transaction.os,'getuid',return_value=os.getuid()+1):
            with self.assertRaisesRegex(SetupError,'owned by the current user'):
                setup_transaction._fingerprint(self.paths[0])
        self.unchanged()

    def test_interrupted_integration_copy_can_retry_without_overwriting_source(self):
        source=self.root/'kit';source.mkdir();(source/'first').write_text('first');(source/'second').write_text('second')
        target=self.root/'integration'
        def fail(_source,destination,**_):
            Path(destination).mkdir();(Path(destination)/'first').write_text('first');raise KeyboardInterrupt
        with mock.patch.object(backend_setup.shutil,'copytree',side_effect=fail):
            with self.assertRaises(KeyboardInterrupt):backend_setup._copy_kit(source,target)
        self.assertFalse(target.exists())
        backend_setup._copy_kit(source,target)
        self.assertEqual((target/'second').read_text(),'second')
        self.assertEqual((source/'first').read_text(),'first')
        (target/'first').write_text('operator edit')
        with self.assertRaises(SetupError):backend_setup._copy_kit(source,target)
        self.assertEqual((target/'first').read_text(),'operator edit')

    def test_interrupted_profile_copy_can_retry_and_existing_profile_is_preserved(self):
        target=self.root/'staged-profile';original=digest(self.profile/'profile.safetensors')
        real_copy=shutil.copyfile
        def fail(source,destination,*args,**kwargs):
            real_copy(source,destination,*args,**kwargs)
            if Path(source).name=='profile.json':raise KeyboardInterrupt
        with mock.patch.object(profiles.shutil,'copyfile',side_effect=fail):
            with self.assertRaises(KeyboardInterrupt):profiles.stage_profile(self.profile,target)
        self.assertFalse(target.exists());self.assertEqual(digest(self.profile/'profile.safetensors'),original)
        profiles.stage_profile(self.profile,target)
        self.assertEqual(digest(target/'profile.safetensors'),original)
        (target/'profile.safetensors').write_bytes(b'operator edit')
        with self.assertRaises(profiles.ProfileError):profiles.stage_profile(self.profile,target)
        self.assertEqual((target/'profile.safetensors').read_bytes(),b'operator edit')

    def test_staging_failure_keeps_both_configs_approvals_and_plugin(self):
        real_save=wizard.save_config
        def fail(path,value):
            if Path(path)==self.paths[1]:raise OSError('injected second config write')
            return real_save(path,value)
        with mock.patch.object(wizard,'save_config',side_effect=fail):
            with self.assertRaises(OSError):self.provision()
        self.unchanged();self.assertFalse(setup_transaction.pending_setup(self.home))
        self.assertEqual(self.provision()['status'],'awaiting_restart')
        self.assertEqual(digest(self.key),hashlib.sha256(self.before[str(self.key)][0]).hexdigest())

    def test_mid_commit_failure_restores_exact_before_images_and_modes(self):
        original=setup_transaction._install;failed=False
        def fail(source,target,kind,identifier):
            nonlocal failed
            original(source,target,kind,identifier)
            if Path(target)==self.paths[0] and not failed:
                failed=True;raise OSError('injected commit failure')
        with mock.patch.object(setup_transaction,'_install',side_effect=fail):
            with self.assertRaises(OSError):self.provision()
        self.unchanged();self.assertFalse(setup_transaction.pending_setup(self.home))
        self.assertEqual(self.provision()['status'],'awaiting_restart')

    def crash_commit(self):
        def child():
            original=setup_transaction._install
            def crash(source,target,kind,identifier):
                original(source,target,kind,identifier)
                if Path(target)==self.paths[0]:os._exit(37)
            setup_transaction._install=crash
            self.provision()
        process=multiprocessing.get_context('fork').Process(target=child)
        process.start();process.join(15)
        if process.is_alive():process.terminate();process.join();self.fail('crash fixture timed out')
        self.assertEqual(process.exitcode,37)
        self.assertTrue(setup_transaction.pending_setup(self.home))

    def test_abrupt_commit_is_recovered_before_retry(self):
        self.crash_commit();self.assertTrue(setup_transaction.recover_setup(self.home))
        self.unchanged();self.assertFalse(setup_transaction.pending_setup(self.home))
        self.assertEqual(self.provision()['status'],'awaiting_restart')

    def test_recovery_refuses_to_overwrite_an_operator_edit_after_interruption(self):
        self.crash_commit();self.paths[0].write_text('{"operator":"new edit"}\n')
        with self.assertRaisesRegex(SetupError,'was edited'):setup_transaction.recover_setup(self.home)
        self.assertEqual(self.paths[0].read_text(),'{"operator":"new edit"}\n')
        self.assertTrue(setup_transaction.pending_setup(self.home))
        self.assertEqual(self.key.read_bytes(),self.before[str(self.key)][0])

    def test_recovery_refuses_a_parent_symlink_outside_the_home(self):
        self.crash_commit();outside=self.root/'outside';outside.mkdir()
        (outside/'config.yaml').write_text('outside file')
        os.replace(self.home/'.hermes',self.root/'original-hermes')
        (self.home/'.hermes').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(SetupError,'escapes'):setup_transaction.recover_setup(self.home)
        self.assertEqual((outside/'config.yaml').read_text(),'outside file')

    def test_recovery_preserves_an_operator_deleted_config(self):
        self.crash_commit();self.paths[0].unlink()
        with self.assertRaisesRegex(SetupError,'was edited'):setup_transaction.recover_setup(self.home)
        self.assertFalse(self.paths[0].exists())

    def test_failed_native_registration_does_not_apply_routing(self):
        engine=Engine(self.key.read_bytes(),profiles=('custom-owner',));self.addCleanup(engine.close)
        executable=self.root/'openclaw';executable.write_text('#!/bin/sh\nexit 1\n');executable.chmod(0o700)
        found={name:{'detected':True,'runtime_compatible':True,'executable':str(executable),'model_urls':[]} for name in ['hermes','openclaw']}
        plan=self.root/'plan.json';engine_setup.private_json(plan,{'harnesses':['hermes','openclaw'],'base_url':engine.base_url,'model':'bench','profile':str(self.profile),'engine_key_file':str(self.key),'contacts':[{'channel':'rcs','account':'default','contact':'+15550000011'}],'tools_default':False})
        with mock.patch.object(wizard,'discover',return_value=found),mock.patch.object(wizard,'restart_harnesses',side_effect=AssertionError('must not restart')),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SetupError,'registration failed'):
                wizard.main(['--home',str(self.home),'--config',str(plan),'--apply'])
        self.unchanged()
        executable.write_text('#!/bin/sh\nexit 0\n')
        with mock.patch.object(wizard,'discover',return_value=found),mock.patch.object(wizard,'restart_harnesses',side_effect=AssertionError('must not restart')),redirect_stdout(io.StringIO()):
            result=wizard.main(['--home',str(self.home),'--config',str(plan),'--apply'])
        self.assertEqual(result['status'],'awaiting_restart')

    def test_engine_receipt_failure_keeps_reusable_preparation_and_key(self):
        bundle=make_bundle(self.root/'bundle');model=self.root/'model';model.mkdir()
        (model/'config.json').write_text('{"model_type":"glm_moe_dsa","hidden_size":2}')
        spec={'kind':'tensorfold','model_path':str(model),'bundle':str(bundle),'devices':['0']}
        real=engine_setup.private_json
        def fail(path,value):
            if Path(path).name=='deployment.json':raise OSError('injected deployment receipt failure')
            return real(path,value)
        with mock.patch.object(engine_setup,'private_json',side_effect=fail):
            with self.assertRaises(OSError):engine_setup.prepare_engine(self.home,spec,self.profile,model_id='bench')
        keys=list((self.home/'.config/owner-mode/engines').glob('*/engine.key'));self.assertEqual(len(keys),1)
        before=digest(keys[0]);result=engine_setup.prepare_engine(self.home,spec,self.profile,model_id='bench')
        self.assertEqual(result['status'],'prepared');self.assertEqual(digest(Path(result['key_file'])),before)
        self.assertFalse((Path(result['state'])/'running.json').exists())

    def test_new_profile_payload_uses_new_engine_state_and_preserves_previous_files(self):
        bundle=make_bundle(self.root/'bundle');model=self.root/'model';model.mkdir()
        (model/'config.json').write_text('{"model_type":"glm_moe_dsa","hidden_size":2}')
        spec={'kind':'tensorfold','model_path':str(model),'bundle':str(bundle),'devices':['0']}
        first=engine_setup.prepare_engine(self.home,spec,self.profile,model_id='bench')
        saved=digest(Path(first['profile_path'])/'profile.safetensors');key=digest(first['key_file'])
        make_profile(self.profile,values=(0.,1.))
        second=engine_setup.prepare_engine(self.home,spec,self.profile,model_id='bench')
        self.assertNotEqual(first['state'],second['state'])
        self.assertEqual(digest(Path(first['profile_path'])/'profile.safetensors'),saved)
        self.assertEqual(digest(first['key_file']),key)


if __name__=='__main__':unittest.main()
