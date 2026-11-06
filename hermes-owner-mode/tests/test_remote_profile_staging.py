# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_setup import make_profile
from owner_mode import native_cluster, native_remote
from owner_mode.profile_staging import remote_profile_stage
from owner_mode.profile_lifecycle import LifecycleError


class RemoteProfileStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=make_profile(self.root/'source')
        self.remote=self.root/'remote';self.remote.mkdir(mode=0o700)
        self.target={'host':'example-host','staging_directory':str(self.remote)}

    def stage(self,host,files,**fields):
        self.assertEqual(host,'example-host')
        wire={name:(Path(data).read_bytes() if isinstance(data,Path) else data,executable)
              for name,(data,executable) in files.items()}
        return native_remote.stage({**fields,'files':native_cluster.file_records(files)},native_cluster.archive(wire))

    def discard(self,host,action,**fields):
        self.assertEqual(host,'example-host');self.assertEqual(action,'discard_artifact')
        return native_remote.discard_artifact(fields)

    def test_snapshot_is_transferred_and_removed_after_operation(self):
        with mock.patch.object(native_cluster,'stage_files',side_effect=self.stage), \
             mock.patch.object(native_cluster,'rpc',side_effect=self.discard):
            with remote_profile_stage(self.source,self.target) as directory:
                path=Path(directory)
                self.assertEqual((path/'profile.safetensors').read_bytes(),(self.source/'profile.safetensors').read_bytes())
                self.assertNotEqual(path,self.source)
                temporary=path.parent
            self.assertFalse(temporary.exists())
            self.assertTrue(self.source.is_dir())

    def test_failed_operation_also_discards_unchanged_upload(self):
        with mock.patch.object(native_cluster,'stage_files',side_effect=self.stage), \
             mock.patch.object(native_cluster,'rpc',side_effect=self.discard):
            with self.assertRaisesRegex(RuntimeError,'operator failed'):
                with remote_profile_stage(self.source,self.target) as directory:
                    temporary=Path(directory).parent
                    raise RuntimeError('operator failed')
            self.assertFalse(temporary.exists())

    def test_changed_upload_is_retained_for_inspection(self):
        with mock.patch.object(native_cluster,'stage_files',side_effect=self.stage), \
             mock.patch.object(native_cluster,'rpc',side_effect=self.discard):
            with self.assertRaisesRegex(ValueError,'changed'):
                with remote_profile_stage(self.source,self.target) as directory:
                    path=Path(directory)
                    (path/'profile.safetensors').write_bytes(b'changed')
            self.assertTrue(path.exists())

    def test_discard_does_not_accept_a_managed_process_directory(self):
        files={'unit.json':(b'{}',False)}
        fields={'parent':str(self.remote),'id':'a'*20,'files':native_cluster.file_records(files)}
        receipt=native_remote.stage(fields,native_cluster.archive(files))
        with self.assertRaisesRegex(ValueError,'Only temporary'):
            native_remote.discard_artifact(fields)
        self.assertTrue(Path(receipt['state']).exists())

    def test_remote_settings_are_not_executable_metadata(self):
        for invalid in ({**self.target,'module':'evil'}, {**self.target,'host':'local'},
                        {**self.target,'staging_directory':'relative'}):
            with self.assertRaises(LifecycleError):
                with remote_profile_stage(self.source,invalid):pass
