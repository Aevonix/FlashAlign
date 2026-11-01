# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Exact native payloads keep their identity across declared serving bases."""
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_native_profile import native_fixture, wizard
from test_native_setup import make_bundle
from owner_mode import native_cluster, native_remote
from owner_mode.native_bundle import check_bundle, validate_manifest
from owner_mode.native_setup import check, prepare
from owner_mode.profiles import NATIVE_FORMAT, NATIVE_REQUIRES, validate_profile
from owner_mode.setup_runtime import SetupError
from owner_mode.quickstart import make_default_plan


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for part in iter(lambda:source.read(8192), b''):value.update(part)
    return value.hexdigest()


class TensorFoldNativeProfile(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.home = self.root/'home';self.home.mkdir()
        self.profile = self.root/'profile'
        self.doc, self.payload = native_fixture(self.profile, world=4)
        self.model = self.root/'model';self.model.mkdir()
        (self.model/'config.json').write_text(json.dumps({'model_type': 'deepseek_v41', 'hidden_size': 32,
            'quantization_config': {'quant_method': 'exl3'}}))
        (self.model/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        self.bundle = make_bundle(self.root/'bundle')
        self.manifest = json.loads((self.bundle/'manifest.json').read_text())
        self.manifest.update(runtime_version='1.2.0', model_types=['deepseek_v41'],
            transforms=['native_matrix_override_v1'], profile_schemas=[NATIVE_FORMAT],
            supported_profile_requires=list(NATIVE_REQUIRES), supported_profile_strengths=[1.0],
            profile_target_model=self.doc['model'], tensor_parallel_sizes=[4],
            compatible_profile_bases=[{'id':'fixture/exl3-base','revision':'3'*40,'model_type':'deepseek_v41',
                'format':'EXL3','config_sha256':sha(self.model/'config.json'),
                'index_sha256':sha(self.model/'model.safetensors.index.json')}])
        self.write_manifest()
        self.spec = {'kind':'tensorfold','bundle':str(self.bundle),'model_path':str(self.model),
                     'devices':['0','1','2','3']}

    def write_manifest(self):
        (self.bundle/'manifest.json').write_text(json.dumps(self.manifest))

    def test_contract_keeps_original_target_and_explicit_serving_base(self):
        original = (self.profile/'profile.json').read_bytes()
        checked = check(self.home, self.spec, validate_profile(self.profile), model_id='fixture/model')
        self.assertEqual(checked['runtime_version'], '1.2.0')
        self.assertEqual(checked['manifest']['profile_target_model'], self.doc['model'])
        self.assertEqual((self.profile/'profile.json').read_bytes(), original)
        self.assertNotEqual(checked['manifest']['compatible_profile_bases'][0]['id'], self.doc['model']['id'])

    def test_unsupported_contract_or_wrong_base_stops_before_download(self):
        (self.profile/'native.safetensors').unlink()
        self.manifest['compatible_profile_bases'][0]['config_sha256'] = '0'*64
        self.write_manifest()
        with mock.patch('owner_mode.profile_assets.materialize_profile', side_effect=AssertionError('download')):
            with self.assertRaisesRegex(SetupError, 'metadata differs'):
                prepare(self.home, self.spec, self.profile, model_id='fixture/model')
        for change in ({'runtime_version':'1.1.9'}, {'supported_profile_requires':[]},
                       {'supported_profile_strengths':[1.1]}, {'tensor_parallel_sizes':[1,4]}):
            with self.subTest(change=change), self.assertRaises(SetupError):
                validate_manifest({**self.manifest, **change})

    def test_prepare_materializes_once_and_local_ranks_share_exact_payload(self):
        (self.profile/'native.safetensors').unlink()
        with mock.patch('owner_mode.profile_assets._download', side_effect=lambda r,p:p.write_bytes(self.payload)) as fetch:
            result = prepare(self.home, self.spec, self.profile, model_id='fixture/model')
        self.assertEqual(fetch.call_count, 1)
        final = Path(result['profile_path'])
        self.assertEqual((final/'native.safetensors').read_bytes(), self.payload)
        self.assertEqual((final/'profile.json').read_bytes(), (self.profile/'profile.json').read_bytes())
        self.assertEqual(len(result['commands']), 4)
        for command in result['commands']:
            argv = command['argv']
            self.assertEqual(argv[argv.index('--owner-profile')+1], str(final))
        self.assertFalse(list(Path(result['state']).glob('rank-*/profile/native.safetensors')))

    def test_default_wizard_accepts_only_a_declared_tf_base_without_changing_profile(self):
        args=SimpleNamespace(home=self.home,model_path=self.model,profile=self.profile,
            engine_bundle=self.bundle,backend='tensorfold',profiles=[],engine_python=None,
            devices='0,1,2,3',tools_default=None,engine_only=True)
        with mock.patch('builtins.input',side_effect=AssertionError('unexpected prompt')):
            path=make_default_plan(args,{},wizard.Console(color=False))
        plan=json.loads(path.read_text())
        self.assertEqual(plan['model'],self.doc['model']['id'])
        self.assertEqual(plan['profile'],str(self.profile.resolve()))
        self.assertEqual(plan['engine']['kind'],'tensorfold')
        self.manifest['compatible_profile_bases'][0]['index_sha256']='0'*64
        self.write_manifest()
        with self.assertRaisesRegex(SetupError,'declared native-profile TensorFold serving base'):
            make_default_plan(args,{},wizard.Console(color=False))

    def test_bundle_builder_emits_valid_latest_schema_aware_package(self):
        source=self.root/'source';source.mkdir();(source/'LICENSE').write_text('Fixture source license\n')
        def git(*args):
            return subprocess.check_output(['git','-C',str(source),*args],text=True,stderr=subprocess.DEVNULL).strip()
        git('init');git('add','LICENSE');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','Fixture')
        revision=git('rev-parse','HEAD')
        protected={'version','engine','runtime_version','source_revision','platform','cuda_architectures',
            'binary','follower_binary','source_archive','source_license_notice','binary_transform_receipt','files'}
        caps=self.root/'capabilities.json';caps.write_text(json.dumps({k:v for k,v in self.manifest.items() if k not in protected}))
        output=self.root/'new-bundle';catalog=self.root/'catalog-entry.json'
        script=Path(__file__).resolve().parents[2]/'tools/build-native-bundle.py'
        command=[sys.executable,str(script),'--source-tree',str(source),'--source-revision',revision,
            '--binary',str(self.bundle/'bin/tensorfold-native'),'--capabilities',str(caps),
            '--platform',self.manifest['platform'],'--cuda-architecture','sm_120',
            '--output',str(output),'--release-tag','fixture','--catalog-entry',str(catalog)]
        auxiliary=self.root/'reference-source.tar.gz';auxiliary.write_bytes(b'fixture source archive')
        notice=self.root/'reference-license.txt';notice.write_text('Fixture reference source license\n')
        command+=['--auxiliary-source','reference-source.tar.gz='+str(auxiliary),
                  '--auxiliary-source-notice','reference-source.tar.gz='+str(notice)]
        for name in ('libnccl.so.2','libtfgrammar.so'):
            command+=['--library',name+'='+str(self.bundle/'lib'/name),
                      '--library-notice',name+'='+str(self.bundle/'licenses'/(name+'.txt'))]
        subprocess.run(command,check=True,capture_output=True,text=True)
        packaged=json.loads((output/'manifest.json').read_text())
        self.assertEqual(packaged['runtime_version'],'1.2.0')
        self.assertEqual(packaged['profile_target_model'],self.doc['model'])
        check_bundle(output,{'model_type':'deepseek_v41'},validate_profile(self.profile))
        entry=json.loads(catalog.read_text())
        self.assertEqual(entry['profile_schemas'],[NATIVE_FORMAT])
        self.assertEqual(entry['transforms'],['native_matrix_override_v1'])
        self.assertNotIn('profile.json',entry['files'])
        self.assertEqual(packaged['auxiliary_sources'],[{
            'archive':'sources/auxiliary/reference-source.tar.gz',
            'license':'sources/auxiliary/reference-source.tar.gz.LICENSE.txt'}])
        self.assertEqual((output/packaged['auxiliary_sources'][0]['archive']).read_bytes(),auxiliary.read_bytes())


class StreamingRankFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_file_tar_transfer_and_receiver_never_require_unbounded_reads(self):
        source = self.root/'native.safetensors'
        with source.open('wb') as output:
            for _ in range(17):output.write(b'x'*(1024*1024))
        files = {'profile/native.safetensors':(source, False),'unit.json':(b'{}',False)}
        def rpc(host, operation, **kwargs):
            self.assertEqual(operation, 'stage');self.assertNotIn('payload', kwargs)
            class Bounded:
                def __init__(self, source):self.source=source
                def read(self, size=-1):
                    if not 0 <= size <= 8*1024*1024:raise AssertionError('unbounded read')
                    return self.source.read(size)
            with Path(kwargs['payload_file']).open('rb') as stream:
                return native_remote.stage(kwargs, Bounded(stream))
        with mock.patch('owner_mode.native_cluster.rpc', side_effect=rpc):
            result = native_cluster.stage_files('local', files, parent=str(self.root/'ranks'), id='a'*20)
        self.assertEqual(sha(Path(result['state'])/'profile/native.safetensors'), sha(source))

    def test_rpc_uses_file_stdin_instead_of_materializing_input(self):
        payload = self.root/'payload.tar';payload.write_bytes(b'payload')
        def run(argv, **kwargs):
            self.assertNotIn('input', kwargs)
            request = kwargs['stdin']
            size = struct.unpack('>I', request.read(4))[0]
            self.assertEqual(json.loads(request.read(size))['operation'], 'stage')
            self.assertEqual(request.read(), b'payload')
            return SimpleNamespace(returncode=0, stdout=b'{"status":"prepared"}')
        with mock.patch('owner_mode.native_cluster.subprocess.run', side_effect=run):
            self.assertEqual(native_cluster.rpc('local','stage',payload_file=payload)['status'], 'prepared')

    def test_pax_size_bomb_is_rejected_before_body_allocation(self):
        member=tarfile.TarInfo('metadata');member.type=tarfile.XHDTYPE;member.size=2**31
        request={'parent':str(self.root/'ranks'),'id':'b'*20,
                 'files':{'unit.json':{'sha256':hashlib.sha256(b'{}').hexdigest(),'bytes':2}}}
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            native_remote.stage(request, member.tobuf()+b'\0'*1024)


if __name__ == '__main__':unittest.main()
