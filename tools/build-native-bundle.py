#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a pinned native runtime archive from exact binaries and a Git source commit."""
import argparse
import gzip
import hashlib
import io
import importlib
import importlib.util
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-tree',type=Path,required=True)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--runtime-version', choices=['1.1.9', '1.2.0'], default='1.2.0',
                   help='Exact TensorFold release used by the pinned source revision')
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--follower-binary',type=Path)
    p.add_argument('--capabilities',type=Path,required=True)
    p.add_argument('--platform',choices=['linux-x86_64','linux-aarch64'],required=True)
    p.add_argument('--cuda-architecture',action='append',required=True)
    p.add_argument('--library',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--library-notice',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--auxiliary-source',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--auxiliary-source-notice',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--assets',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary-transform-receipt',type=Path,help='Audited strip-debug receipt for supplied binaries')
    p.add_argument('--release-tag',help='Planned FlashAlign release tag; does not publish')
    p.add_argument('--catalog-entry',type=Path,help='Write a reviewed download entry outside the runtime directory')
    args=p.parse_args()
    def source_files(values):
        result={}
        for value in values:
            name,sep,path=value.partition('=')
            if not sep or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',name) or name in result:
                p.error('auxiliary sources require unique safe NAME=PATH pairs')
            if not Path(path).is_file():p.error('auxiliary source file is missing')
            result[name]=Path(path)
        return result
    auxiliary=source_files(args.auxiliary_source)
    auxiliary_notices=source_files(args.auxiliary_source_notice)
    if set(auxiliary)!=set(auxiliary_notices):
        p.error('each auxiliary source requires its license and provenance notice')
    if bool(args.release_tag) != bool(args.catalog_entry):p.error('release-tag and catalog-entry must be provided together')
    if args.release_tag and not re.fullmatch(r'[A-Za-z0-9._-]+',args.release_tag):p.error('invalid release tag')
    if args.catalog_entry and args.catalog_entry.absolute().is_relative_to(args.output.absolute()):p.error('catalog entry must be outside the runtime directory')
    revision=subprocess.check_output(['git','-C',str(args.source_tree),'rev-parse',args.source_revision+'^{commit}'],text=True).strip()
    if revision!=args.source_revision:
        p.error('source-revision must be the complete immutable commit ID')
    if args.output.exists():p.error('output already exists; use a fresh directory')
    args.output.mkdir(parents=True)
    def put(source,name):
        dest=args.output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    put(args.binary,'bin/'+args.binary.name)
    if args.follower_binary:put(args.follower_binary,'bin/'+args.follower_binary.name)
    libraries=dict(item.split('=',1) for item in args.library)
    notices=dict(item.split('=',1) for item in args.library_notice)
    if set(libraries)!=set(notices):p.error('each bundled library needs its corresponding license/notice file')
    for name,source in libraries.items():
        if Path(name).name!=name or 'libcuda.so' in name:p.error('invalid library name or bundled GPU driver')
        put(Path(source),'lib/'+name);put(Path(notices[name]),'licenses/'+name+'.txt')
    if args.assets:shutil.copytree(args.assets,args.output/'assets/runtime')
    repo=Path(__file__).resolve().parents[1]
    put(repo/'LICENSE','LICENSE')
    put(repo/'LICENSE','licenses/FlashAlign/AGPL-3.0-only.txt')
    templates=repo/'tools/runtime-licenses'
    put(templates/'NOTICE.md','NOTICE.md');put(templates/'COMMERCIAL.md','COMMERCIAL.md')
    archive=subprocess.check_output(['git','-C',str(args.source_tree),'archive','--format=tar',revision])
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as source_tar:
        for member in source_tar:
            if member.isfile() and (member.name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md') or member.name.startswith('LICENSES/')):
                destination = args.output/'licenses/TensorFold'/member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source_tar.extractfile(member).read())
    source=args.output/'sources/tensorfold-source.tar.gz';source.parent.mkdir()
    with source.open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as zipped:zipped.write(archive)
    addendum=args.output/'sources/FLASHALIGN-LICENSE.md'
    addendum.write_text((templates/'FLASHALIGN-LICENSE.md').read_text().replace('@SOURCE_ARCHIVE@',source.name).replace('@SOURCE_REVISION@',revision))
    auxiliary_records=[]
    for name,path in auxiliary.items():
        archive_name='sources/auxiliary/'+name
        notice_name=archive_name+'.LICENSE.txt'
        put(path,archive_name);put(auxiliary_notices[name],notice_name)
        auxiliary_records.append({'archive':archive_name,'license':notice_name})
    if auxiliary_records:
        with (args.output/'NOTICE.md').open('a') as notice:
            notice.write('\nAdditional runtime kernel sources and their license/provenance notices are '
                         'included under `sources/auxiliary/` and listed in `manifest.json`.\n')
    transform_path=None
    if args.binary_transform_receipt:
        receipt=json.loads(args.binary_transform_receipt.read_text())
        selected=[]
        for binary in [args.binary,*([args.follower_binary] if args.follower_binary else [])]:
            matches=[row for row in receipt.get('artifacts',[]) if row.get('basename')==binary.name
                     and row.get('source_revision')==revision and row.get('stripped_sha256')==digest(binary)]
            if len(matches)!=1 or matches[0].get('all_allocated_sections_identical') is not True or matches[0].get('program_headers_identical') is not True:
                p.error('binary transform receipt does not verify this exact executable and source revision')
            selected.append(matches[0])
        filtered={**receipt,'artifacts':selected}
        transform_path=args.output/'sources/binary-transform-receipt.json'
        transform_path.write_text(json.dumps(filtered,indent=2)+'\n')
    caps=json.loads(args.capabilities.read_text())
    protected={'version','engine','runtime_version','source_revision','platform','cuda_architectures','binary','follower_binary','source_archive','source_license_notice','binary_transform_receipt','files','auxiliary_sources'}
    if set(caps)&protected:p.error('capabilities file may not override pinned artifact metadata')
    manifest={**caps,'version':2,'engine':'tensorfold-native','runtime_version':args.runtime_version,'source_revision':revision,
              'platform':args.platform,'cuda_architectures':args.cuda_architecture,'binary':'bin/'+args.binary.name,
              'source_archive':str(source.relative_to(args.output)),
              'source_license_notice':'sources/FLASHALIGN-LICENSE.md'}
    if transform_path:manifest['binary_transform_receipt']=str(transform_path.relative_to(args.output))
    if auxiliary_records:manifest['auxiliary_sources']=auxiliary_records
    if args.follower_binary:manifest['follower_binary']='bin/'+args.follower_binary.name
    files={}
    for file in sorted(args.output.rglob('*')):
        if file.is_file():
            name=str(file.relative_to(args.output));files[name]={'sha256':digest(file),'bytes':file.stat().st_size}
            if name.startswith('bin/'):files[name]['executable']=True;file.chmod(0o755)
    manifest['files']=files
    package_spec=importlib.util.spec_from_file_location('_flashalign_bundle_build',
        repo/'hermes-owner-mode/__init__.py',submodule_search_locations=[str(repo/'hermes-owner-mode')])
    package=importlib.util.module_from_spec(package_spec);sys.modules[package_spec.name]=package
    package_spec.loader.exec_module(package)
    importlib.import_module('_flashalign_bundle_build.native_bundle').validate_manifest(manifest)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target=Path(str(args.output)+'.tar.gz')
    if target.exists():p.error('archive already exists')
    with target.open('wb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|') as tar:
                for file in sorted(args.output.rglob('*')):
                    if file.is_file():
                        info=tar.gettarinfo(str(file),arcname=str(file.relative_to(args.output)))
                        info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                        with file.open('rb') as contents:tar.addfile(info,contents)
    if args.catalog_entry:
        entry={'id':args.output.name,'model_types':manifest['model_types'],'platforms':[args.platform],
               'runtime_version': manifest['runtime_version'],
               'cuda_architectures':args.cuda_architecture,
               'url':f'https://github.com/Aevonix/FlashAlign/releases/download/{args.release_tag}/{target.name}',
               'sha256':digest(target),'bytes':target.stat().st_size,
               'files':{str(file.relative_to(args.output)):{'sha256':digest(file),'bytes':file.stat().st_size}
                        for file in sorted(args.output.rglob('*')) if file.is_file()}}
        for field in ('profile_schemas', 'transforms'):
            if field in manifest:entry[field] = manifest[field]
        args.catalog_entry.parent.mkdir(parents=True,exist_ok=True)
        args.catalog_entry.write_text(json.dumps(entry,indent=2)+'\n')
    print(json.dumps({'directory':str(args.output),'archive':str(target),'sha256':digest(target),
                      'bytes':target.stat().st_size,'source_revision':revision}))


if __name__=='__main__':main()
