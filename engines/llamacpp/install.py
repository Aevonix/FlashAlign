# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Build the pinned FlashAlign llama.cpp runtime in an owned deployment folder."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PIN='8642541b255bfb5dc39cd351a178be09013a55d6'
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--cuda',choices=['auto','on','off'],default='auto')
    p.add_argument('--jobs',type=int,default=8)
    a=p.parse_args()
    if not 1<=a.jobs<=64:raise ValueError('jobs must be in 1..64')
    state=a.state.expanduser().resolve();state.mkdir(parents=True,exist_ok=True,mode=0o700)
    source=state/'source';build=state/'build'
    if source.exists():raise ValueError('Choose a fresh state directory; source already exists')
    patch=ROOT/'flashalign.patch';receipt=json.loads((ROOT/'patch.json').read_text())
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=receipt['patch_sha256']:raise ValueError('Patch checksum mismatch')
    source.mkdir(mode=0o700)
    commands=[['git','init',str(source)],['git','-C',str(source),'remote','add','origin','https://github.com/ggml-org/llama.cpp.git'],
              ['git','-C',str(source),'fetch','--depth','1','origin',PIN],['git','-C',str(source),'checkout','--detach','FETCH_HEAD'],
              ['git','-C',str(source),'apply','--check',str(patch)],['git','-C',str(source),'apply',str(patch)]]
    use_cuda=a.cuda=='on' or (a.cuda=='auto' and shutil.which('nvcc') is not None)
    commands.extend([[sys.executable,'-m','venv',str(state/'venv')],
                     [str(state/'venv/bin/python'),'-m','pip','--isolated','install','--index-url','https://pypi.org/simple','-r',str(ROOT/'requirements.txt')],
                     ['cmake','-S',str(source),'-B',str(build),'-DGGML_CUDA='+('ON' if use_cuda else 'OFF'),
                      '-DLLAMA_BUILD_TESTS=ON','-DLLAMA_BUILD_EXAMPLES=OFF','-DLLAMA_BUILD_TOOLS=ON','-DLLAMA_CURL=OFF'],
                     ['cmake','--build',str(build),'--target','llama-server','test-flashalign-output','-j',str(a.jobs)],
                     [str(build/'bin/test-flashalign-output')]])
    with (state/'build.log').open('w') as log:
        for command in commands:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    binary=build/'bin/llama-server'
    result={'engine':'llamacpp','revision':PIN,'binary':str(binary),'gguf_python':str(source/'gguf-py'),'python':str(state/'venv/bin/python'),
            'cuda':use_cuda,'patch_sha256':receipt['patch_sha256'],
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'status':'prepared','started':False}
    (state/'installation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
