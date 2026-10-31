# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launch a prepared SGLang environment; preserve supplied speculative options."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--profile',action='append',required=True)
    p.add_argument('--key-file',required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8030)
    p.add_argument('--draft-model')
    p.add_argument('--draft-algorithm',default='EAGLE',choices=['EAGLE','NEXTN'])
    p.add_argument('--tensor-parallel',type=int,default=2,choices=[2])
    p.add_argument('--source',type=Path,help='Prepared SGLang source folder when not installed')
    a,extra=p.parse_known_args()
    if extra[:1]==['--']:extra=extra[1:]
    if any(x in extra for x in ('--disable-speculative-decoding','--no-drafts')):
        p.error('Speculative decoding may not be disabled by this launcher')
    model=Path(a.model).resolve()
    metadata=json.loads((Path(__file__).resolve().parent/'backend.json').read_text())
    actual={name:hashlib.sha256((model/name).read_bytes()).hexdigest() for name in ('config.json','model.safetensors.index.json')}
    matched=[entry for entry in metadata['allowed_model_metadata'] if entry['sha256']==actual]
    if len(matched)!=1:p.error('Model metadata does not match a pinned FlashAlign Qwen4 checkpoint')
    os.environ['FLASHALIGN_MODEL_SCOPE']=hashlib.sha256(json.dumps(matched[0],sort_keys=True).encode()).hexdigest()
    os.environ['FLASHALIGN_PROFILES']=','.join(str(Path(x).resolve()) for x in a.profile)
    os.environ['FLASHALIGN_KEY_FILE']=str(Path(a.key_file).resolve())
    if a.source:
        os.environ['PYTHONPATH']=str(a.source.resolve()/'python')+os.pathsep+os.environ.get('PYTHONPATH','')
    command=[sys.executable,'-m','sglang.launch_server','--model-path',a.model,
             '--host',a.host,'--port',str(a.port),
             '--tp-size',str(a.tensor_parallel),'--ep-size',str(a.tensor_parallel),
             '--ple-offload-embedding','--ple-offload-backend','pinned',
             '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3',
             '--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled']
    command+=['--speculative-algorithm',a.draft_algorithm,
              '--speculative-draft-model-path',a.draft_model or a.model,
              '--speculative-eagle-topk','1','--speculative-num-steps','3',
              '--speculative-num-draft-tokens','4']
    command+=extra
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
