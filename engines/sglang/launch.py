# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launch a prepared SGLang environment; preserve supplied speculative options."""
import argparse
import os
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--profile',action='append',required=True)
    p.add_argument('--key-file',required=True)
    p.add_argument('--cuda-graphs', action='store_true', help='Enable fused graph execution with per-batch eager fallback')
    p.add_argument('--lifecycle-state', type=Path, help='Private durable profile state directory')
    p.add_argument('--admin-socket', type=Path, help='Owner-UID Unix administration socket')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8030)
    p.add_argument('--draft-model')
    p.add_argument('--draft-algorithm',default='DFLASH')
    p.add_argument('--source',type=Path,help='Prepared SGLang source folder when not installed')
    a,extra=p.parse_known_args()
    if bool(a.lifecycle_state) != bool(a.admin_socket):
        p.error('--lifecycle-state and --admin-socket must be supplied together')
    if a.lifecycle_state:
        a.lifecycle_state.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.environ['FLASHALIGN_LIFECYCLE_STATE'] = str(a.lifecycle_state.resolve())
        os.environ['FLASHALIGN_ADMIN_SOCKET'] = str(a.admin_socket.absolute())
    if extra[:1]==['--']:extra=extra[1:]
    if any(x in extra for x in ('--disable-speculative-decoding','--no-drafts')):
        p.error('Speculative decoding may not be disabled by this launcher')
    os.environ['FLASHALIGN_PROFILES']=','.join(str(Path(x).resolve()) for x in a.profile)
    os.environ['FLASHALIGN_KEY_FILE']=str(Path(a.key_file).resolve())
    if a.cuda_graphs:
        os.environ['FLASHALIGN_CUDA_GRAPHS'] = '1'
    if a.source:
        os.environ['PYTHONPATH']=str(a.source.resolve()/'python')+os.pathsep+os.environ.get('PYTHONPATH','')
    command=[sys.executable,'-m','sglang.launch_server','--model-path',a.model,
             '--host',a.host,'--port',str(a.port),
             '--tool-call-parser','qwen3_coder','--reasoning-parser','qwen3',
             '--cuda-graph-backend-decode','full' if a.cuda_graphs else 'disabled',
             '--cuda-graph-backend-prefill','breakable' if a.cuda_graphs else 'disabled']
    if a.draft_model:
        command+=['--speculative-algorithm',a.draft_algorithm,'--speculative-draft-model-path',a.draft_model]
    command+=extra
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
