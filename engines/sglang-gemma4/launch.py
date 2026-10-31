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
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8030)
    p.add_argument('--draft-model')
    p.add_argument('--draft-algorithm',default='NGRAM')
    p.add_argument('--source',type=Path,help='Prepared SGLang source folder when not installed')
    a,extra=p.parse_known_args()
    if extra[:1]==['--']:extra=extra[1:]
    if any(x in extra for x in ('--disable-speculative-decoding','--no-drafts')):
        p.error('Speculative decoding may not be disabled by this launcher')
    os.environ['FLASHALIGN_PROFILES']=','.join(str(Path(x).resolve()) for x in a.profile)
    os.environ['FLASHALIGN_KEY_FILE']=str(Path(a.key_file).resolve())
    if a.source:
        os.environ['PYTHONPATH']=str(a.source.resolve()/'python')+os.pathsep+os.environ.get('PYTHONPATH','')
    command=[sys.executable,'-m','sglang.launch_server','--model-path',a.model,
             '--host',a.host,'--port',str(a.port),
             '--tool-call-parser','gemma4','--reasoning-parser','gemma4',
             '--cuda-graph-backend-decode','disabled','--cuda-graph-backend-prefill','disabled']
    if a.draft_model or a.draft_algorithm.upper()!='NGRAM':
        p.error('Gemma SGLang requires NGRAM without an external draft model')
    command+=['--speculative-algorithm','NGRAM']
    command+=extra
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
