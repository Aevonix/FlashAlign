# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launch a managed vLLM instance with mandatory FlashAlign boundaries."""
import argparse
import json
import os
from pathlib import Path
import sys

from .core import load_key,load_profiles
from .arguments import normalize_arguments,option_name,require_full_options


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--model',required=True)
    p.add_argument('--profile',type=Path,action='append',required=True)
    p.add_argument('--key-file',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8030)
    p.add_argument('--draft-model')
    p.add_argument('--draft-method',choices=['eagle','eagle3','dflash','dspark'])
    p.add_argument('--num-speculative-tokens',type=int,default=3)
    p.add_argument('--speculative-config',help='Preserved vLLM speculative configuration JSON')
    args,extra=p.parse_known_args(normalize_arguments(sys.argv[1:] if argv is None else argv))
    forbidden=['--tensor-parallel-size','-tp','--pipeline-parallel-size','-pp','--data-parallel-size','-dp',
               '--api-server-count','-asc','--kv-transfer-config','--enable-prompt-embeds','--middleware',
               '--compilation-config','-cc','--speculative-config','--config','--worker-cls','--worker-extension-cls',
               '--enable-log-requests','--enable-lora','--enable-dbo','--enforce-eager','--no-enforce-eager']
    try:require_full_options(extra,set(forbidden)|set(p._option_string_actions)|{'--max-model-len'})
    except ValueError as error:p.error(str(error))
    if any(option_name(x) in forbidden for x in extra):
        p.error('Unsupported override: parallel engines, external KV/embeddings, middleware, compilation or speculation')
    profiles=load_profiles(args.profile)
    model_ids={profile.meta['model']['id'] for profile in profiles.values()}
    if len(model_ids)!=1:
        p.error('All loaded profiles must target the same model')
    load_key(args.key_file)
    config=json.loads(args.speculative_config) if args.speculative_config else None
    if args.draft_model:
        if config is not None or not args.draft_method:
            p.error('--draft-model requires --draft-method and cannot replace --speculative-config')
        config={'model':args.draft_model,'method':args.draft_method,'num_speculative_tokens':args.num_speculative_tokens}
    if config is not None and config.get('method') not in ('ngram','ngram_gpu','eagle','eagle3','dflash','dspark'):
        p.error('Unsupported active speculative method; it will not be disabled')
    os.environ.update(FLASHALIGN_PROFILES=json.dumps([str(x.resolve()) for x in args.profile]),
                      FLASHALIGN_KEY_FILE=str(args.key_file.resolve()),FLASHALIGN_MODEL_ID=args.model,
                      FLASHALIGN_SECURE_LAUNCH='1',
                      VLLM_NO_USAGE_STATS='1',HF_HUB_DISABLE_TELEMETRY='1',
                      VLLM_PLUGINS='flashalign',VLLM_USE_V2_MODEL_RUNNER='0')
    # Numba otherwise initializes its pool to all host cores and can override
    # Torch's OpenMP limit during ngram warmup. Honor an explicit Numba choice.
    if 'NUMBA_NUM_THREADS' not in os.environ:
        limit=os.environ.get('OMP_NUM_THREADS','1')
        if limit.isdigit() and int(limit)>0:
            os.environ['NUMBA_NUM_THREADS']=limit
    from .plugin import register
    register()
    sys.argv=['vllm','serve',args.model,'--host',args.host,'--port',str(args.port),
              '--enforce-eager','--tensor-parallel-size','1','--pipeline-parallel-size','1',
              '--data-parallel-size','1','--api-server-count','1','--reasoning-parser','qwen3',
              '--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
              '--served-model-name',next(iter(model_ids)),
              '--middleware','flashalign_vllm.middleware.OwnerMiddleware',*extra]
    if config is not None:sys.argv.extend(['--speculative-config',json.dumps(config,separators=(',',':'))])
    from vllm.entrypoints.cli.main import main as serve
    serve()


if __name__=='__main__':main()
