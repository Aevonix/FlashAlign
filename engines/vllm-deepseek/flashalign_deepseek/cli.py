# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Start an isolated, explicitly experimental DeepSeek engine with its gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from .core import load_key,load_profiles
from .arguments import normalize_arguments,option_name,require_full_options


def validate_model(directory,receipt=None):
    root=Path(directory)
    if not root.is_dir():raise ValueError('This research adapter requires the pinned local stock checkpoint')
    pins=json.loads((Path(__file__).parent/'model-pins.json').read_text())
    if receipt is not None:
        from .verify_checkpoint import verify
        verify(root,receipt,pins)
        return pins['model']
    for name,digest in pins['metadata_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Stock checkpoint metadata differs: '+name)
    for name,info in pins['shards'].items():
        path=root/name
        if path.stat().st_size!=info['bytes'] or path.with_suffix(path.suffix+'.verified').read_text().strip()!=info['sha256']:
            raise ValueError('Stock checkpoint inherited integrity receipt differs: '+name)
    # Existing verified receipts are accepted; this does not claim a fresh
    # 510 GB payload rehash on each launch.
    return pins['model']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--model',required=True);p.add_argument('--profile',type=Path,action='append',required=True)
    p.add_argument('--model-receipt',type=Path)
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8041)
    p.add_argument('--replay-file',type=Path)
    p.add_argument('--allow-research',action='store_true')
    p.add_argument('--speculative-config')
    args,extra=p.parse_known_args(normalize_arguments(sys.argv[1:] if argv is None else argv))
    blocked={'--tensor-parallel-size','-tp','--pipeline-parallel-size','-pp','--data-parallel-size','-dp',
        '--enable-expert-parallel','--enable-dbo','--worker-cls','--worker-extension-cls','--api-server-count',
        '--middleware','--enable-prompt-embeds','--enable-mm-embeds','--enable-lora','--kv-transfer-config','--config','--compilation-config',
        '--served-model-name','--tokenizer','--tokenizer-mode','--reasoning-parser','--trust-remote-code','--load-format','--engram-config',
        '--enable-log-requests','--enable-log-outputs','--block-size','--speculative-config',
        '--enforce-eager','--no-enforce-eager','--hf-overrides','--hf-config-path','--quantization','-q','--dtype'}
    try:require_full_options(extra,blocked|set(p._option_string_actions)|{'--max-model-len'})
    except ValueError as error:p.error(str(error))
    if any(option_name(x) in blocked for x in extra):p.error('Unsupported override of the secured research configuration')
    if not args.allow_research:p.error('This native integration requires explicit --allow-research')
    # Set before profile loading imports torch and before host allocator init.
    os.environ.setdefault('PYTORCH_ALLOC_CONF','pinned_max_round_threshold_mb:128')
    model_id=validate_model(args.model,args.model_receipt)
    load_key(args.key_file);profiles=load_profiles(args.profile,allow_research=True)
    speculation=json.loads(args.speculative_config) if args.speculative_config else {'method':'dspark','model':args.model,'num_speculative_tokens':5}
    if speculation.get('method')!='dspark':p.error('Only native DSpark is currently qualified for research; it will not be disabled')
    os.environ.update(FLASHALIGN_DEEPSEEK_PROFILES=json.dumps([str(x.resolve()) for x in args.profile]),
        FLASHALIGN_DEEPSEEK_KEY_FILE=str(args.key_file.resolve()),FLASHALIGN_DEEPSEEK_MODEL_ID=model_id,
        FLASHALIGN_DEEPSEEK_ALLOW_RESEARCH='1',FLASHALIGN_DEEPSEEK_SECURE_LAUNCH='1',
        FLASHALIGN_DEEPSEEK_SM120_GEOMETRY='64',
        VLLM_PLUGINS='flashalign_deepseek',VLLM_USE_V2_MODEL_RUNNER='1',
        VLLM_NO_USAGE_STATS='1',HF_HUB_DISABLE_TELEMETRY='1')
    os.environ.pop('FLASHALIGN_PROFILES',None)
    if args.replay_file is not None:
        os.environ['FLASHALIGN_REPLAY_STORE']=str(args.replay_file.expanduser().absolute())
    from .plugin import register
    register()
    sys.argv=['vllm','serve',args.model,'--host',args.host,'--port',str(args.port),
        '--tensor-parallel-size','4','--pipeline-parallel-size','1','--data-parallel-size','1','--api-server-count','1',
        '--enforce-eager','--tokenizer-mode','deepseek_v41','--reasoning-parser','deepseek_v41',
        '--block-size','64',
        '--served-model-name',next(iter(profiles.values())).meta['model']['id'],
        '--safetensors-load-strategy','lazy','--engram-config','{"cpu_offload":true,"dp_shared_memory":false,"use_thp":false}',
        '--speculative-config',json.dumps(speculation,separators=(',',':')),
        '--middleware','flashalign_deepseek.middleware.OwnerMiddleware',*extra]
    from vllm.entrypoints.cli.main import main as serve
    serve()
