# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Make an isolated exact-version SGLang source copy with FlashAlign support."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PIN='94602c9c2b7cbdb8efd5c52802dac6a1c180089e'
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists():raise ValueError('Output directory must be new')
    import subprocess
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()
    if actual!=PIN:raise ValueError('SGLang source revision differs from supported pin')
    if subprocess.check_output(['git','status','--porcelain'],cwd=a.source,text=True).strip():
        raise ValueError('SGLang source must be a clean pinned checkout')
    shutil.copytree(a.source,out,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    edits=[]
    def edit(name,before,after):
        path=out/name;old=path.read_bytes();text=old.decode()
        if text.count(before)!=1:raise ValueError('Unsupported source preimage: '+name)
        path.write_text(text.replace(before,after))
        edits.append({'path':name,'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    edit('python/sglang/srt/entrypoints/http_server.py',
         'app.router.route_class = ORJSONRoute',
         'app.router.route_class = ORJSONRoute\nfrom sglang.srt.flashalign.http import Middleware as FlashAlignMiddleware\napp.add_middleware(FlashAlignMiddleware)')
    edit('python/sglang/srt/entrypoints/http_server.py',
         '    # Launch subprocesses\n',
         '    from sglang.srt.flashalign.runtime import check_server_args\n    check_server_args(server_args)\n    # Launch subprocesses\n')
    edit('python/sglang/srt/model_executor/forward_batch_info.py',
         '    # For dumper: request IDs for cross-step sequence tracking',
         '    flashalign_cache_salts: Optional[List[str]] = None\n    # For dumper: request IDs for cross-step sequence tracking')
    edit('python/sglang/srt/model_executor/forward_batch_info.py',
         '            lora_ids=[req.lora_id for req in batch.reqs],',
         '            lora_ids=[req.lora_id for req in batch.reqs],\n            flashalign_cache_salts=[req.cache_salt for req in batch.reqs],')
    edit('python/sglang/srt/model_executor/model_runner.py',
         '        self.model = loaded.model\n',
         '        self.model = loaded.model\n        from sglang.srt.flashalign.runtime import install\n        install(self)\n')
    edit('python/sglang/srt/model_executor/model_runner.py',
         '    def forward(\n        self,\n        forward_batch: ForwardBatch,',
         '    from sglang.srt.flashalign.runtime import owner_forward\n\n    @owner_forward\n    def forward(\n        self,\n        forward_batch: ForwardBatch,')
    shutil.copytree(ROOT/'flashalign_sglang',out/'python/sglang/srt/flashalign')
    receipt={'upstream':'https://github.com/sgl-project/sglang','tag':'v0.5.20','commit':PIN,'files':edits,
             'status':'candidate; live model tests required','restrictions':['single GPU/worker','eager execution','pinned Gemma4 target','no HiCache or session cache','NGRAM speculation with non-overlapping scheduling']}
    (out/'flashalign-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'source':str(out),'commit':PIN,'edits':len(edits)}))


if __name__=='__main__':main()
