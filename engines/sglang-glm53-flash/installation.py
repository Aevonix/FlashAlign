# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Validate an owned, prepared SGLang tree before launching this kit."""
import hashlib
import json
from pathlib import Path

PIN = '94602c9c2b7cbdb8efd5c52802dac6a1c180089e'
ROOT = Path(__file__).resolve().parent


def validate_prepared_source(source):
    source = Path(source).resolve()
    receipt_path = source/'flashalign-build.json'
    if not receipt_path.is_file():
        raise ValueError('SGLang source has not been prepared for this FlashAlign kit')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('commit') != PIN:
        raise ValueError('Prepared SGLang revision differs from this kit')
    final = {row['path']:row['after_sha256'] for row in receipt.get('files', [])}
    expected = {'python/sglang/srt/entrypoints/http_server.py',
                'python/sglang/srt/model_executor/model_runner.py',
                'python/sglang/srt/model_executor/forward_batch_info.py',
                'python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py'}
    if set(final) != expected:
        raise ValueError('Prepared SGLang hook receipt is incomplete')
    for name, digest in final.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared SGLang hook changed: '+name)
    for path in sorted((ROOT/'flashalign_sglang').glob('*.py')):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        target = source/'python/sglang/srt/flashalign'/path.name
        if (receipt.get('flashalign_code_sha256', {}).get(path.name) != digest
                or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
            raise ValueError('Prepared FlashAlign runtime differs from this kit: '+path.name)
    for name in ('expected-sites.json', 'model-metadata.json'):
        digest = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if (receipt.get('flashalign_data_sha256', {}).get(name) != digest
                or hashlib.sha256((source/'python/sglang/srt'/name).read_bytes()).hexdigest() != digest):
            raise ValueError('Prepared FlashAlign metadata differs from this kit: '+name)
    return receipt
