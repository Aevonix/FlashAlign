# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
# Includes a vLLM Apache-2.0 adaptation; see the package NOTICE.md.
"""Process-only backport of upstream PR60762; never changes installed files."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

APPLIED=False


def apply_geometry():
    global APPLIED
    if APPLIED:return
    if importlib.metadata.version('vllm')!='0.32.0':raise RuntimeError('Expected vLLM 0.32.0')
    import flashinfer
    if flashinfer.__version__!='0.7.1rc5':raise RuntimeError('SM120 geometry requires the isolated FlashInfer 0.7.1rc5 overlay')
    from flashinfer.mla import supported_sparse_mla_sm120_configs
    config=supported_sparse_mla_sm120_configs().get('dsv4')
    if config is None or not getattr(config,'page_block_size_is_runtime',False):
        raise RuntimeError('FlashInfer lacks runtime sparse-MLA page sizes')
    import vllm
    pins=json.loads((Path(__file__).parent/'source-pins.json').read_text())
    for name in ['v1/attention/backends/mla/indexer.py','models/deepseek_v41/nvidia/flashinfer_sparse.py']:
        if hashlib.sha256((Path(vllm.__file__).parent/name).read_bytes()).hexdigest()!=pins[name]:
            raise RuntimeError('Unrecognized geometry source: '+name)
    from vllm.platforms import current_platform
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError('This private geometry backport is restricted to SM120')
    from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import DeepseekV4FlashInferMLASparseBackend
    # Exact declarations from PR60762 on SM12x. Leave SWA and compressor
    # geometry untouched, and keep the engine's usual cache bookkeeping.
    for backend in [DeepseekV41IndexerBackend,DeepseekV4FlashInferMLASparseBackend]:
        backend.get_supported_kernel_block_sizes=staticmethod(lambda kv_cache_spec=None:[64])
    APPLIED=True
