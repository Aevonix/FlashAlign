# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned public native DeepSeek runtime contract."""

IMAGE = 'ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1b10ec7b853a495e66b7dbf15ff54f62741e444dea0b44e16736a10a9'
MODEL_ID = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
MODEL_REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
MODEL_MOUNT = '/models/DeepSeek-V4.1-Flash-1017'
DEFAULT_SPECULATION = {
    'method': 'dspark', 'attention_backend': 'B12X',
    'draft_sample_method': 'probabilistic', 'rejection_sample_method': 'standard',
    'num_speculative_tokens': 7, 'draft_tensor_parallel_size': 4,
    'enable_adaptive_verification': True, 'adaptive_verification_cost_scale': 1.0,
}
DEFAULT_ARGUMENTS = [
    '--async-scheduling', '--attention-backend', 'B12X', '--block-size', '256',
    '--compilation-config', '{"cudagraph_mode":"FULL_AND_PIECEWISE"}',
    '--decode-context-parallel-size', '1', '--decode-refill-target', 'auto',
    '--default-chat-template-kwargs', '{"thinking":true,"reasoning_effort":"high"}',
    '--dtype', 'bfloat16', '--enable-auto-tool-choice', '--enable-chunked-prefill',
    '--enable-force-include-usage', '--enable-prefix-caching', '--enable-prompt-tokens-details',
    '--enable-request-id-headers',
    '--engram-config', '{"cpu_offload":false,"table_memory":"ram","disk_resident_scales":false,"projection_tp":false}',
    '--generation-config', 'vllm', '--gpu-memory-utilization', '0.93',
    '--jit-monitor-mode', 'warn', '--kv-cache-dtype', 'fp8', '--linear-backend', 'b12x',
    '--load-format', 'instanttensor', '--max-cudagraph-capture-size', '128',
    '--max-model-len', '-1', '--max-num-batched-tokens', '4096', '--max-num-seqs', '8',
    '--max-parallel-prefills', '1', '--moe-backend', 'b12x',
    '--override-generation-config', '{"temperature":1.0,"top_p":1.0}',
    '--pipeline-parallel-size', '1', '--prefill-compute-share', '0.4',
    '--prefill-policy', 'round-robin', '--prefill-schedule-interval', '1',
    '--reasoning-parser', 'deepseek_v41', '--safetensors-load-strategy', 'lazy',
    '--no-scheduler-reserve-full-isl', '--swa-block-size', '128',
    '--tensor-parallel-size', '4', '--tokenizer-mode', 'deepseek_v41',
    '--tool-call-parser', 'deepseek_v41', '--no-enable-log-requests',
]
ENVIRONMENT = {
    'PYTHONPATH': '/fa-package',
    'FLASHALIGN_NATIVE_ENGINE_ROOT': '/opt/venv/lib/python3.12/site-packages/vllm',
    'VLLM_PLUGINS': 'flashalign_native', 'VLLM_USE_V2_MODEL_RUNNER': '1',
    'VLLM_USE_BREAKABLE_CUDAGRAPH': '0', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1', 'DO_NOT_TRACK': '1',
    'HF_HOME': '/cache/huggingface', 'XDG_CACHE_HOME': '/cache',
    'B12X_COMPILE_CACHE_DIR': '/cache/b12x/compile',
    'B12X_CUTE_COMPILE_CACHE_DIR': '/cache/b12x/cute',
    'SPARKINFER_COMPILE_CACHE_DIR': '/cache/b12x/compile',
    'CUTE_DSL_CACHE_DIR': '/cache/cute-dsl', 'CUDA_CACHE_PATH': '/cache/cuda',
    'TRITON_CACHE_DIR': '/cache/triton', 'TORCHINDUCTOR_CACHE_DIR': '/cache/torchinductor',
    'TORCH_EXTENSIONS_DIR': '/cache/torch-extensions', 'INSTANTTENSOR_BACKEND': 'BUFFERED',
    'PYTHONDONTWRITEBYTECODE': '1',
}
