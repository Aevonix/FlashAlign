# FlashAlign for vLLM

Experimental runtime adapter for **vLLM 0.32.0**, pinned to upstream commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`. This package does not create profiles
or include model weights. It loads the same FlashAlign runtime artifacts used
by the other backends.

## Install in an isolated environment

```sh
python3 -m venv .venv
.venv/bin/python -m pip install vllm==0.32.0 /path/to/flashalign-vllm
```

Start with an operator-provisioned private 32-byte signing key and local profile:

```sh
.venv/bin/python -m flashalign_vllm \
  --model /models/qwen38-27b --profile /profiles/qwen38-27b-v1 \
  --key-file /private/engine.key --host 127.0.0.1 --port 8030 \
  --max-model-len 65536 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":3}'
```

Preserve an existing speculative configuration with `--speculative-config` and
its exact JSON. `--draft-model` is also accepted with an explicit
`--draft-method` (`eagle`, `eagle3`, `dflash` or `dspark`). Unsupported active
methods stop startup. The launcher never adds a flag that disables speculation.

## Supported scope

This adapter supports the dense Qwen3.5/3.8 architecture and standard
projection profiles at embedding, attention-output and MLP-output sites. It
requires one GPU, one API process and eager execution. Disabling CUDA graphs
is an explicit performance limitation; speculative decoding remains available.
Native MTP is not supported by the current main-model-only Qwen starter profile.
Other model families, tensor/pipeline/data parallelism, microbatch overlap,
combined LoRA adapters, external KV transfer and client-supplied prompt embeddings
are rejected.

The HTTP boundary validates the raw signed body before setting any worker
selection. It strips forged internal fields and authentication headers, adds the
actual `X-Owner-Mode-Effective` response header, and gives each profile/digest/
strength its own prefix-cache namespace. Standard requests also have a separate
namespace. Only Chat Completions and Completions accept POST requests; alternate
generation transports and administrative mutation routes are closed. Invalid or missing authentication is standard mode.

Profiles load in the engine workers. Hooks modify only selected batch rows at
the actual embedding/output modules. The base weights stay unchanged; standard
rows bypass all transform arithmetic. External draft models keep their ordinary
weights; the target model verifies draft tokens under the selected profile.

The tested configuration uses Qwen3.8-27B BF16, the bundled 129-site profile
and ngram speculation on one RTX PRO 6000. See
[validation](../../docs/validation.md) for protocol, answer and long-input results.

Wizard defaults are a 65,536-token context, up to four requests, an 8,192-token
prefill batch and GPU-memory fraction 0.85. Four simultaneous maximum-length
requests remain untested. The full `qwen3_5` model accepts images;
`qwen3_5_text` is text only.

Installing the package leaves plain vLLM unchanged until profiles are explicitly
configured through this launcher. Usage telemetry is disabled by the launcher.

The extension uses vLLM's [general plugin mechanism](https://docs.vllm.ai/en/v0.32.0/design/plugin_system/).
Its runner integration is pinned to the [0.32.0 GPU runner](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_model_runner.py).
The cache namespace uses vLLM's [request salt in block hashes](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_utils.py).
