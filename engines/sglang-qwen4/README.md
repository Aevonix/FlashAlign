# Aevonix FlashAlign for Qwen Flash Next

Apply a FlashAlign profile per request with SGLang. This backend supports
Qwen4Exp target models and their native MTP proposer. It works directly through
the engine API and does not require an agent harness.

The backend is pinned to SGLang v0.5.20, commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`. Supported model and quantization
fingerprints are listed in `backend.json`. The profile declares its compatible
models separately from the checkpoint pair used to create it.

## Setup

Use an isolated environment with `requirements.txt` and a clean checkout of the
pinned SGLang revision. Preparation creates a new source directory; it does not
patch a running engine.

```sh
python prepare.py --source /engines/sglang-upstream --output /engines/flashalign-sglang
python launch.py \
  --source /engines/flashalign-sglang \
  --model /models/qwen-flash-next-fp8 \
  --profile /profiles/qwen-flash-next \
  --key-file /private/flashalign.key \
  --tensor-parallel 2 --host 127.0.0.1 --port 8132 \
  -- --context-length 65536 \
     --max-running-requests 2 --mem-fraction-static 0.85 --enable-cache-report
```

A profile directory contains `profile.json`, `profile.safetensors` and its
source-model license. The signing key must be an owned private regular file of
exactly 32 random bytes. Models and profiles must already be present.

The default context is 65,536 tokens, tested with serial requests on two
95 GiB RTX PRO 6000 Blackwell Max-Q GPUs. See
[validation](../../docs/validation.md) for setup, long-input and harness results.

The launcher preserves native MTP and uses TP/EP two with PLE tables in pinned
host memory. The qualified path uses eager execution. Unsupported speculative
algorithms or execution layouts are rejected instead of being disabled.

## Requests

Send `owner_mode` and `owner_mode_scale` in `/v1/chat/completions`,
`/v1/completions` or `/generate`, with `X-Owner-Mode-Auth` signed over the raw
body. `X-Owner-Mode-Effective` reports the selected profile or `off`. Missing,
expired, replayed or invalid authentication serves the stock model.

Profile discovery is available at `/v1/owner-mode/profiles`. Both cache modes
include the pinned model identity. Owner caches additionally include the
profile digest and exact strength. Caller-provided cache salts are wrapped.

The profile covers target and native-MTP attention/MLP writers, token embedding
before image merging, and the PLE value projection before its nonlinear
operations. Stock rows bypass transform arithmetic. Image bytes remain part of
the authenticated request body.

The current scope is one HTTP worker, PP/DP/CP one and TP/EP two, without
CUDA graphs, torch.compile, LoRA, hierarchical cache or session-cache reuse.
The backend fails startup for unsupported profile requirements or missing model
hooks. Generic engine access does not grant tool permissions; those remain with
the calling application.
