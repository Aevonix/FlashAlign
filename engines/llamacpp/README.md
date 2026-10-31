# FlashAlign llama.cpp runtime

This runtime adds a native `flashalign_projection` adapter to the pinned
llama.cpp v0.8.0, revision `8642541b255bfb5dc39cd351a178be09013a55d6`.
This is the upstream base version at that exact commit. Upstream defaults its
development-build flag to on, so a built binary can report `0.8.0-dev`.
The installer records the commit and binary hash in `installation.json`.
The adapter supports the Qwen dense `qwen35` and Gemma 4 `gemma4` graph families.
Qwen uses embedding, attention and MLP output sites. Gemma uses attention and
dense/shared MLP output sites. DeepSeek and GLM profile operators are unsupported.
The qualified input path is text; image input has not been validated here.

The adapter operates on the computed matrix output, before residual addition.
For Gemma, it runs before the following normalization and leaves routed experts
unchanged. The GGUF adapter container carries the profile's Q factors, their
negative transpose and each site's positive FP32 strength. It applies
`y - owner_mode_scale * site_scale * (y @ Q.T) @ Q`. Source weights are read-only.
No checkpoint pair, fitting procedure or private derivation data is required.

## Build without starting a service

The installer creates a new owned source checkout and build directory. It
pins the upstream commit, checks the patch hash, builds the native runtime and
runs its projection arithmetic test. Choose a fresh destination.

```sh
python install.py --state /private/flashalign-llamacpp --cuda on
```

Dependencies are Git, CMake, a C++17 toolchain, and CUDA when selected. The Python
launcher and profile converter use the dependencies in `requirements.txt`; the installer prepares a private Python environment for them. `installation.json` records the
binary and bundled GGUF Python module location.

## Qwen3.8-27B checkpoint

The qualified Qwen path uses `Qwen3.8-27B-Q8_0.gguf` from
[`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/a316ea527fde32c12487be7574e2db5ac2ad2a7e)
at revision `a316ea527fde32c12487be7574e2db5ac2ad2a7e`.
Its external `draft-simple` model is `Qwen3.5-0.8B-Q4_0.gguf` from
[`ggml-org/Qwen3.5-0.8B-GGUF`](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/tree/807dfd9a6fadbf703207aac3bfb0b2b3040d077c)
at revision `807dfd9a6fadbf703207aac3bfb0b2b3040d077c`. This is the small
draft model used with the Qwen3.8 main model during qualification. Its SHA-256 is
`57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`.
The [model download guide](../../docs/model-downloads.md#llamacpp-checkpoints)
has both download commands and a complete setup example.

## Gemma 4 checkpoint

See the [model download guide](../../docs/model-downloads.md) for Qwen and Gemma
commands and the alternative engine formats.

The managed Gemma path uses the official
[`ggml-org/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/tree/9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2)
Q8_0 checkpoint at revision `9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2`.
Download the weights separately:

```sh
hf download ggml-org/gemma-4-26B-A4B-it-GGUF \
  --revision 9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2 \
  gemma-4-26B-A4B-it-Q8_0.gguf --local-dir /models/gemma4
```

Setup checks the file size and a hash covering the GGUF metadata and tensor
directory. This identifies the declared checkpoint layout; it is not a full
weight-payload hash. The model's full SHA-256 is
`b5108fd13147d1c866bb595295bc9d56f5fe744d7209f18421031d0cc47009c6`.
The bundled profile remains unchanged. Custom profiles targeting the same
declared base can use the adapter if their sites and operator are supported.

From the FlashAlign checkout, prepare and activate a standalone Gemma server:

```sh
bash ./install.sh --engine-only --backend llamacpp \
  --model-path /models/gemma4/gemma-4-26B-A4B-it-Q8_0.gguf
```

New Gemma deployments use `ngram-simple` without another model download.
For the official MTP draft, download `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` from
the same pinned repository and pass its local path with `--drafter`.
Setup preserves an existing supported speculative configuration.

## Start a direct engine endpoint

The model must be a compatible local GGUF. Use a private 32-byte key file with
mode 0600. The normal FlashAlign setup flow creates this deployment key.

```sh
/private/flashalign-llamacpp/venv/bin/python launcher.py \
  --binary /private/flashalign-llamacpp/build/bin/llama-server \
  --gguf-python /private/flashalign-llamacpp/source/gguf-py \
  --model /models/qwen38-27b.gguf \
  --profile /profiles/qwen38-27b-v1 \
  --key-file /private/engine.key --state /private/flashalign-llamacpp/runtime \
  --host 127.0.0.1 --port 8030 \
  --draft-model /models/compatible-draft.gguf \
  --engine-args /private/engine-args.json
```

`engine-args.json` is a JSON string array of native server arguments. For
example, `["--ctx-size", "131072", "--parallel", "2", "--n-gpu-layers", "99",
"--spec-type", "draft-simple", "--spec-draft-n-max", "4"]`. Preserve the
existing engine's speculative type, draft model and tuning arguments when
migrating. The launcher does not replace these with a disabled draft path.
Model, listener and adapter overrides in this array are rejected.

For Gemma, use `--spec-type draft-mtp` with its official draft, or
`--spec-type ngram-simple` without `--draft-model`. Qwen uses `draft-simple`.
The launcher rejects methods that have not been qualified for the selected
model family.

The native listener is loopback-only on a separate port. The external endpoint
accepts the FlashAlign signed request contract and always overrides caller
adapter fields. Missing, expired, invalid or replayed signatures use standard
mode. The key and headers are not logged. A bounded private nonce store prevents replay across gateway restarts and fails closed if storage is unavailable. Profile discovery is available at
`GET /v1/owner-mode/profiles`.

The endpoint does not require Hermes or OpenClaw. Another harness can use the
same signed Chat Completions API. Requests use ordinary OpenAI-compatible
messages plus `owner_mode` and `owner_mode_scale`; identity verification belongs
to the caller, and the shared deployment key is the trust boundary.

## Cache and execution isolation

Requests with different adapters or strengths are placed in separate native
batches. The live slot and shared in-memory prompt cache both carry the exact
adapter identity and strength. Zero-scaled adapters are omitted from graph
construction, so standard requests perform no profile arithmetic. A tied
vocabulary head is excluded from the embedding-only projection.

Context shifting retains that identity while replacing token bookkeeping.
Native disk slot save and restore are disabled when FlashAlign profiles are
loaded; the in-memory cache remains available in both modes.

The launcher exposes Chat Completions, Completions, model discovery, health and
metrics. Native slot-save and adapter-administration endpoints are not exposed.
Profile tensors and the loaded model stay immutable for the process lifetime;
a restart resets its in-memory caches. Local users who can bypass the loopback
listener are inside the deployment's trust boundary.

## Evidence and limits

The native operator tests cover non-unit site strengths, standard-mode bypass,
draft-model separation and cache identity during allocation and context shift.
Python checks cover the converter, request gate and speculative configuration.

Gemma Q8_0 passed 32 direct checks with MTP and 32 with ngram decoding. These
cover signed activation, invalid/replayed signatures, twelve stock/profile
answer fixtures, cache separation by strength, concurrent requests, streaming
and long inputs. Separate copy probes confirmed accepted ngram draft tokens in
both modes. Qwen Q8_0 passed 32 direct checks with its external draft.
See [validation](../../docs/validation.md) for the longer-input and native
harness results, including strict-answer limits. These runs do not establish
broad model-quality equivalence.

The converter accepts standard projection profiles with finite positive site
strengths and rejects unsupported transforms. Other model families, GGUF
quantizations and speculative methods require their own qualification.

The native source and modifications retain the upstream MIT terms in
`LICENSE.llama.cpp`. FlashAlign's distribution retains its applicable Aevonix
license and all upstream notices. No model weights ship with this runtime.

Managed setup defaults to 65,536 tokens per sequence: 131,072 total context
split across two parallel slots, with a 2 GiB in-memory prompt cache. Cache
entries are evicted when they no longer fit; eviction causes a fresh prefill,
not reuse across modes. Explicit context, parallel and cache overrides remain
available through `engine.arguments`.
