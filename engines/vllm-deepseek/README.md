# DeepSeek projection profiles on vLLM

This integration runs locally supplied compact or grouped projection profiles
with the pinned stock `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint.
The current DeepSeek starter uses a different operator and requires the
[native matrix backend](../vllm-deepseek-native/README.md).

```sh
bash ./install.sh --engine-only --backend vllm \
  --model-path /models/DeepSeek-V4.1-Flash-1017 \
  --profile /profiles/deepseek-projection --devices 0,1,2,3
```

`flashalign.grouped_projection.v1` profiles may select supported sites with
orthonormal basis ranks from 1 through 128. Omitted sites keep their standard
output. The loader and wizard validate combined MoE output, Engram value slices
and pre-bias image outputs. Unknown targets, unused operators and unsupported
requirements are rejected. Validate each supplied profile against its intended
checkpoint.

## Checkpoint and runtime preparation

Setup installs pinned vLLM 0.32.0 and the native plugin into an owned environment.
This backend requires Python 3.12 or newer. Setup checks that requirement before
writing deployment state or installing dependencies. If no engine interpreter
was selected, it can choose a compatible Python already installed locally.
An explicit incompatible `engine.python` stops setup; no system Python is installed.
The FlashInfer 0.7.1rc5 compatibility dependency is installed into a separate,
hash-checked overlay. Installed vLLM source files are not edited; the process-local
compatibility patch checks the source pins before applying.

The first apply hashes the complete local checkpoint, approximately 510 GB of
reads, and saves a resumable integrity receipt in private deployment state. It
does not download, copy, convert or write model weights. Read-only Hugging Face
snapshot symlinks are supported; both link and resolved target changes invalidate
the receipt. Subsequent checks reuse a receipt only while its file identities
match. Setup creates the receipt automatically.

The launcher passes this receipt explicitly with `--model-receipt` and keeps
persistent replay state at a separate `--replay-file`. Neither file belongs in a
repository or model directory. Reusing a deployment signing key also requires
preserving that deployment's replay state. See [REPLAY.md](REPLAY.md).

## Requirements and defaults

The launcher requires four SM120 GPUs with at least 90 GiB free on each,
TP4/PP1/DP1 and unified Linux cgroup v2 accounting. It uses eager execution,
native DSpark5 and Engram CPU offload. An incompatible active speculative
method or external draft stops setup.

The unattended-start policy requires 365 GiB of available host/cgroup memory.
For a controlled lower-memory deployment, set `engine.allow_untested_memory`
to true in the plan. This exception does not disable the 64 GiB runtime memory
floor: the launcher stops its workers if available memory falls below it.
Reclaimable file cache is accounted for separately from pinned and shared memory.

Default context is 65,536 tokens with two concurrent requests. Setup rejects a
harness context or output budget above the selected engine limit before changing
its route. See [validation](../../docs/validation.md) for tested configurations,
including lower-memory runs.

The standalone launcher requires Linux pidfd and child-subreaper support. It
adopts and reaps native workers if their parent exits, including workers in a
separate process group. Shutdown escalates from termination to kill when needed;
failure to clean up is an error. Process identities are checked before signaling.

## Request behavior

The API accepts the standard FlashAlign exact-body HMAC header, `owner_mode` and
`owner_mode_scale`. Only selected rows receive a transform. Normal rows bypass
transform arithmetic. Model/profile/strength namespaces separate KV and image
cache entries. Cached image encoder features are immutable. DSpark remains
active; image requests use the target image path and text-only draft input.

The launcher enables the native `deepseek_v41` tool and reasoning parsers.
For Hermes, setup adds an exact model override to `low` when its generic effort
default is unsupported, leaving global and other-model settings intact. Existing
supported choices are preserved. Explicitly disabled thinking remains disabled:
the trusted DeepSeek route translates `reasoning_effort=none` to
`chat_template_kwargs.enable_thinking=false` after capability validation and
before the final request is signed. Setup reports this model compatibility rule.
This backend also declares native image input. Setup updates the
selected Hermes model's vision capability and OpenClaw's input descriptor;
explicit text or auxiliary-image routing preferences remain unchanged.

Alternative generation transports, unsupported parallel modes, precomputed
multimodal embeddings, LoRA and external KV transfer fail closed. The host
application retains responsibility for approving contacts and tools. Engine
profiles never grant tool permissions.

`UPSTREAM_PROVENANCE.json` records the source-only upstream handoff. `sha256.json` records
this complete packaged integration, including the wizard launcher.
