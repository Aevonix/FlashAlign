# DeepSeek V4.1 Flash on native vLLM V2

This experimental integration accepts the pinned stock
`deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint and a locally supplied FlashAlign
profile. It contains no model weights, profile tensors or profile-creation tools.
The wizard selects it by model family and exact checkpoint metadata:

```sh
bash ./install.sh --engine-only --backend vllm \
  --model-path /models/DeepSeek-V4.1-Flash-1017 --devices 0,1,2,3
```

The included 85-site DeepSeek profile is accepted without rewriting its manifest,
basis tensors, strengths or digest. It has passed bounded engine, cache,
image-routing and direct-gateway checks; its quality status remains experimental.
Operator-supplied `flashalign.grouped_projection.v1` profiles are also supported;
the validator checks their typed sites, shared operators, Engram slices and image
bias handling. A valid artifact is not proof of edited-checkpoint parity or model
quality. Profile-specific evidence belongs with the supplied profile.

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

## Qualified scope and experimental memory policy

The native engine was exercised on four SM120 GPUs with approximately 95 GiB
VRAM each, TP4/PP1/DP1, eager execution, native DSpark5, and Engram CPU offload.
The wizard preserves that speculative configuration. An incompatible active
method or external draft stops setup instead of being disabled or replaced.
This package does not extend that qualification to other hardware or parallelism.

Before activation, the wrapper requires four SM120 GPUs with at least 90 GiB
free on each. Its conservative unattended-start policy requires 365 GiB of
available host/cgroup headroom. That policy is **not a measured minimum**: the
successful reference run started with 367.65 GiB host `MemAvailable`, while its
initial cgroup headroom was not recorded. The observed steady host usage was
approximately 202 GiB. Lower-memory machines remain unqualified. A controlled
experiment can explicitly set `engine.allow_untested_memory` to true in its plan;
the result continues to report that exception. It is not a support claim.

The wrapper monitors host and cgroup headroom and stops its child workers below
64 GiB. Unified Linux cgroup v2 accounting is required. Cgroup reports retain raw current/max, file, shmem and inactive-file
counters. Working headroom may credit ordinary inactive file cache, bounded by
`min(inactive_file, file - shmem - unevictable - dirty - writeback, current)`
with negative values clamped to zero. Pinned and shared-memory allocations earn
no credit. This prevents the preceding checkpoint hash pass from being counted
entirely as unreclaimable memory. Host `MemAvailable` is an independent cap;
the launcher never drops page cache. This floor cannot be disabled by the experimental override. The monitor
is a safety measure, not a guarantee against every allocation failure. Default
context is 65,536 tokens with two concurrent requests, matching the native
engine context ceiling exercised separately. Other resource settings need their own check.
Setup rejects an existing harness context or output budget above that selected
engine limit before changing its route. It preserves the configured values and
does not automatically enlarge this experimental preset.

The wizard and this package completed full-model activation and shutdown checks
with the 85-site profile under the explicit lower-memory test setting. That
result does not remove the default 365 GiB startup policy or qualify every
lower-memory host. Native harness coverage is listed in the
[starter pack](../../profiles/README.md).

Verified and Always On native Hermes checks completed with the profile active.
All 14 isolated OpenClaw cases passed protocol and strict-answer checks,
including image authorization, both tool-inheritance settings and same-session
CLI behavior. Native DSpark stayed active during these checks.

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
This validated serving lane also declares native image input. Setup updates the
selected Hermes model's vision capability and OpenClaw's input descriptor;
explicit text or auxiliary-image routing preferences remain unchanged.

Alternative generation transports, unsupported parallel modes, precomputed
multimodal embeddings, LoRA and external KV transfer fail closed. The host
application retains responsibility for approving contacts and tools. Engine
profiles never grant tool permissions.

`UPSTREAM_PROVENANCE.json` records the source-only upstream handoff. `sha256.json` records
this complete packaged integration, including the wizard launcher.
