# DeepSeek native matrix backend

The Docker wizard's full-model activation path is not yet qualified end to end.
See [validation results](../../docs/validation.md#deepseek-native-profile) for
the separate CPU lifecycle and native GPU checks.

This backend runs the `deepseek41-flash1017-v1` starter with the pinned official DeepSeek V4.1 Flash 1017 checkpoint. Owner requests use the profile's immutable WO-B matrices at the declared layers. Off requests keep the original execution path. The profile is a separate format from compact output projections and accepts strength **1.0 only**.

Use the FlashAlign wizard for installation. It downloads the profile payload into your deployment cache, verifies the model and profile, prepares the exact container image, and starts this launcher. The image is public and pinned by digest; a different image or engine revision is rejected. Standard Hugging Face cache snapshots are supported: only the selected model repository is mounted, retaining its relative blob links. Local materialized model directories also work. Absolute links and links escaping either layout are rejected before activation. Existing compact DeepSeek profiles use the separate `vllm-deepseek` backend.

## Requirements

- Linux x86_64, Docker Engine and NVIDIA Container Toolkit, with readable unified
  cgroup v2 accounting for the launcher. Missing or unresolvable accounting is
  rejected before activation.
- Four SM120 GPUs with at least 94 GiB total and 90 GiB free memory per GPU.
- The exact model revision in `backend.json`, including its original quantization.
- 221 GiB available host/cgroup memory before an unattended start. The container is limited to 224 GiB and the supervisor stops it if available host memory falls below 24 GiB. These are the launcher’s current memory limits.
- Space for the model, the container image, compilation caches and the approximately 1.02 GiB native profile payload.

The default keeps adaptive DSpark7, B12X, FP8 KV cache and the engine's graph path enabled. This pinned model uses FULL decode graphs and eager prefill within its `FULL_AND_PIECEWISE` configuration. Prefix caches are separated by effective profile, payload digest and strength. Compatible profiles can be staged and activated through the local administration socket without restarting the engine. Changing the model or engine configuration still requires a restart.

Text requests are validated. Image requests remain untested. Other GPU architectures, tensor-parallel sizes, expert/context parallelism, external KV transfer and LoRA are unsupported.

## Direct engine setup

The setup API used by the wizard is `prepare.prepare_environment(state_dir, pull=True)`. Preparation downloads only the pinned engine image and starts no model. To launch directly after full model verification:

```sh
python engines/vllm-deepseek-native/launch.py \
  --model /path/to/DeepSeek-V4.1-Flash-1017 \
  --profile /path/to/materialized-profile \
  --key-file /path/to/private/deployment.key \
  --replay-file /path/to/private/replay/nonces.sqlite \
  --model-receipt /path/to/private/model-receipt.json \
  --state-dir /path/to/private/engine-state \
  --devices 0,1,2,3 --host 127.0.0.1 --port 8041
```

Create the deployment key with the main FlashAlign setup tools. The key must contain exactly 32 random bytes and have mode `0600`; private directories must have mode `0700`. The container runs as the invoking UID and mounts the original key, model, profile and runtime read-only. It never copies or changes key ownership.

`SIGTERM` or `SIGINT` stops the exact owned container. A separate parent-death guard does the same if the supervisor exits unexpectedly. Keep the supervisor running. Readiness requires the live model API and fresh, complete receipts from all four workers, including stock and profile partition hashes. No prompt or response text is included in those receipts.

`--arguments-json` accepts only bounded context, batch, capture-size and GPU-memory overrides. Unsupported flags fail before activation. Existing speculative settings must match this backend's supported DSpark7 configuration; setup does not disable speculation to make an incompatible configuration run.

## Low-rank profiles and profile sets

This backend can load up to seven low-rank profiles beside the native profile.
A low-rank profile uses the `input_low_rank_delta_v1` operator. It stands for a
low-rank change `strength * scale * Q^T B` to a writer matrix. Two sites are
supported, with a rank of up to 128:

| Site | Writer matrix | Input width |
| --- | --- | --- |
| `attn_out` | `layers.N.attn.wo_b.weight` | 8192 |
| `shared_expert_out` | `layers.N.ffn.shared_experts.w2.weight` | 2304 |

The model weights stay as loaded. The engine adds each selected profile's term
to the writer's output, only on the rows of the requests that selected it.

Add one `--low-rank-profile` argument per profile to the launch command:

```sh
  --low-rank-profile /path/to/first-low-rank-profile \
  --low-rank-profile /path/to/second-low-rank-profile \
```

A request can then name one profile or a
[profile set](../../docs/engine-contract.md#profile-sets). A set holds at most
one native profile and any of the loaded low-rank profiles. Each low-rank
profile takes its own strength in `(0, 1]`. The native profile still takes 1.0
only. The engine applies every named profile or rejects the request.

- The engine applies low-rank profiles in the order of the launch arguments.
  The order of the names in a request does not change the result.
- Rows of requests that selected no profile are not read or written by a
  profile's kernel.
- Prefix caches are separated by the whole set: each member's digest, its
  strength and the order of the names.
- `X-Owner-Mode-Effective` lists the applied names in request order. When a
  low-rank profile is applied, `X-FlashAlign-Profile-Digest` carries one digest
  per name in the same order.
- Low-rank profiles load at start and are not part of profile updates. Changing
  them needs a restart. They stay selectable while a profile update has the
  native profile deactivated.
- Each loaded low-rank profile adds a fixed amount of work per targeted layer
  to every request, including requests that did not select it.

The low-rank path has CPU tests for loading, authorization, cache keys and the
four-rank arithmetic, and its kernels were checked through Triton's CPU
interpreter in the pinned image. It has not been qualified on GPUs yet.

## Validation

Run the CPU contract and lifecycle tests with:

```sh
python -m pytest -q engines/vllm-deepseek-native/tests
```

Tests cover artifact validation, row selection, authentication, cache isolation and container lifecycle. See [validation results](../../docs/validation.md#deepseek-native-profile) for GPU checks and the remaining Docker GPU/wizard test gap.

See [profile updates](../vllm/LIFECYCLE.md) for generation loading, the local
administration socket and the container's shared staging directory.
