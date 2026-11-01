# DeepSeek native matrix backend

This backend runs the `deepseek41-flash1017-v1` starter with the pinned official DeepSeek V4.1 Flash 1017 checkpoint. Owner requests use the profile's immutable WO-B matrices at the declared layers. Off requests keep the original execution path. The profile is a separate format from compact output projections and accepts strength **1.0 only**.

Use the FlashAlign wizard for installation. It downloads the profile payload into your deployment cache, verifies the model and profile, prepares the exact container image, and starts this launcher. The image is public and pinned by digest; a different image or engine revision is rejected. Existing compact DeepSeek profiles use the separate `vllm-deepseek` backend.

## Requirements

- Linux x86_64, Docker Engine and NVIDIA Container Toolkit.
- Four SM120 GPUs with at least 94 GiB total and 90 GiB free memory per GPU.
- The exact model revision in `backend.json`, including its original quantization.
- 221 GiB available host/cgroup memory before an unattended start. The container is limited to 224 GiB and the supervisor stops it if available host memory falls below 24 GiB. These are deployment policies, not a claim of the smallest possible machine.
- Space for the model, the container image, compilation caches and the approximately 1.02 GiB native profile payload.

The default keeps adaptive DSpark7, B12X, FP8 KV cache and the engine's graph path enabled. This pinned model uses FULL decode graphs and eager prefill within its `FULL_AND_PIECEWISE` configuration. Prefix caches are separated by effective profile, payload digest and strength. Switching between already loaded off and owner modes does not restart the engine. Loading a different artifact does.

The current native starter is validated for text. Image cache isolation is implemented but image requests are not yet part of this backend's published qualification. This backend does not claim compatibility with other GPU architectures, tensor-parallel sizes, expert/context parallelism, external KV transfer or LoRA.

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

## Validation

Run the CPU contract and lifecycle tests with:

```sh
python -m pytest -q engines/vllm-deepseek-native/tests
```

These test artifact parsing, corruption rejection, owner/off row mapping, separate graph identities, authentication, cache namespaces, immutable bindings and container ownership. CPU tests do not establish GPU behavior, throughput or a refusal rate. GPU qualification is recorded separately in the release evidence and backend metadata.
