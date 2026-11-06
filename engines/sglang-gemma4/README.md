# Gemma 4 on SGLang

This experimental integration applies the unchanged `gemma4-26b-a4b-v1` profile
to the pinned `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` checkpoint. It uses the
SGLang 0.5.20 source revision
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, declared in `backend.json`.

The profile covers 50 attention-output and dense-MLP-output projections. Hooks
run before Gemma's post-projection normalization. Stock rows bypass the
projection; authenticated profile and strength selections use separate cache
namespaces. The native MoE branch is unchanged.

NGRAM speculative decoding with three draft tokens remains active. This variant requires one GPU,
eager execution and non-overlapping scheduling. It does not accept external
draft models, LoRA composition, distributed workers, HiCache or session-cache
reuse. The Qwen SGLang integration remains separate.

For managed setup with an existing local checkpoint:

```sh
bash ./install.sh --engine-only --backend sglang \
  --model-path /models/gemma4-26b-a4b-fp8
```

The runtime checks model identity, all 50 writers and the supplied payload
checksums. The wizard checks the declared checkpoint metadata and preserves
compatible active NGRAM settings without asking for a draft model.

The tested GB10 configuration used a 65,536-token context, two concurrent
requests and 0.60 memory fraction. Direct protocol and answer checks passed;
the native Hermes strict-answer check failed on one of four cases.
Long-context inputs, image quality, OpenClaw and full wizard engine activation
remain untested for this pairing.
See [validation](../../docs/validation.md) for the results.

## Live profile updates

Start with both `--lifecycle-state /private/flashalign/state` and
`--admin-socket /private/flashalign/admin.sock` to enable live profile management.
The socket parent must be an owned mode-0700 directory. The socket accepts only
processes running as its owner; the request-signing key does not authorize admin
operations.

The local HTTP interface provides `GET /status` and `POST /operations`. Operations
are `stage` (local `path`), `activate` or `rollback` (`alias`, `generation`,
`expected_revision`), `deactivate` (`alias`, `expected_revision`), `retire`
(`generation`), `recover`, and `evict_cache`.
Each request resolves its alias to an immutable generation. Old requests and
cache entries retain that generation after a switch. Retirement waits for all
references and GPU work to finish. Explicit cache eviction requires an idle
scheduler and keeps caching enabled afterward. Updates require acknowledgement
from every tensor-parallel worker, including preparation of native draft runners
where the model uses them.

`GET /flashalign/capabilities` and `GET /flashalign/status` are read-only public
metadata endpoints. Loaded generations and worker acknowledgements describe
runtime state; they are not evidence of numerical or behavioral parity.
