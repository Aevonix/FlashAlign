# GLM Flash on SGLang

This integration applies a saved normalized output profile to authenticated
requests on the supported GLM Flash checkpoint. Native EAGLE or NEXTN decoding
stays enabled. Off requests skip transform arithmetic; cache namespaces include
the model, profile and strength.

The stock chat template always opens a reasoning section. It supports low,
high and max effort; `enable_thinking=false` does not disable reasoning. Setup
rejects a saved OpenClaw thinking-off selection for this model.

The supported configuration uses one host, four SM120 or SM121 GPUs, tensor
parallelism four, separate shared experts and BF16 KV cache. The preparation
step selects the measured lower-memory TileLang schedule only for the matching
no-RoPE BF16 attention lane. Other attention lanes keep the upstream schedule.

The setup wizard prepares a pinned SGLang source copy and verifies stock
checkpoint integrity before activation. Profiles must include the complete
declared target and native-MTP writer inventory. The runtime accepts the
documented normalized operator format; it does not depend on how a profile was
created.

See [validation](../../docs/validation.md) for protocol, cache, long-input and
native harness results on four SM120 GPUs with EAGLE.

Strict visual-answer checks failed in stock, runtime profile and exact edited
reference. NEXTN and a complete SM121 deployment remain untested on GPU.

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
