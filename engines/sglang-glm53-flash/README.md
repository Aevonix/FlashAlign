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
