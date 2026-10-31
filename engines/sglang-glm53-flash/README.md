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

The four-GPU SM120 configuration completed protocol, cache, native Hermes and
OpenClaw checks at a 65,536-token context with EAGLE active. Stock, runtime profile
and exact edited reference each passed 16/16 common fixtures. Both stock and
profile retrieved the correct answer from 59,042 input tokens. Native Verified
and Always On paths passed their protocol checks, including child and image
verification behavior.

Strict visual-answer checks failed in stock, runtime profile and exact edited
reference. Those failures remain part of the qualification; image transport and
authorization success is not a visual-quality guarantee. NEXTN and a complete
SM121 deployment have not been GPU-qualified by this run.
