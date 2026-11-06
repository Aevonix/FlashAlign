# Release validation

The [testing policy](testing.md) separates shared regression coverage from
change-scoped compatibility qualification. Releases do not rerun every model,
engine, quantization and hardware combination; the records below retain the
scope and limits of each check.
An unresolved backend failure remains a limitation of that deployment path;
passing shared regressions does not establish that the path works.

The five starter profiles are experimental. This page records runtime,
authorization and answer-quality checks for each tested configuration. Full
edited-checkpoint equivalence and broad capability retention are not established.
The [automated Qwen FR443 result](qwen-fr443-results.md) and
[Qwen/SGLang timing study](qwen-sglang-timing.md) have separate scopes and limits.

Earlier model runs qualify their recorded static-profile builds. The v1.2
runtime checks are listed separately below.

## Qwen graph answer-quality confirmation

A fixed released-eager versus opt-in-graph Owner Mode comparison passed
completion and GSM8K, but did not establish every prespecified capability
gate. IFEval and MBPP did not establish their conservative 2-percentage-point
bounds at the fixed maximum. Both arms had the same profile active; eager remains
the default. [Results and limits](qwen-sglang-quality.md),
[aggregate evidence](../benchmarks/results/qwen-sglang-quality-20261102.json).

## FlashAlign 1.2 lifecycle checks

The core suite passed 1,083 tests on both Python 3.10 and 3.12. These include
duplicate plugin loads, reload and unload ownership, concurrent verification,
and request-bound identity diagnostics. Separate software
checks cover shared engine gates, SGLang's lifecycle protocol and graph contract,
projection generations and native DeepSeek contracts. The installed-wheel check
matched the Python runtime modules and all 290 bundled setup assets against
source. From outside the checkout it discovered the engine kits and starter
profiles, copied each kit, ran the setup wizard's detection command and checked
its resume command.
These are software checks, separate from the live engine results below.

An offline 1.1.1 → 1.2.0 → 1.1.1 package cycle preserved the signing key,
contact approvals, client configuration, replay namespace and existing
deployment files. Preparation passed at each step, and rollback removed the
new modules and CLI. This used synthetic deployment fixtures; it does not
establish live engine migration or replay-journal compatibility.
[Package results](../benchmarks/results/package-upgrade-20261102.json).

A fresh Qwen TensorFold installation on Linux x86_64 verified all 375 installed
wheel files, four built-in request checks and eight manual requests. DFlash
remained active. Stock and profile requests each reused 4,096 tokens from a
6,768-token input, with no reuse across modes. Public shutdown and GPU release
passed. [Installed-wizard results](../benchmarks/results/qwen-tensorfold-x86-installed-20261102.json).

| Runtime | Models and speculative methods | Live checks |
| --- | --- | --- |
| SGLang 0.5.20 | Qwen3.8-27B BF16, DFLASH | Twelve transport/cache/cancellation cases and 42 sampled forward attestations passed, including graph and eager prefill, speculative verification and mixed standard/profile rows. A separate run passed 1,000 requests across 100 activation/rollback cycles, with 200 observed live swaps and 100 observations of concurrent generations. |
| SGLang 0.5.20 | Gemma 4 26B A4B FP8 compressed tensors, BF16 compute, NGRAM, one DGX Spark | Profile activation, generation/digest checks, cache isolation, cancellation and retirement passed. A separate run passed 1,000 requests across 100 cycles, with 200 observed live swaps and 100 observations of concurrent generations. |
| SGLang 0.5.20 | GLM 5.3 Flash, official FP8, EAGLE, four SM120 GPUs | Fourteen transport checks, separate cold/warm caches, cancellation and profile lifecycle checks passed. The stress run completed 1,000 requests and 100 cycles with no failed requests, 200 live swaps and 100 observations of concurrent generations. Both modes accepted speculative tokens. The original generation was restored, both process trees were reaped and all four GPUs drained. [Results](../benchmarks/results/glm-flash-sglang-lifecycle-20261102.json). |
| vLLM 0.32.0 | Qwen3.8-27B BF16, ngram | Full-model checks passed for profile loading, activation, stale-revision rejection, rollback, generation/digest headers, separate cold/warm caches, active-stream cancellation, busy retirement, eviction and replacement allocation. All 129 configured sites were exercised. A separate run passed 1,000 requests and 100 update cycles, with 200 observed live swaps, 100 observations of concurrent generations, and no wrong-generation responses or request retries. |
| vLLM 0.32.0 | Gemma 4 26B A4B FP8 compressed tensors, ngram | Eleven bounded inference checks, four authorization checks and one cancelled stream passed. All 50 configured sites were exercised. Separate cold/warm caches, distinct profile payloads, activation, stale-revision rejection, rollback, overlapping generations, eviction, retirement and replacement allocation were checked. Draft proposals and acceptances were observed in both modes. |
| Native vLLM `0.1.dev21836+g6d66eed35.cu134` | DeepSeek V4.1 Flash 1017, official FP8, DSpark, four SM120 GPUs | Fifteen request checks covered authorization, replay rejection, separate caches, speculation, live activation, cancellation and retirement. Four later diagnostic requests checked two profile generations. Eight packing records covered all four ranks, and two mixed-generation observations matched across ranks. Managed shutdown and GPU drain passed. [Combined record](../benchmarks/results/deepseek-native-lifecycle-20261102.json). |
| llama.cpp, upstream `8642541b255bfb5dc39cd351a178be09013a55d6` | Qwen3.8-27B Q8_0 with an external draft; Gemma 4 26B A4B Q8_0 with ngram and MTP | Both models passed profile loading, activation, rollback, stale-revision rejection, generation/digest checks, cache isolation, cancellation and retirement. Gemma ngram also passed 1,000 requests and 100 live switches between distinct profile generations. |
| TensorFold 1.2, native CUDA integration | Qwen3.8-27B, MLX-format 4-bit, DFlash | Fresh managed installation passed 1,102 alias updates without an engine restart. An old-generation request remained active across transaction-history rollover while a request using a distinct generation completed. Stock and two profile generations had separate cold/warm caches; each warm request reused 4,096 of 5,120 prompt tokens. Retirement, replacement allocation, tool routing and shutdown passed with speculative decoding active. An earlier run passed 1,000 requests and 100 update cycles. |
| TensorFold 1.2, native CUDA integration | Gemma 4 26B A4B, MLX-format 4-bit, DFlash, one DGX Spark | Activation, stale-revision rejection, rollback, retirement and replacement allocation passed. Stock and two distinct profile generations each had separate cold/warm caches; a repeated 3,322-token prompt reused 3,315 tokens. Concurrent generations, client disconnect and request draining were checked. Seven additional requests finished normally, and three authorization checks passed. Draft proposals and acceptances were recorded in both modes. |
| TensorFold 1.2, native CUDA integration | GLM 5.3 Flash, NVIDIA NVFP4, native MTP, four DGX Sparks | Profile activation, stale-revision rejection, rollback, retirement, separate caches and stream cancellation passed. Both modes reused 2,048 tokens from a 2,561-token prompt. Separate simultaneous stock/profile streams passed. The stress run completed 1,000 requests and 100 update cycles, with 200 live swaps, 100 concurrent-generation observations and no failed requests. All four ranks agreed on the final state and were reaped. |
| TensorFold 1.2, native CUDA integration | GLM 5.3 Full, EXL3 2.75-bpw, native MTP, four DGX Sparks | The managed 8,192-token, two-slot configuration passed focused authentication/cache checks, simultaneous 512-token stock/profile streams and separate 6,148-input-token requests. It completed 1,000 requests and 100 update cycles, with 200 live swaps, 100 concurrent-generation observations and no failed requests. The original generation was restored and all four ranks shut down cleanly. |

Separate six-request checks passed for the GLM Full and Flash 65,536-token
presets with two configured slots and native MTP. Each mode reused 12,032 of
12,053 input tokens on Full, and 10,240 of 12,050 on Flash. Both modes accepted
a 32,768-token output allowance, and all four ranks shut down cleanly. These
runs did not test full-window inputs or generate 32,768 output tokens.
[Full results](../benchmarks/results/glm-full-tensorfold-context-20261102.json),
[Flash results](../benchmarks/results/glm-flash-tensorfold-context-20261102.json).

The Qwen, Gemma vLLM and llama.cpp runs used one RTX PRO 6000 Blackwell Max-Q per model;
the Gemma SGLang and TensorFold runs used one DGX Spark. The Qwen SGLang
attestations check sampled output rows against the runtime operator and require
sampled standard rows to remain byte-identical. They do not measure fit to an
edited checkpoint. The llama.cpp runs check full-model lifecycle behavior;
they do not add numerical GPU-site attestations.

Speculative decoding remained active. These checks establish neither refusal
rates nor model-quality equivalence or representative throughput. Single-worker
results do not qualify distributed updates. The four-rank GLM Flash and Full
stress runs check normal operation. Separate Full recovery checks are recorded
below; they do not qualify Flash recovery.

An earlier vLLM stress run stopped on a connection error after 791 successful
requests. The completed follow-up used a one-second client idle-connection
expiry; the earlier run used five seconds. Both results are retained. This
does not establish resilience to every transport failure.

The native DeepSeek record combines two runs. The first exceeded its shutdown
wait; the later run required an offline correction to its diagnostic reader.
Both original outcomes are retained in the linked record. These checks do not
establish output parity, answer quality, full-context operation or Docker GPU
activation.

### GLM Full distributed recovery

Separate bounded runs checked TensorFold's four-rank GLM Full recovery with
an 8,192-token context, two slots, an 8 GiB prompt cache and native MTP depth 4.
Six fault cases passed: asymmetric artifact-transfer rejection, bridge loss
before and after a durable decision, lost commit and resume acknowledgements,
and retired-generation tombstone recovery. That cohort later failed a
test-harness history-accounting assertion and remains a failed overall run.

A later continuation used 1,019 distinct aborted native transactions to fill
history, with all four ranks acknowledging each abort and resume. It checked
history rollover after a lost reply, stale-epoch rejection and intentional
worker loss. A fresh whole-cohort restart then restored the durable alias,
generation and profile digests, rejected old-boot control and served an owner
request using the restored generation. Independent cleanup checks found all
recorded processes gone, GPU compute inventories empty and leases released.

The [combined recovery record](../benchmarks/results/glm-full-tensorfold-recovery-20261102.json)
pins the source, model, profiles, settings and limits for each scope. Earlier
failed or incomplete attempts remain so. Aborted transactions are not successful
activations; this evidence does not establish replay persistence across process
restarts, 64K operation, answer quality or performance.

## Model and engine coverage

The following table records earlier static-profile builds.

| Starter | Tested runtime | Evidence and current limits |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | Native vLLM, pinned image libraries, official FP8, four SM120 GPUs | Eight public-package protocol/cache checks passed. All 104 layer/rank bindings matched, and mixed owner/off FULL graph execution was verified on all four workers. Docker CPU checks passed separately; full Docker GPU/wizard activation was not tested. |
| DeepSeek V4.1 Flash 1017 | TensorFold 1.2.0, declared EXL3 3.5-bpw base, four SM120 GPUs | 780 operator comparisons and a fresh 27-request managed installation and lifecycle check passed. Both modes retained DSpark and separate RAM caches. Profile swaps, mixed streams, cancellation, 60K inputs and shutdown passed. See the [qualification details](#deepseek-native-profile-on-tensorfold-120). |
| GLM 5.3 Full | TensorFold 1.1.9, EXL3 2.75-bpw, four DGX Sparks | Four-host wizard activation and 18 model cases passed, including deterministic arithmetic, concurrent requests, cache reuse and automatic tool calls. Invalid and replayed signatures were rejected. Configured context 8,192, two concurrent sequences, native MTP depth 4. A 2,423-token prompt reused 2,304 tokens independently in stock and owner mode; returning to stock retained its own cache and output hash. Forced tools, JSON Schema and images are unsupported. |
| GLM 5.3 Flash | SGLang 0.5.20, four SM120 GPUs | All 60 target/MTP sites checked with EAGLE and a 65,536-token context. Both modes completed 59,042-token inputs. Eight Verified/Always On Hermes and OpenClaw protocol phases passed; image-answer limits remain. |
| GLM 5.3 Flash | TensorFold 1.1.9, NVIDIA NVFP4, four DGX Sparks | Four-host wizard activation, 23 runtime/auth/cache/concurrency/harness-hook cases, and four JSON Schema/required-tool cases passed. Context 8,192, two concurrent sequences, fixed native MTP active. |
| Qwen3.8-27B | vLLM, SGLang and llama.cpp | Direct requests, cache isolation, native harness routing and approximately 65K-token inputs exercised with the documented checkpoint and speculative method. |
| Qwen3.8-27B | TensorFold 1.1.9, MLX-format 4-bit, one Linux/NVIDIA DGX Spark | Wizard activation, 23 short runtime/auth/cache/concurrency/harness-hook cases and four grammar cases passed with DFlash2. A 48,037-token prompt reused 45,056 tokens separately in stock and owner namespaces. |
| Qwen3.8-27B | TensorFold 1.1.9, EXL3 2.0-bpw, one SM120 GPU | 26 device cases passed, including DFlash2, authentication and grammar checks. |
| Gemma 4 26B A4B | TensorFold 1.1.9, MLX-format 4-bit, one Linux/NVIDIA DGX Spark | 20 fixture/cache/concurrency runs plus auth/replay checks passed with DFlash active. A 51,231-token prompt reused 51,224 tokens in separate stock/strength namespaces. |
| Gemma 4 26B A4B | vLLM 0.32.0, ngram, one GPU | All 50 sites observed. Eleven native Hermes/OpenClaw protocol checks passed; exact-answer and unexpected image-tool calls remain documented below. |
| Gemma 4 26B A4B | SGLang 0.5.20, NGRAM, one DGX Spark | All 50 sites, mixed rows, auth/replay and strength-separated caches checked. Six direct text-quality checks passed. Hermes protocol passed four of four; strict answers passed three of four. |
| Gemma 4 26B A4B | llama.cpp 0.8.0, Q8_0, one SM120 GPU | 32 direct checks passed with MTP and 32 with ngram. Authentication, replay, concurrent requests, streaming and separate stock/profile/strength caches checked. Both methods completed four 63,929-token requests; warm requests reused 63,928 tokens in their own namespaces. Native agent answer-quality limits are described below. |
| Qwen3.8-27B | llama.cpp 0.8.0, Q8_0, one SM120 GPU | 32 direct checks passed with an external draft. Four 63,927-token requests passed; warm stock and profile requests each reused 63,923 tokens. |

Hardware and settings in this table identify the tested deployments. See
[compatibility](compatibility.md) for current engine requirements.
The llama.cpp rows use upstream commit
`8642541b255bfb5dc39cd351a178be09013a55d6`.
TensorFold bundles check CPU and GPU architecture separately. The
four-host GLM Flash run used a shared read-only checkpoint.

The GLM Full TensorFold run used source
`933d3d766320d9460df45a819ea661755a7d9a0a` and checkpoint
`Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` at
`cf1fee8923b770b0c47f7a032e1b0f30a57c626a`. Stock arithmetic matched the
unmodified TensorFold baseline. MTP recorded 43 drafted tokens on ordinary stock
requests and 45 on valid owner requests; invalid-signature fallback calls also
passed. Fused mixed-batch execution, broad answer quality and throughput were
outside this run's scope.

The Qwen and GLM Flash TensorFold harness checks used the real Hermes Runtime and
OpenClaw adapter hooks with synthetic native events and real model HTTP
requests. Messaging transports and other gateway versions require their own checks.

Qwen's 8 GiB host cache could not retain three separate 48K-token snapshots at
once. Eviction caused a fresh prefill, and stock output returned to the same
token hash. A separate 9,637-token test retained all three mode/strength
namespaces and reused 8,192 tokens in each. At strength 2, the 48K arithmetic
fixture ended immediately without answer text; strength 1 remains the default.

## Qwen profile and runtime checks

The Qwen3.8-27B profile was checked with 256 fresh inputs at each of its 129
sites. Maximum relative output error against the selected edited checkpoint
was 0.196%. The export excludes that checkpoint's native MTP changes.

| Runtime | Recorded checks |
| --- | --- |
| vLLM, BF16, ngram, one RTX PRO 6000 | 18 CPU tests and 15 live protocol/cache checks passed. All 16 arithmetic/Python-output fixtures passed in each mode without truncation. Serial requests on a 96 GB GPU processed 65,470 input tokens in both modes and exercised native Hermes/OpenClaw tools and child inheritance. Four simultaneous maximum-length requests were not tested. |
| SGLang 0.5.20, BF16, DFlash2, one 95 GiB RTX PRO 6000 Blackwell Max-Q | 23 CPU tests and 47 live checks passed. All 43 generated-answer checks passed, including 16 shared math/code fixtures per mode and two streaming requests. Every generation reported active speculative verification. Repeated requests reused 576 tokens in their own mode; the first owner request and a strength change reused zero. Serial 65,000-input-token requests and managed wizard/native harness checks passed with a 65,536-token context. |

The SGLang long-input run reused pinned dependencies.
[Timing samples](benchmark-results.md#qwen-sglang-startup-and-ttft) are recorded
separately.

The additional Qwen Flash Next SGLang integration passed managed wizard,
native Hermes/OpenClaw and serial 65,000-input-token checks in both modes on
two 95 GiB RTX PRO 6000 Blackwell Max-Q GPUs. Its configured context was 65,536
tokens; the full operator checks used 2,048 tokens. The context run reused
pinned dependencies.

## DeepSeek native profile

The native starter introduced in 1.1.0 contains the exact 26 WO-B matrices and
26 scale tensors from the pinned Keys source, at strength 1.0 only. It retains
the `deepseek41-flash1017-v1` ID. Results for the previous 85-site projection
profile, including its TensorFold EXL3 runs, apply only to that previous artifact.
External projection profiles can still use those integrations.

Native-operator checks on the pinned engine:

- Source-byte validation covered all 104 layer/rank bindings and 208 tensor
  partitions across four workers.
- Sixty prepared-WO GPU cases at layers 10, 22 and 35 produced 960 bit-identical
  comparisons, including stock, owner and mixed rows, repeated graph replay and
  live rows below the prepared capacity. These are rank-local matrix checks.
- Eight protocol cases passed, covering signed requests, invalid and replayed
  signatures and separate profile caches. All four workers also recorded actual
  mixed owner/stock execution through FULL decode graphs.
- Twelve short fixtures passed in each of stock, native-profile and full-edited
  control modes. The fixtures covered arithmetic, bounded code and benign
  completion, with maximum reasoning. Profile and control final-content hashes
  matched in six cases; reasoning hashes matched in one.

The 1 November 2026 public-package check passed all eight protocol/cache
requests using the pinned image's libraries on four SM120 GPUs:

- Signed requests selected the native profile; unsigned and replayed requests
  used stock mode.
- A repeated owner prefix reused 2,816 tokens. Its first owner request and the
  same-prefix stock request each reused zero.
- All four workers recorded mixed FULL graph execution with eight owner rows,
  eight off rows and no invalid row flags.
- Owner requests recorded 59 drafted tokens and 30 accepted tokens with
  adaptive DSpark7 active.

The runtime reported a configured 1,048,576-token context and 7,809,386 KV-token
capacity. Million-token input and throughput were not tested. See the
[aggregate record](../benchmarks/results/deepseek-native-qualification-20261101.json).

The GPU run used a host process with the pinned image's libraries. Separate
checks in the actual Docker image validated the full profile, signing-key
ownership, plugin discovery and read-only Hugging Face snapshot mounts.
GPU activation through Docker and the wizard, a fresh GPU dependency
installation, image requests and behavioral benchmarks were outside this run's
scope. Exact source tensors avoid low-rank fitting error; all-layer output,
logit and full-model answer identity remain unverified.

The pinned runtime uses adaptive DSpark with seven draft tokens. Its requested
`FULL_AND_PIECEWISE` setting resolves to FULL decode graphs with eager prefill
on this model. PIECEWISE execution is not supported by this path. Speculative
decoding remained active in stock and native-profile tests.

## Previous DeepSeek projection profile

The 85-site profile completed full-model wizard activation and shutdown on
four approximately 95 GiB SM120 GPUs with TP4/PP1/DP1, eager execution, native
DSpark5 and Engram CPU offload. Verified and Always On Hermes checks passed;
all 14 isolated OpenClaw cases passed protocol and strict-answer checks,
including images, both tool-inheritance settings and same-session CLI access.

The lower-memory test override was enabled. Host `MemAvailable` was 367.65 GiB
at startup, with approximately 202 GiB steady host usage; initial cgroup
headroom was not recorded. These observations are separate from the
[launcher's memory policy](../engines/vllm-deepseek/README.md#requirements-and-defaults)
and from the current native starter's requirements.

## Answer quality

GLM Flash passed 16 of 16 common text fixtures in stock mode, with the runtime
profile, and with the exact edited reference. Its structured tool fixture also
passed. One image-color fixture failed in all three configurations. Its image
authorization and profile propagation checks passed.

Gemma's vLLM harness run used one GB10, a 65,536-token context, one concurrent
sequence, 0.60 GPU memory utilization and ngram with three draft tokens. It
completed 38 model requests and recorded 11,676 draft tokens. Separate
stock/profile retrieval checks passed at 59,045 tokens; another run on a 96 GB
RTX PRO 6000 passed at 59,987 tokens in both modes. The launcher's
two-sequence, 0.85-memory defaults were not tested on GB10.

Gemma's vLLM routing checks passed. Always On Hermes matched three of four text
answers and one of two parent answers; both child answers matched. Some
OpenClaw child and unknown-contact answers missed their exact targets. Two
Hermes image cases called a vision tool unexpectedly; one captionless image
reached the test's 4,096-token reasoning budget before native recovery. Only
hashes were retained for the color answer, so semantic accuracy is unknown.

Gemma's [SGLang integration](../engines/sglang-gemma4/README.md) uses
source revision `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` and the unchanged
50-site profile. Its 16 direct requests checked stock/profile isolation,
authentication, replay rejection, cache separation by profile strength and
active NGRAM decoding. The tested cache strengths were 0.5 and 1.0.
Arithmetic, factual and JSON checks passed in both
stock and profile modes.

The SGLang Hermes run completed four native protocol cases in five model
requests, including revocation on a reused agent and a real `read_file` call.
Three answers matched exactly. The initial approved no-tool case returned the
requested number followed by an unsolicited introduction, failing the strict
answer check. The engine used a fresh isolated dependency
environment, a configured 65,536-token context, two concurrent requests and
0.60 memory fraction. Long-context capacity, image quality, OpenClaw and full
wizard activation were not tested for this pairing.

## llama.cpp qualification

The current integration uses source revision
`8642541b255bfb5dc39cd351a178be09013a55d6`. Gemma uses the official
`ggml-org/gemma-4-26B-A4B-it-GGUF` Q8_0 checkpoint at
`9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2`, with the unchanged 50-site
starter profile. The native operator preserves each site's strength and
applies attention and dense/shared MLP projections before post-normalization.
Routed experts are unchanged. A separate instrumented run observed all 50
projection sites on the GPU for an owner request and zero for stock requests,
including a stock request after the owner request. Timing runs used the
uninstrumented build. The equivalent Qwen check observed 129 owner sites and
zero stock sites.

Each 32-case direct run includes six short exact-answer smoke tests in each mode.
The runs use two slots, 65,536 tokens per slot and a 2 GiB in-memory prompt
cache. At 512 MiB, separate mode/strength entries were evicted; at 2 GiB, the
tested entries were retained. Both configurations kept mode caches isolated.

Gemma's official MTP draft and ngram lookup both recorded accepted draft tokens
in stock and profile modes. A repeated copy fixture recorded 48 accepted ngram
draft tokens in each mode. Other prompts generated no ngram matches.

Native agent tests had exact-answer misses and repeated or omitted delegation
calls. Gemma's agent-workflow quality remains experimental. These llama.cpp
checks cover text input; images are untested.

Gemma's native Hermes checks covered revocation, a real file tool and delegated
work. OpenClaw checked approved and unknown contacts, inherited child access
with the tool flag on and off, and Always On. Fresh isolated child-policy cases
passed for all three settings. Thinking on/off settings reached the engine;
same-turn reasoning history was preserved when thinking was on.
Reused-conversation cases failed workflow checks when they repeated or omitted
child calls.
The [runtime summary](../benchmarks/results/llamacpp-qualification-20261030.json)
records the engine and profile hashes, settings and measured scope.

## Small benchmark pilots

Both TensorFold pilots used the first test questions, five examples per
question and a 512-token output budget. These non-random samples are not full
GSM8K scores.

| Model | Sample | Stock correct | Profile correct | Speculation | Record |
| --- | ---: | ---: | ---: | --- | --- |
| Gemma 4 26B A4B | 16 | 14 | 15 | DFlash | [Results](../benchmarks/results/gemma-native-gsm8k-pilot.json) |
| Qwen3.8-27B | 32 | 31 | 31 | DFlash2 | [Results](../benchmarks/results/qwen-native-gsm8k-pilot.json) |

## Software and distribution checks

Automated tests cover request authorization, message/media binding, adapter
registration, terminal sessions, tool inheritance, Always On, setup and process
supervision. Engine packages have separate operator and runtime checks.

The 1.1.0 update passed 768 core tests, 19 shared engine tests, 88 native
DeepSeek engine tests and eight legacy projection tests.

Gemma vLLM passed pinned-version, import, entrypoint and CPU operator checks
in a fresh isolated dependency installation. Its serving checks used a
separately prepared GPU environment.

Fresh dependency installations and wizard activation were checked for Qwen on
vLLM and SGLang with BF16 weights. A fresh llama.cpp build passed wizard
activation with Q8_0 weights. The TensorFold wizard passed on Qwen MLX
with one DGX Spark, GLM Flash NVFP4 with four Sparks, and GLM Full EXL3 with
four Sparks. Qwen EXL3 passed device checks on x86. A separate fresh x86_64
managed installation with the pinned MLX-format 4-bit Qwen checkpoint passed
bounded wizard, signed-request and cache checks, as recorded above.
Standard and signed-profile requests completed with the
declared speculative decoding active, followed by clean shutdown. Other
installation coverage is limited to the combinations recorded here.

Fresh installs of a 1.2.0 candidate wheel passed the public setup command on one
DGX Spark per model, using TensorFold source `a7ed78d1c16fc9745f7540c504ebf87289be5e9a`.
All 375 installed files matched the wheel, including 289 setup assets. Each run
completed four standard/profile requests with the expected profile identity,
generation and digest. DFlash remained active and accepted draft tokens on every
request.

| Model | Configured context / slots | Input tokens | Cold cached tokens, standard / profile | Repeated cached tokens, standard / profile |
| --- | --- | ---: | --- | --- |
| Qwen3.8-27B | 65,536 / 2 | 6,204 | 0 / 0 | 4,096 / 4,096 |
| Gemma 4 26B A4B | 131,072 / 2 | 1,344 | 0 / 0 | 1,337 / 1,337 |

Public shutdown and independent cleanup checks confirmed that the workers exited
and GPU reservations were released. These are installation, routing and cache
checks, not maximum-context input, throughput or answer-quality tests. The
earlier Qwen lifecycle run above separately covered administration, tool routing
and live profile updates. The [aggregate record](../benchmarks/results/tensorfold-installed-wheel-v120.json)
contains the exact build, profile and checkpoint pins and per-request counts.

Gemma's llama.cpp setup passed a fresh wizard check, preparation and activation
with the default ngram method and a 2 GiB prompt cache. Post-activation checks
covered both modes, 63,929-token inputs, separate cache reuse and accepted draft
tokens. No separate draft-model download is needed for this default.

The release source and SDK are built from a pinned tree. Packaging checks verify
file contents and executable modes; a fresh SDK install verifies installed
files, dependencies and the command-line entry point. Runtime downloads are
pinned by size, SHA-256 digest and complete file inventory.

## DeepSeek native profile on TensorFold 1.2.0

The native Keys profile passed operator and runtime checks on four RTX PRO
6000 GPUs with the pinned LibertAIDAI EXL3 3.5-bpw base, a 65,536-token context,
two slots and active DSpark.

The operator checks covered 60 cases at layers 10, 22 and 35 across all four
logical tensor-parallel partitions, including mixed rows and changing graph
masks. All 780 comparisons passed. The largest sampled relative L2 error to
the independent FP64 reference was about 0.168%, within the checked rounding bound.
This measures the matrix operator; cross-engine full-model equivalence is unverified.
See the [operator validation record](../benchmarks/results/deepseek-tensorfold-native-qualification-20261101.json).

A fresh managed installation using TensorFold source `b505ce273d3c` passed a
27-request sequence covering signed requests, replay and invalid-key handling,
profile activation, rollback, retirement, overlapping generations, mixed
stock/profile streams and cancellation. OpenAI usage reported 1,376 reused
prefix tokens in stock, owner and alternate-profile requests. The stock cache
also remained usable after an owner-profile switch. All four ranks shut down
and released their GPU locks.

Both modes completed requests with 60,004 input tokens and two output tokens.
This checks those inputs within the configured window. It does not measure
full-window capacity, refusal rate, answer quality or throughput. The
[installation and lifecycle record](../benchmarks/results/deepseek-tensorfold-installed-20261102.json)
contains the exact pins and results. JSON Schema and guided output remain
unsupported on this backend.
