# Release validation

The five starter profiles are experimental. This page records runtime,
authorization and answer-quality checks for each tested configuration. Full
edited-checkpoint equivalence and broad refusal-rate benchmarks are not established.

## Model and engine coverage

| Starter | Tested runtime | Evidence and current limits |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | Native vLLM, pinned image libraries, official FP8, four SM120 GPUs | Eight public-package protocol/cache checks passed. All 104 layer/rank bindings matched, and mixed owner/off FULL graph execution was verified on all four workers. Docker CPU checks passed separately; full Docker GPU/wizard activation was not tested. |
| DeepSeek V4.1 Flash 1017 | TensorFold 1.2.0, declared EXL3 3.5-bpw base, four SM120 GPUs | 780 operator comparisons and 15 HTTP checks passed. Both modes retained DSpark and separate RAM session reuse; stream cancellation released slots. The actual bundle passed four-rank wizard preparation and full-size profile staging. JSON Schema and guided output are unsupported in both modes. See the [qualification details](#deepseek-native-profile-on-tensorfold-120). |
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
four Sparks. Qwen EXL3 passed device checks on x86; a full x86 wizard deployment
was not tested. Standard and signed-profile requests completed with the
declared speculative decoding active, followed by clean shutdown. Other
model/engine installation combinations remain untested.

Gemma's llama.cpp setup passed a fresh wizard check, preparation and activation
with the default ngram method and a 2 GiB prompt cache. Post-activation checks
covered both modes, 63,929-token inputs, separate cache reuse and accepted draft
tokens. No separate draft-model download is needed for this default.

The release source and SDK are built from a pinned tree. Packaging checks verify
file contents and executable modes; a fresh SDK install verifies installed
files, dependencies and the command-line entry point. Runtime downloads are
pinned by size, SHA-256 digest and complete file inventory.

## DeepSeek native profile on TensorFold 1.2.0

The unchanged native Keys profile passed operator and runtime checks on four
RTX PRO 6000 GPUs with the pinned LibertAIDAI EXL3 3.5-bpw base. The run used a
65,536-token context, two slots and active DSpark.

The operator checks covered 60 cases at layers 10, 22 and 35 across all four
logical tensor-parallel partitions, including mixed rows and changing graph
masks. All 780 comparisons passed. The largest sampled relative L2 error to
the independent FP64 reference was about 0.168%, within the checked rounding bound.
This measures the matrix operator; cross-engine full-model equivalence is unverified.

Fifteen HTTP checks passed their expected outcomes, including signed requests,
unsigned and replayed requests, invalid profiles and strengths, repeated cache
use and concurrent stock/profile requests. Both modes proposed speculative
tokens. Cancelled streams in both modes released their slots; subsequent
requests succeeded, and all four ranks shut down cleanly. Repeated stock and
profile sessions each resumed an 8,720-token prefix
from RAM. These family cache hits appear in the runtime log, not the OpenAI
cached-token field. JSON-schema requests were rejected in both modes, as the
current TensorFold DeepSeek backend does not support them.

The [qualification record](../benchmarks/results/deepseek-tensorfold-native-qualification-20261101.json)
contains exact model, profile and engine pins. Refusal rate, answer quality and
throughput were not measured in this run.
