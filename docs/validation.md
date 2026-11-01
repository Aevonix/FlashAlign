# Release validation

The five starter profiles remain experimental. Model execution, request
authorization and answer quality are tested separately. Loading a profile does
not establish full edited-checkpoint equivalence.

## Model and engine coverage

| Starter | Tested runtime | Evidence and current limits |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | Native vLLM, pinned image libraries, official FP8, four SM120 GPUs | Eight public-package protocol/cache checks passed. All 104 layer/rank bindings matched, and mixed owner/off FULL graph execution was verified on all four workers. Docker CPU checks passed separately; full Docker GPU/wizard activation was not tested. |
| DeepSeek V4.1 Flash 1017 | TensorFold 1.2.0, declared EXL3 3.5-bpw base, four SM120 GPUs | 780 operator comparisons and 15 HTTP checks passed. Both modes retained DSpark and separate RAM session reuse; stream cancellation released slots. The actual bundle passed four-rank wizard preparation and full-size profile staging. JSON Schema and guided output are unsupported in both modes. See the [qualification details](#deepseek-native-profile-on-tensorfold-120). |
| GLM 5.3 Full | TensorFold 1.1.9, EXL3 2.75-bpw, four DGX Sparks | Four-host wizard activation and 18 model cases passed, including deterministic arithmetic, concurrent requests, cache reuse and automatic tool calls. Invalid and replayed signatures were rejected. Configured context 8,192, two concurrent sequences, native MTP depth 4. A 2,423-token prompt reused 2,304 tokens independently in stock and owner mode; returning to stock retained its own cache and output hash. Forced tools, JSON Schema and images are unsupported. |
| GLM 5.3 Flash | SGLang 0.5.20, four SM120 GPUs | All 60 target/MTP sites checked. Both modes completed 59,042-token inputs. Eight Verified/Always On Hermes and OpenClaw protocol phases passed; image-answer limits remain. |
| GLM 5.3 Flash | TensorFold 1.1.9, NVIDIA NVFP4, four DGX Sparks | Four-host wizard activation, 23 runtime/auth/cache/concurrency/harness-hook cases, and four JSON Schema/required-tool cases passed. Context 8,192, two concurrent sequences, fixed native MTP active. |
| Qwen3.8-27B | vLLM, SGLang and llama.cpp | Direct requests, cache isolation, native harness routing and approximately 65K-token inputs exercised with the documented checkpoint and speculative method. |
| Qwen3.8-27B | TensorFold 1.1.9, MLX-format 4-bit, one Linux/NVIDIA DGX Spark | Wizard activation, 23 short runtime/auth/cache/concurrency/harness-hook cases and four grammar cases passed with DFlash2. A 48,037-token prompt reused 45,056 tokens separately in stock and owner namespaces. |
| Qwen3.8-27B | TensorFold 1.1.9, EXL3 2.0-bpw, one SM120 GPU | 26 device cases passed, including DFlash2, authentication and grammar checks. |
| Gemma 4 26B A4B | TensorFold 1.1.9, MLX-format 4-bit, one Linux/NVIDIA DGX Spark | 20 fixture/cache/concurrency runs plus auth/replay checks passed with DFlash active. A 51,231-token prompt reused 51,224 tokens in separate stock/strength namespaces. |
| Gemma 4 26B A4B | vLLM 0.32.0, ngram, one GPU | All 50 sites observed. Eleven native Hermes/OpenClaw protocol checks passed; exact-answer and unexpected image-tool calls remain documented below. |
| Gemma 4 26B A4B | SGLang 0.5.20, NGRAM, one DGX Spark | All 50 sites, mixed rows, auth/replay and strength-separated caches checked. Six direct text-quality checks passed. Hermes protocol passed four of four; strict answers passed three of four. |
| Gemma 4 26B A4B | llama.cpp 0.8.0, Q8_0, one SM120 GPU | 32 direct checks passed with MTP and 32 with ngram. Authentication, replay, concurrent requests, streaming and separate stock/profile/strength caches checked. Both methods completed four 63,929-token requests; warm requests reused 63,928 tokens in their own namespaces. Native agent answer-quality limits are described below. |
| Qwen3.8-27B | llama.cpp 0.8.0, Q8_0, one SM120 GPU | 32 direct checks passed with an external draft. Four 63,927-token requests passed; warm stock and profile requests each reused 63,923 tokens. |

These results apply to the listed checkpoint formats, hardware and settings.
The llama.cpp rows use upstream commit
`8642541b255bfb5dc39cd351a178be09013a55d6`.
TensorFold bundles check CPU and GPU architecture separately. The
four-host GLM Flash run used a shared read-only checkpoint.

The final GLM Full TensorFold run used source
`933d3d766320d9460df45a819ea661755a7d9a0a` and checkpoint
`Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold` at
`cf1fee8923b770b0c47f7a032e1b0f30a57c626a`. Stock arithmetic matched the
unmodified TensorFold baseline. MTP recorded 43 drafted tokens on ordinary stock
requests and 45 on valid owner requests; invalid-signature fallback calls were
checked separately. This run checked concurrent requests; it does not
establish fused mixed-batch execution, broad answer quality or throughput.

The Qwen and GLM Flash TensorFold harness checks used the real Hermes Runtime and
OpenClaw adapter hooks with synthetic native events and real model HTTP
requests. They do not certify every gateway version or messaging transport.

Qwen's 8 GiB host cache could not retain three separate 48K-token snapshots at
once. Eviction caused a fresh prefill, and stock output returned to the same
token hash. A separate 9,637-token test retained all three mode/strength
namespaces and reused 8,192 tokens in each. At strength 2, the 48K arithmetic
fixture ended immediately without answer text; strength 1 remains the default.

## DeepSeek native profile

Version 1.1.0 replaces the compact DeepSeek starter with a native tensor profile
at the same starter ID. It contains the exact 26 WO-B matrices and 26 scale
tensors from the pinned Keys source. It supports strength 1.0 only. The earlier
85-site projection integration and TensorFold EXL3 tests used a different
artifact; their results do not qualify this native starter. External projection
profiles can still use those integrations.

The native operator was checked on the pinned engine before public packaging:

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

Those are prior native-operator results. A separate public-package check on
1 November 2026 passed all eight protocol/cache requests using the pinned
image's libraries on four SM120 GPUs:

- Signed requests selected the native profile; unsigned and replayed requests
  used stock mode.
- A repeated owner prefix reused 2,816 tokens. Its first owner request and the
  same-prefix stock request each reused zero.
- All four workers recorded mixed FULL graph execution with eight owner rows,
  eight off rows and no invalid row flags.
- Owner requests recorded 59 drafted tokens and 30 accepted tokens with
  adaptive DSpark7 active.

The public runtime reported a configured 1,048,576-token context and 7,809,386
KV-token capacity. This was not a million-token input test or a throughput
benchmark. See the [aggregate record](../benchmarks/results/deepseek-native-qualification-20261101.json).

The GPU run used a host process with the pinned image's libraries. Separate
checks in the actual Docker image validated the full profile, signing-key
ownership, plugin discovery and read-only Hugging Face snapshot mounts.
Full GPU activation through Docker and the wizard, a fresh GPU dependency
installation, and image requests were not tested in this run. No behavioral
benchmark was repeated.

These checks do not establish all-layer reduced-output or full-logit identity,
a refusal rate or broad quality parity. Retaining exact source tensors avoids
a low-rank fitting approximation; it does not guarantee identical full-model
answers.

The pinned runtime uses adaptive DSpark with seven draft tokens. Its requested
`FULL_AND_PIECEWISE` setting resolves to FULL decode graphs with eager prefill
on this model. PIECEWISE execution is not supported by this path. Speculative
decoding remained active in stock and native-profile tests.

## Answer quality

GLM Flash passed 16 of 16 common text fixtures in stock mode, with the runtime
profile, and with the exact edited reference. Its structured tool fixture also
passed. One image-color fixture failed in all three configurations. Its image
authorization and profile propagation checks passed.

Gemma's vLLM harness run used one GB10, a 65,536-token context, one concurrent
sequence, 0.60 GPU memory utilization and ngram with three draft tokens. It
completed 38 new model requests and recorded 11,676 draft tokens. The earlier
standard/profile 59,045-token retrieval passes remain separate evidence. This
run did not test the launcher's two-sequence, 0.85-memory defaults on GB10.

Gemma's vLLM routing checks passed independently of its answer quality. Always On
Hermes matched three of four text answers and one of two parent answers; both
child answers matched. Some OpenClaw child and unknown-contact answers missed
their exact targets. Two Hermes image cases called a vision tool unexpectedly;
one captionless image reached the test's 4,096-token reasoning budget before
native recovery. Hash-only records cannot establish whether the color answer
was semantically wrong. Earlier failures remain recorded.

Gemma's separate [SGLang integration](../engines/sglang-gemma4/README.md) uses
source revision `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` and the unchanged
50-site profile. Its 16 direct requests checked stock/profile isolation,
authentication, replay rejection, cache separation by profile strength and
active NGRAM decoding. Arithmetic, factual and JSON checks passed in both
stock and profile modes.

The SGLang Hermes run completed four native protocol cases in five model
requests, including revocation on a reused agent and a real `read_file` call.
Three answers matched exactly. The initial approved no-tool case returned the
requested number followed by an unsolicited introduction, so the strict native
quality result remains failed. The engine used a fresh isolated dependency
environment, a configured 65,536-token context, two concurrent requests and
0.60 memory fraction. This run used short prompts; it did not establish
long-context capacity, image quality, OpenClaw coverage or full wizard engine
activation for this pairing.

The profiles do not guarantee a particular response, zero refusals, or lossless
reproduction of an edited model. Use your workload to assess answer quality.

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

Each 32-case direct run includes six short exact-answer fixtures in each mode.
These are smoke tests, not a benchmark of reasoning, coding or refusal rates.
The runs use two slots, 65,536 tokens per slot and a 2 GiB in-memory prompt
cache. An earlier 512 MiB cache evicted the separate mode/strength entries;
the larger cache retained the tested entries. Neither configuration reused a
cache entry from another mode.

Gemma's official MTP draft and ngram lookup both recorded accepted draft tokens
in stock and profile modes. A repeated copy fixture recorded 48 accepted ngram
draft tokens in each mode. Some prompts generated no ngram matches, so an enabled
method alone was not counted as evidence of active drafting.

Native agent tests include exact-answer misses and repeated or omitted
delegation calls. Successful authorization does not guarantee tool-following
or output-format compliance. Gemma's agent-workflow quality remains
experimental. The llama.cpp paths currently qualify text input; image input
requires separate validation.

Gemma's native Hermes checks covered revocation, a real file tool and delegated
work. OpenClaw checked approved and unknown contacts, inherited child access
with the tool flag on and off, and Always On. Fresh isolated child-policy cases
passed for all three settings. Thinking on/off settings reached the engine;
same-turn reasoning history was preserved when thinking was on.
Reused-conversation cases that repeated or omitted child calls failed their
workflow checks and remain in the validation record.
The [runtime summary](../benchmarks/results/llamacpp-qualification-20261030.json)
records the engine and profile hashes, settings and measured scope.

## Small benchmark pilots

The [Gemma TensorFold GSM8K pilot](../benchmarks/results/gemma-native-gsm8k-pilot.json)
used the fixed first 16 test questions with five examples, a 512-token output
budget and DFlash active. Stock answered 14 of 16 correctly; the profile answered
15 of 16. This is a small, non-random pilot, not a full GSM8K score or evidence
of a general quality improvement. Refusal and over-refusal benchmarks remain
separate work.

The [Qwen TensorFold GSM8K pilot](../benchmarks/results/qwen-native-gsm8k-pilot.json)
used the first 32 test questions, five examples per question and a 512-token
output budget with DFlash2 active. Both stock and profile modes answered 31 of
32 correctly. This is also a small, non-random pilot, not a full GSM8K score or
proof of general quality parity.

## Software and distribution checks

Automated tests cover request authorization, message/media binding, adapter
registration, terminal sessions, tool inheritance, Always On, setup and process
supervision. Engine packages have separate operator and runtime checks. Test
counts change as the release tree is updated; model results above remain tied
to their recorded runtime configurations.

The 1.1.0 update passed 768 core tests, 19 shared engine tests, 88 native
DeepSeek engine tests and eight legacy projection tests. These software checks
are separate from the GPU request checks above.

Fresh dependency installations and wizard activation were checked for Qwen on
vLLM and SGLang with BF16 weights. A fresh llama.cpp build passed wizard
activation with Q8_0 weights. The TensorFold wizard passed on Qwen MLX
with one DGX Spark, GLM Flash NVFP4 with four Sparks, and GLM Full EXL3 with
four Sparks. The x86 Qwen EXL3 engine has separate device coverage; a full x86
wizard deployment was not part
of that test. Standard and signed-profile requests completed with the
declared speculative decoding active, followed by clean shutdown. These are
startup checks for the listed configurations. Fresh installation of other
model and engine combinations requires separate tests.

Gemma's llama.cpp setup passed a fresh wizard check, preparation and activation
with the default ngram method and a 2 GiB prompt cache. Post-activation checks
covered both modes, 63,929-token inputs, separate cache reuse and accepted draft
tokens. No separate draft-model download is needed for this default.

The release source and SDK are built from a pinned tree. Packaging checks verify
file contents and executable modes; a fresh SDK install verifies installed
files, dependencies and the command-line entry point. Runtime downloads are
pinned by size, SHA-256 digest and complete file inventory. Model behavior on a
different deployment still requires a model run.

## DeepSeek native profile on TensorFold 1.2.0

The unchanged native Keys profile passed operator and runtime checks on four
RTX PRO 6000 GPUs with the pinned LibertAIDAI EXL3 3.5-bpw base. The run used a
65,536-token context, two slots and active DSpark.

The operator checks covered 60 cases at layers 10, 22 and 35 across all four
logical tensor-parallel partitions, including mixed rows and changing graph
masks. All 780 comparisons passed. The largest sampled relative L2 error to
the independent FP64 reference was about 0.168%, within the checked rounding bound.
This is matrix-site evidence, not full-model equivalence to another engine.

Fifteen HTTP checks passed their expected outcomes, including signed requests,
unsigned and replayed requests, invalid profiles and strengths, repeated cache
use and concurrent stock/profile requests. Both modes proposed speculative
tokens. Cancelled streams in both modes released their slots; subsequent
requests succeeded, and all four ranks shut down cleanly. Repeated stock and profile sessions each resumed an 8,720-token prefix
from RAM. These family cache hits appear in the runtime log, not the OpenAI
cached-token field. JSON-schema requests were rejected in both modes, as the
current TensorFold DeepSeek backend does not support them.

The [qualification record](../benchmarks/results/deepseek-tensorfold-native-qualification-20261101.json)
contains exact model, profile and engine pins. These are functional checks,
not refusal-rate, quality or throughput benchmarks.
