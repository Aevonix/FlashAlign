# Measured results

Each report names the model, profile, engine, settings and metric it measured.
Refusal behavior, answer quality and request timing are separate measurements.
[Run your own comparison](benchmarks.md) with the included tools.

## Qwen3.8-27B: automated refusal comparison

On the complete 443-case FR443 suite, the judge counted **289/443 stock answers
as refusals (65.24%)**, compared with **11/443 profile answers (2.48%)**. All
886 answers and judgments were valid. The paired change was −62.75 percentage
points, with a 95% interval of [−67.19, −58.24].

The corpus mixes safe, harmful and unsafe requests. This is a judge-rated
refusal result; answer correctness and general capability were not scored.
The predeclared 40-case human audit is pending.

[Full report, subsets and settings](qwen-fr443-results.md) ·
[Machine-readable evidence](../benchmarks/results/qwen-fr443-20261101.json)

## Qwen/SGLang: graph Owner Mode timing

A separate comparison measured **1.324× end-to-end completion-token throughput**
for the opt-in graph Owner Mode path versus the previous release's eager Owner Mode
path. The conditional 95% interval was **[1.313, 1.335]**, based on three paired
process blocks. The primary workload used 258 input tokens and exactly 256
output tokens, with 90 primary requests per arm.

This includes initial request latency and is not a decode-only tok/s figure.
Both arms had the profile active; it is not an off-versus-on comparison. The
study does not establish general capability or the separate completion-regression
confidence gate.

[Full report and limitations](qwen-sglang-timing.md) ·
[Offline recomputation bundle](../benchmarks/evidence/qwen-sglang-20261101/README.md)

## Qwen/SGLang: graph Owner Mode quality

The fixed paired comparison did not establish all capability gates.
Completion and GSM8K passed at the first look; IFEval and MBPP did not
establish their conservative 2-percentage-point bounds at the fixed maximum.
Both arms had the same profile active. Eager remains the default and
the graph path remains opt-in.

[Results, criterion and limits](qwen-sglang-quality.md) ·
[Aggregate evidence](../benchmarks/results/qwen-sglang-quality-20261102.json)

## GLM Full / TensorFold: distributed recovery

Six fault cases passed in a cohort that later failed a test-harness assertion.
Separate runs checked history rollover using 1,019 aborted transactions,
intentional worker loss and recovery from the retained journal after a fresh
four-rank restart. The tests used an 8,192-token context, two slots and active
native MTP. This is combined functional evidence, not a throughput or quality
measurement, and does not turn the earlier failed cohort into a pass.

[Scope and limitations](validation.md#glm-full-distributed-recovery) ·
[Machine-readable evidence](../benchmarks/results/glm-full-tensorfold-recovery-20261102.json)

## Earlier qualification runs

These are small runs on the configurations named below. They do not establish
full capability, refusal-rate or maximum-throughput results.

**The DeepSeek measurements below use the retired version 1.0 compact profile.**
The current native Keys starter uses a different artifact and engine path.

The [30 October 2026 aggregate results](../benchmarks/results/native-qualification-20261030.json)
include model revisions, engine and profile hashes, recorded settings, sample
counts and timing values. They omit prompts, response logs and private
deployment configuration.

### DeepSeek: the same tasks in both modes

Setup: DeepSeek V4.1 Flash 1017, EXL3 3.5-bpw, TensorFold 1.1.9, four
RTX PRO 6000 Blackwell Max-Q GPUs at a 325 W power limit per GPU. DSpark
and CUDA graphs were active. The speed runs used a 4,096-token context, an
8,192-token pool and one live slot. The profile was
`deepseek41-flash1017-v1` at strength 1.

Each row has three measured requests per mode, after one warmup pair per run.
These selected tasks produced equal output lengths in both modes. The prose
prompts are the `linux-process` and `photography` cases in the
[original benchmark fixtures](../benchmarks/demo-cases.jsonl).

| Task | Output tokens per request | Off decode tok/s | On decode tok/s | Off TTFT | On TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linux process explanation | 256 | 180.19 | 177.74 | 0.457 s | 0.465 s |
| Portrait photography checklist | 256 | 150.29 | 136.11 | 0.458 s | 0.445 s |
| Merge-intervals code | 230 | 278.74 | 258.40 | 0.399 s | 0.441 s |

Values are arithmetic means from sequential requests. Decode speed is the
engine's per-request rate, excluding prefill; TTFT includes client, queue and
network time. Output and draft acceptance affect speed. The 64K preset, other
power limits and concurrent throughput were not measured here.

The retained summary does not contain the complete generation-parameter
manifest, limiting reproducibility. The code case completed with the same
output hash in both modes, but only two of its three generated self-tests
passed in either mode.

### DeepSeek: 64K input and separate cache reuse

The same engine binary and checkpoint also ran a 65,536-token context preset,
a 131,072-token pool and one live slot with DSpark and CUDA graphs active. An
original fictional reading fixture produced 63,886 input tokens. Both modes
returned the expected retrieval answer on all six requests.

| Cache state | Off TTFT | On TTFT | Cached input tokens |
| --- | ---: | ---: | ---: |
| First request in each mode | 15.585 s | 14.809 s | 0 |
| Two later requests per mode | 0.665–0.871 s | 0.609–0.653 s | 63,872 |

The model was already loaded. Requests alternated modes; each mode first
reported no cached prefix, then reused its own. Six-token answers make this a
cache and retrieval check, with no long-output decode or general reasoning score.

### Gemma: a small GSM8K pilot

TensorFold with Gemma 4 26B A4B, MLX-format 4-bit weights and DFlash active on one
Linux/NVIDIA DGX Spark scored **14/16 off** and **15/16 on**. MLX describes the
checkpoint format; the runtime was TensorFold. The test used the first 16
GSM8K test questions, five examples per question and a 512-token output budget.

The [Gemma result file](../benchmarks/results/gemma-native-gsm8k-pilot.json) records
the dataset, evaluator, model and engine revisions and extraction metrics.

### Qwen: runtime checks and a GSM8K pilot

Qwen3.8-27B EXL3 2.0-bpw passed 26 TensorFold checks on one SM120 GPU:
16 prose/code fixture requests, four structured-output/tool cases, and six
authorization or concurrent-request cases. DFlash2 stayed active, with FP8 KV,
two streams and a 4,096-token context.

A separate GSM8K pilot scored **31/32 off** and **31/32 on**, using the first
32 test questions, five-shot prompting, seed 20261030 and a 512-token output
budget. DFlash2 remained active. Both strict and flexible answer extraction
gave the same counts. The [pilot record](../benchmarks/results/qwen-native-gsm8k-pilot.json)
contains the exact model, engine, dataset and evaluator revisions.

The two GSM8K pilots use small, non-random samples and do not establish general
quality gains or parity. Broader capability and cross-model results are not
published. See [validation](validation.md) for other model and engine checks.

### Qwen SGLang: startup and TTFT

The BF16 Qwen3.8-27B SGLang 0.5.20 run used DFlash2 on one RTX PRO 6000
Blackwell Max-Q. Two individual warm streaming requests measured TTFT of
**0.113 seconds off** and **0.139 seconds on**. Startup took **295 seconds**
with first-time kernel compilation and **33 seconds** with those kernels cached.
These are individual samples from that deployment, not latency distributions.

### llama.cpp: short prose comparison

These runs used llama.cpp 0.8.0 at upstream commit
`8642541b255bfb5dc39cd351a178be09013a55d6`, the pinned Q8_0 checkpoints, one RTX PRO 6000
Blackwell Max-Q per model, two slots and a 2 GiB prompt cache. Each row reports
the median of two warm requests per mode after one warmup. Requests ran
sequentially with no instrumented build or simultaneous generation.

| Model and draft method | Output tokens | Off decode tok/s | On decode tok/s | Off TTFT | On TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 26B A4B, MTP | 256 | 264.66 | 235.05 | 0.076 s | 0.089 s |
| Gemma 4 26B A4B, ngram | 256 | 208.96 | 198.51 | 0.076 s | 0.073 s |
| Qwen3.8-27B, external draft | 256 | 44.01 | 41.18 | 0.192 s | 0.129 s |

The [llama.cpp result file](../benchmarks/results/llamacpp-qualification-20261030.json)
contains the original library-explanation prompt, generation settings and
individual measurements. The 256-token cap cuts off the explanation. Decode
rate comes from the engine; TTFT comes from the streaming client. Thinking was
off. Ngram had no draft matches on this prompt; a separate copy fixture
confirmed active drafting. GPU power limits were observed at 325 W after the
run, with no continuous power record. Two samples per mode are insufficient
to establish a general performance difference.
