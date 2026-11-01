# Measured results

These are small qualification runs. Results apply to the listed
configurations; they are not full capability, refusal-rate or
maximum-throughput benchmarks. [Run your own comparison](benchmarks.md) with
the included recorder and optional evaluation tools.

**The DeepSeek measurements use the retired version 1.0 compact profile.**
The current native Keys starter uses a different artifact and engine path.

The [30 October 2026 aggregate results](../benchmarks/results/native-qualification-20261030.json)
include model revisions, engine and profile hashes, recorded settings, sample
counts and timing values. They omit prompts, response logs and private
deployment configuration.

## DeepSeek: the same tasks in both modes

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

## DeepSeek: 64K input and separate cache reuse

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

## Gemma: a small GSM8K pilot

TensorFold with Gemma 4 26B A4B, MLX-format 4-bit weights and DFlash active on one
Linux/NVIDIA DGX Spark scored **14/16 off** and **15/16 on**. MLX describes the
checkpoint format; the runtime was TensorFold. The test used the first 16
GSM8K test questions, five examples per question and a 512-token output budget.

The [Gemma result file](../benchmarks/results/gemma-native-gsm8k-pilot.json) records
the dataset, evaluator, model and engine revisions and extraction metrics.

## Qwen: runtime checks and a GSM8K pilot

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
quality gains or parity. Full refusal-rate, over-refusal and cross-model
capability results are not published. See [validation](validation.md) for
other model and engine checks.

## Qwen SGLang: startup and TTFT

The BF16 Qwen3.8-27B SGLang 0.5.20 run used DFlash2 on one RTX PRO 6000
Blackwell Max-Q. Two individual warm streaming requests measured TTFT of
**0.113 seconds off** and **0.139 seconds on**. Startup took **295 seconds**
with first-time kernel compilation and **33 seconds** with those kernels cached.
These are individual samples from that deployment, not latency distributions.

## llama.cpp: short prose comparison

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
