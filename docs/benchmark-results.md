# Measured results

These are small qualification runs from 30 October 2026. They show measured
behavior on the listed configurations, not a score for every supported model
or a maximum-throughput claim. [Run your own comparison](benchmarks.md) with the
included recorder and optional evaluation tools.

The [aggregate results](../benchmarks/results/native-qualification-20261030.json)
include exact model revisions, engine and profile hashes, settings recorded by
the checks, sample counts and timing values. Prompts, response logs and private
deployment configuration are not included in that file.

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

Values are arithmetic means. Decode speed is the native engine's reported
per-request decode rate; it excludes prefill. TTFT is measured by the client
and includes request, queue and network time. These are sequential requests,
not a concurrent server-throughput test. Speed depends on the output and draft
acceptance. These measurements do not establish speed at the 64K preset or at
another GPU power limit.

The retained summary does not contain the complete generation-parameter
manifest for these speed runs. Treat them as qualification measurements rather
than a fully reproducible benchmark report. The code case completed in both
modes and produced the same output hash, but only two of its three generated
self-tests passed in either mode. Timing does not establish coding accuracy.

## DeepSeek: 64K input and separate cache reuse

The same engine binary and checkpoint also ran a 65,536-token context preset,
a 131,072-token pool and one live slot with DSpark and CUDA graphs active. An
original fictional reading fixture produced 63,886 input tokens. Both modes
returned the expected retrieval answer on all six requests.

| Cache state | Off TTFT | On TTFT | Cached input tokens |
| --- | ---: | ---: | ---: |
| First request in each mode | 15.585 s | 14.809 s | 0 |
| Two later requests per mode | 0.665–0.871 s | 0.609–0.653 s | 63,872 |

“First” means no prefix reused for that request. The model was already loaded;
this is not a cold model-start measurement. The requests alternated modes.
Each mode initially reported no cached prefix, then reused its own prefix.
The six-token answers make this a cache and retrieval check, not a useful
long-output decode benchmark or a general long-context reasoning score.

## Gemma: a small GSM8K pilot

TensorFold with Gemma 4 26B A4B, MLX-format 4-bit weights and DFlash active on one
Linux/NVIDIA DGX Spark scored **14/16 off** and **15/16 on**. This checkpoint format
does not imply use of Apple's MLX runtime. The test used the first 16 GSM8K test
questions, five examples per question and a 512-token output budget.

This fixed-prefix pilot does not establish a quality improvement. The
[Gemma result file](../benchmarks/results/gemma-native-gsm8k-pilot.json) records
the dataset, evaluator, model and engine revisions and extraction metrics.

## Qwen: runtime checks and a GSM8K pilot

Qwen3.8-27B EXL3 2.0-bpw passed 26 TensorFold checks on one SM120 GPU:
16 prose/code fixture requests, four structured-output/tool cases, and six
authorization or concurrent-request cases. DFlash2 stayed active, with FP8 KV,
two streams and a 4,096-token context. This is functional coverage, not a
capability benchmark or a throughput measurement.

A separate GSM8K pilot scored **31/32 off** and **31/32 on**, using the first
32 test questions, five-shot prompting, seed 20261030 and a 512-token output
budget. DFlash2 remained active. Both strict and flexible answer extraction
gave the same counts. The [pilot record](../benchmarks/results/qwen-native-gsm8k-pilot.json)
contains the exact model, engine, dataset and evaluator revisions. This small
fixed-prefix sample does not establish full-benchmark parity.

Full refusal-rate, over-refusal and cross-model capability benchmarks are not
published yet. See [validation](validation.md) for the other model and engine
checks, including failures and limits.

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
individual measurements. The fixed output cap measures a 256-token prefix,
not completion of the requested explanation. Decode rate is reported by the
engine; TTFT is measured by the streaming client. GPU power limits were
observed after the run at 325 W; they were not sampled throughout it.
Thinking was off. Ngram produced no draft matches on this prose prompt; its
active drafting was verified separately with a copy fixture.
Two samples per mode do not establish a general performance difference,
maximum throughput or answer quality.
