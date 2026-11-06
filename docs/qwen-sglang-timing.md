# Qwen/SGLang: graph Owner Mode timing

The opt-in graph Owner Mode path delivered **1.324× end-to-end completion-token
throughput** compared with the previous release's eager Owner Mode path, with
a conditional 95% interval of **[1.313, 1.335]**.

This comparison used Qwen3.8-27B, a 258-token input and exactly 256 output
tokens. Throughput is output tokens divided by the whole request time,
including initial latency. It is not a decode-only tok/s measurement or a
comparison of FlashAlign off versus on.

## What was compared

| Arm | FlashAlign revision | Owner Mode execution |
| --- | --- | --- |
| A | `979222c00070001126eaf3634d22ec03111de399` | Released eager path |
| B | `c3379354dbaa350c86bdeb7c62a05e5a6fde5ad0` | Graph path |

Both arms used the same BF16 Qwen/Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, SGLang revision
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, and `qwen38-27b-v1` at strength 1.0.
DFlash2 with eight draft tokens remained active. One nominal 96 GB NVIDIA RTX
PRO 6000 Blackwell Max-Q ran at a 325 W power limit, with a 65,536-token context,
two allowed concurrent requests and diagnostics disabled.

There were three independent paired process blocks in the predeclared order
AB / AB / BA. Each arm completed 30 primary requests per block, for 90 per arm.
Six separate stock starts supplied bracketing controls. Across all measured
conditions and controls, the study recorded 1,440 requests in 1,260 windows.

Primary requests ran sequentially on a loaded engine with a cold prefix.
Warmups were excluded. Generation used temperature 0, seed 1729, thinking
disabled, `ignore_eos=true`, and both minimum and maximum output tokens set to
256. These settings keep output length fixed; the result does not describe
natural answer lengths.

## Uncertainty and limits

The interval uses the three paired process-block log ratios and Student's t
distribution with two degrees of freedom. It assumes independent blocks and
approximately normal log ratios; three pairs cannot validate those assumptions.
Individual requests were not treated as independent experimental replicates.

The primary throughput gate and all ten p95 latency and two sampled-memory
gates passed their predeclared bounds. The primary lower interval endpoint had
to be at least 1.05; latency and memory upper endpoints had to be at most 1.05.
Sampled steady-memory peaks do not establish instantaneous or startup peaks.

The separate [fixed-workload quality follow-up](qwen-sglang-quality.md)
passed its completion gate but did not establish all capability gates.
The timing data alone do not establish answer quality, DeepSeek performance,
other-engine performance or the performance of every later build. The
[FR443 refusal comparison](qwen-fr443-results.md) is a separate experiment.

## Recompute the results

The [evidence bundle](../benchmarks/evidence/qwen-sglang-20261101/README.md)
includes the analysis code, aggregate request records, pseudonymized timing
and memory observations, source identities and synthetic workload. It needs
only Python 3.12's standard library:

```sh
cd benchmarks/evidence/qwen-sglang-20261101
python -B recompute.py
python -B -m unittest -v test_recompute
```

The verifier checks the saved hashes, assignments, observations and all 13
intervals against the reference report. The export supports numerical and
telemetry recomputation. It does not recreate the removed collection receipts,
process identities or a full GPU-serving run. It contains no model weights,
model answers, credentials, endpoint addresses or private deployment paths.
