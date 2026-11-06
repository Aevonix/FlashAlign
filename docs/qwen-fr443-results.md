# Qwen3.8-27B: automated refusal results

On the pinned 443-case [Blackfrost FR443 suite](https://github.com/Blackfrost-AI/r1-false-refusal-443/tree/a2a8051a3f4ff2593f6b6e9c22f9929b3244c742),
the judge labeled **289/443 stock answers as refusals (65.24%)** and **11/443
profile answers as refusals (2.48%)**. All 886 answers and judgments were valid;
there were no uncertain judgments.

The paired change, profile minus stock, was **−62.75 percentage points**, with
a 95% interval of **[−67.19, −58.24] percentage points**. The interval uses the
predeclared percentile bootstrap over 441 unique prompt-text clusters: 2,000
resamples, seed 20261101. Both repeated source prompts remain in the case totals.

This measures judge-rated refusal. It does not score answer correctness,
usefulness or task success. The corpus mixes safe, harmful and unsafe requests;
it is not a benign-only over-refusal test.

## Subsets

| Source subset | Cases | Refusals, stock | Refusals, profile |
| --- | ---: | ---: | ---: |
| AdvBench | 146 | 114 (78.08%) | 0 (0.00%) |
| StrongREJECT | 149 | 102 (68.46%) | 6 (4.03%) |
| XSTest | 148 | 73 (49.32%) | 5 (3.38%) |

| Source label | Cases | Refusals, stock | Refusals, profile |
| --- | ---: | ---: | ---: |
| harmful | 295 | 216 (73.22%) | 6 (2.03%) |
| safe | 75 | 1 (1.33%) | 0 (0.00%) |
| unsafe | 73 | 72 (98.63%) | 5 (6.85%) |

The two tables describe the same 443 cases and must not be added together. The
profile arm had one capability-limitation judgment, counted as non-refusal by
the fixed rubric. The stock arm had none.

## Configuration and scoring

- Qwen/Qwen3.8-27B BF16, revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- SGLang 0.5.20, revision `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
- Profile `qwen38-27b-v1`, strength 1.0, on one NVIDIA RTX PRO 6000 Blackwell
  Max-Q per subject deployment. CUDA graphs and DFlash2 with eight draft tokens
  were active. Context was 65,536 tokens, with at most two running requests.
- Subject requests used temperature 0, top-p 1, seed 20261101, a 32,768-token
  output allowance and request-level thinking disabled.
- The judge ran in a separate stock deployment of the same pinned Qwen model,
  without a FlashAlign profile. It used temperature 0, top-p 1, a 1,024-token
  JSON verdict allowance and thinking disabled.

The serving FlashAlign base was
`c3379354dbaa350c86bdeb7c62a05e5a6fde5ad0`. The collection and scoring bundle
applied seven recorded benchmark/client overlays from
`951260d373e31b0e3d5145012031d9df10435c65`; the whole serving build was not
replaced by that later commit.

Collection used `flashalign.strict-completion.v2`; judging used
`flashalign.fourway-judge.v2` with the pinned Blackfrost four-way rubric and a
durable attempt ledger. Uncertain judgments invalidate the primary score.
Capability limitations count as non-refusals. At most three transport attempts
were allowed; answers were not rerolled for a preferred judgment.

**Human audit is pending:** the predeclared sample is 40 cases, ten per
dataset/label stratum. Its disagreements will be reported separately without
replacing these automatic labels. General capability retention has not been
established by this test. [Performance was measured separately](qwen-sglang-timing.md).

The [machine-readable evidence](../benchmarks/results/qwen-fr443-20261101.json)
contains exact profile, corpus, rubric, build and protocol identities, all
subset counts and the confidence interval. It contains no prompts or answers.
See [the benchmark guide](benchmarks.md#fr443-refusal-comparison) to run the
same protocol with your own deployment.
