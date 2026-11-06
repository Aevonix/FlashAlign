# Qwen/SGLang: graph Owner Mode quality check

The fixed comparison **did not establish all of the prespecified capability gates**.
Completion and GSM8K passed at the first look. IFEval and MBPP did not establish
their 2-percentage-point conservative loss bounds at the fixed maximum. Eager execution
remains the default; the graph path remains opt-in.

## Results

| Metric | Paired cases | Eager successes | Graph successes | Losses | Upper loss bound | Margin | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Completion | 500 | 500 | 500 | 0 | 0.00917 | 0.01 | Passed |
| GSM8K | 500 | 480 | 482 | 0 | 0.00917 | 0.02 | Passed |
| IFEval | 541 | 438 | 443 | 8 | 0.02732 | 0.02 | Not established |
| MBPP | 499 | 361 | 361 | 6 | 0.02438 | 0.02 | Not established |

These are success counts under released eager Owner Mode and candidate graph
Owner Mode. Both arms had the same profile active. A loss means eager succeeded
and graph failed; candidate-only wins are retained in the paired counts but do
not offset losses for this gate. The
[aggregate JSON](../benchmarks/results/qwen-sglang-quality-20261102.json) contains both prespecified looks and all four
paired outcome counts.

## Comparison and rule

Both arms used BF16 Qwen3.8-27B, SGLang 0.5.20, `qwen38-27b-v1` at strength 1.0,
DFlash2 with eight draft tokens and a 65,536-token context. Requests used
temperature 0, seed 1729 and thinking disabled. Each fresh process pair ran the
two arms on separate RTX PRO 6000 Blackwell Max-Q GPUs, with one request at a
time per arm. There were ten completed paired process blocks and 2,040 original attempts
per arm. This compares two active Owner Mode execution paths.

The fixed sufficient criterion inverts the exact one-sided binomial lower tail
for the probability of an eager-only success. Its upper bound must be strictly
below 0.02 for each capability metric; its negative conservatively bounds the
candidate-minus-eager accuracy difference. Completion requires a new-failure
bound strictly below 0.01. The method can leave a positive net accuracy change
without an established gate when it trades wins for losses.

The two prespecified looks spend alpha 0.01 then 0.04. Completion and GSM8K
stopped at their first-look samples of 500. Only the untouched IFEval and MBPP
remainders were collected, reaching 541 and 499 distinct prompt units. IFEval
already had six first-look losses; its fixed maximum-look cutoff permits at
most four. Those originals were retained, and neither the rule nor threshold
was changed. No third look or response regeneration was performed.

## Scoring and limits

GSM8K uses strict numeric extraction; IFEval uses strict prompt accuracy. MBPP
uses one program per prompt and the pinned source setup before its reference
assertions, in a nonroot, network-disabled sandbox. One duplicate MBPP source
prompt was handled before outcomes using a fixed representative. The frozen
chat and code-setup adapters do not assert identical unmodified leaderboard
conditions. An offline input-readability failure was corrected through file
permissions, preserving the failed attempt and all scorer source and response bytes.

The confidence interpretation assumes independent prompt units conditional on
the fixed model and runtime. Benchmark families and process blocks can remain
correlated. Historical training or profile-selection overlap cannot be
excluded. These fixed-workload results do not establish general capability
parity, edited-checkpoint equivalence, full-context reasoning or results for
other models, profiles, engines, hardware or every later source build. The coverage is neither
distribution-free nor cluster-robust.

The [timing result](qwen-sglang-timing.md) and
[stock-versus-profile refusal result](qwen-fr443-results.md) are separate
experiments. This aggregate export omits prompts, model answers and private
collection receipts. See [release validation](validation.md) for other checks
and outstanding qualification limits.
