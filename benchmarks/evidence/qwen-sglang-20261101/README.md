# Qwen / SGLang timing evidence

This bundle recomputes a saved comparison of the released eager owner path
(`979222c00070001126eaf3634d22ec03111de399`) and the graph owner path
(`c3379354dbaa350c86bdeb7c62a05e5a6fde5ad0`). It contains no model weights,
model answers, credentials, endpoints or model-launch configuration.

Run offline with Python 3.12 and its standard library:

```sh
python -B recompute.py
python -B -m unittest -v test_recompute
```

The verifier checks file hashes, all 12 assigned starts, ordered request pairs,
usage and profile metadata, active speculation, complete host observations,
the original memory coverage validator and exact equality with the reference
analysis. It recomputes all 13 intervals and their classifications. No GPU,
network access, installed FlashAlign package or model execution is required.

The primary result is 1.3239068138922923 times the end-to-end completion-token
throughput, with conditional 95% interval
[1.3132339787697629, 1.3346663886296917]. Throughput is actual final completion
tokens divided by summed request send-to-done time, including initial latency.
It is not decode-only throughput. The input is 258 tokens and the output is
exactly 256 tokens. Five conditions retain all assigned measurements; loaded-off
and bracketing stock controls are separate from the owner estimate.

There are three independent A/B process pairs, 30 primary requests per arm per
pair, and six separate stock starts. The predeclared order is AB / AB / BA.
The intervals use three paired process-block log ratios, Student t with two
degrees of freedom, and the assumptions of independent blocks and approximately
normal log ratios. Three pairs cannot validate those assumptions. The primary
lower endpoint must be at least 1.05; each of ten p95 latency and two memory upper
endpoints must be at most 1.05. All passed. Stock stationarity uses the fixed 10%
limit. Memory sampling requires at least 99% paired coverage and gaps no larger
than 250 ms. Sampled steady peaks do not establish instantaneous or startup peaks.

`provenance.json` identifies the measured source, model, draft, profile, client
files and serving settings. This was one nominal 96 GB RTX PRO 6000 Blackwell
Max-Q at a 325 W limit, BF16 Qwen3.8-27B, DFlash2 enabled, TP1, 65,536 context,
two running requests, and diagnostics disabled. `workload.json` contains the
reviewed synthetic counting/padding prompts, rendered token IDs, tokenizer
hashes, seeds, order, sampling settings and warmup counts.
`environment.json` records the pinned environment's package versions and native
library hashes, using paths relative to the environment rather than host paths.
These are serving provenance; offline recomputation needs only Python.

`analysis-input.json` contains all 1,440 measured requests in 1,260 windows.
`telemetry.json.gz` contains the six owner memory sample streams and all 12 host
observation streams. Identity labels preserve equality and distinct process
starts; a common clock offset was subtracted while preserving every duration
exactly. The builder checked the original archive, raw begin/end records and
corrected report before export. Their hashes are retained in provenance.

This is a derived, allowlisted dataset, not the original archive. Removed PID
birth records, launch receipts and source trees cannot be independently audited
from this bundle. Source/identity/collection checks in the analysis input are
receipt-backed assertions; the offline verifier does not recreate those checks.
This minimal bundle supports numerical and telemetry recomputation, not a fully
self-contained rerun of model serving. It establishes no DeepSeek, cross-engine
or entire later-release performance claim, nor a quality or completion-regression
confidence result. The latter remains `not_established`.

The original and corrected analyzers and shared contract are verbatim. Legacy
schema names and experiment-mode labels remain for compatibility. See
`ORDER-ERRATUM.md` for the order-validation correction. No observations,
thresholds or formulas changed. The expected report omits only its original
input/analyzer hash fields, which belong to the pre-export files; those hashes
are recorded separately in provenance and `SHA256SUMS`.

Copyright (C) 2026 Aevonix Inc. Aevonix-authored code and bundle content are
licensed AGPL-3.0-only; see `LICENSE` and `NOTICE`. No profile-creation tooling
is included. Referenced model and tokenizer assets remain separately licensed.
