# FlashAlign for Gemma 4

This vLLM variant applies standard projection profiles to Gemma 4 attention and
MLP outputs before their post-projection normalization. It implements the same
authenticated request and cache-isolation contract as the other FlashAlign
backends. Tool permissions remain with the caller.

The tested checkpoint, metadata fingerprints, dependency versions, single-GPU
layout and evaluation scope are recorded in `backend.json`. This is an
experimental integration. The Aevonix Research starter profile
`gemma4-26b-a4b-v1` is included under `profiles/`. Engine and native harness
protocol checks passed with the answer-quality limits below. Model weights
must be obtained separately.

The launcher defaults use a 65,536-token context, two concurrent sequences and
ngram speculation. An earlier run on one 96 GB NVIDIA RTX PRO 6000 passed a
bounded 59,987-token retrieval check in both modes. The latest native harness
run used one GB10, one concurrent sequence and 0.60 GPU memory utilization,
with all 50 sites attached and three-token ngram active. All 11 protocol checks
passed. Earlier 59,045-token retrieval passes are retained;
the latest GB10 run does not validate the two-sequence, 0.85-memory defaults.

Native images reach the tested `gemma4` model through Hermes and OpenClaw.
The `gemma4_text` variant remains text only. Exact-answer misses, unexpected
vision-tool calls and one reasoning-budget recovery remain recorded in
[validation](../../docs/validation.md). Correct profile activation does not
guarantee an answer matches its target.

Use the FlashAlign wizard with the included profile:

```sh
bash ./install.sh --engine-only --backend vllm \
  --model-path /models/gemma4-fp8 --profile profiles/gemma4-26b-a4b-v1
```

The wizard selects this variant from the model family and profile transform.
It checks the pinned configuration and weight-header layout before preparing
the deployment. Those checks establish compatibility, not full weight-file
integrity. The profile's source-model terms still apply separately.

The launcher configures the Gemma reasoning/tool parsers and preserves a
supported active ngram configuration. Unsupported active speculative methods
stop setup without disabling them. A fresh isolated dependency installation
passed pinned-version, import, entrypoint and CPU operator checks. Serving
checks used the separately prepared GPU environment; GPU startup still needs
the documented CUDA build prerequisites. See `backend.json` for the measured
validation limits.
