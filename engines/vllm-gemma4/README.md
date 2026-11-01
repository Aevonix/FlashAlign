# FlashAlign for Gemma 4

This vLLM variant applies standard projection profiles to Gemma 4 attention and
MLP outputs before their post-projection normalization. It implements the same
authenticated request and cache-isolation contract as the other FlashAlign
backends. Tool permissions remain with the caller.

The pinned checkpoint, dependency versions and single-GPU layout are recorded
in `backend.json`. Use the included `gemma4-26b-a4b-v1` starter profile with a
separately downloaded model checkpoint.

Defaults are a 65,536-token context, two concurrent sequences and ngram
speculation. These defaults have not been tested together on GB10; its recorded
run used one sequence and 0.60 GPU memory utilization. See
[validation](../../docs/validation.md) for long-input results and the recorded
answer-quality failures.

Native images reach the `gemma4` model through Hermes and OpenClaw. The
`gemma4_text` variant is text only.

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
stop setup without disabling them. GPU startup requires the CUDA build
prerequisites listed in `backend.json`.
