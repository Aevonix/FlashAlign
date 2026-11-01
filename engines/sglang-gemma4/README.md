# Gemma 4 on SGLang

This experimental integration applies the unchanged `gemma4-26b-a4b-v1` profile
to the pinned `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` checkpoint. It uses the
SGLang 0.5.20 source revision
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, declared in `backend.json`.

The profile covers 50 attention-output and dense-MLP-output projections. Hooks
run before Gemma's post-projection normalization. Stock rows bypass the
projection; authenticated profile and strength selections use separate cache
namespaces. The native MoE branch is unchanged.

NGRAM speculative decoding with three draft tokens remains active. This variant requires one GPU,
eager execution and non-overlapping scheduling. It does not accept external
draft models, LoRA composition, distributed workers, HiCache or session-cache
reuse. The Qwen SGLang integration remains separate.

For managed setup with an existing local checkpoint:

```sh
bash ./install.sh --engine-only --backend sglang \
  --model-path /models/gemma4-26b-a4b-fp8
```

The runtime checks model identity, all 50 writers and the supplied payload
checksums. The wizard checks the declared checkpoint metadata and preserves
compatible active NGRAM settings without asking for a draft model.

The tested GB10 configuration used a 65,536-token context, two concurrent
requests and 0.60 memory fraction. Direct protocol and answer checks passed;
the native Hermes strict-answer check failed on one of four cases.
Long-context inputs, image quality, OpenClaw and full wizard engine activation
remain untested for this pairing.
See [validation](../../docs/validation.md) for the results.
