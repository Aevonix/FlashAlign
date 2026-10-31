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

The experimental runtime was checked on one GB10 with a fresh dependency
environment, a configured 65,536-token context, two concurrent requests and
0.60 memory fraction. All 50 sites were observed. Mixed stock/profile rows,
authentication, replay rejection and cache separation at strengths 0.5 and
1.0 passed. NGRAM verification and draft proposals remained active. Six direct
arithmetic, factual and JSON checks passed across stock and profile modes.

Four actual Hermes native protocol cases passed in five model requests,
including revocation on a reused agent and a real `read_file` call. Three
answers matched exactly. The initial approved no-tool case appended an
unsolicited introduction after the requested number; the strict native quality
result remains failed.

These checks used short prompts. Long-context capacity, image quality,
OpenClaw and full wizard engine activation were not tested for this pairing.
See [validation](../../docs/validation.md) for the other engines' results.
