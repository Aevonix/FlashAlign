# FlashAlign starter profiles

Experimental test profiles based on abliterated checkpoints from
[OrcaRouter](https://huggingface.co/orcarouter),
[Dealign](https://huggingface.co/dealignai), and
[MoonMath](https://huggingface.co/MoonMath-ai).
Fitting and runtime packaging by Aevonix.

The [checkpoint sources](SOURCES.md) list exact versions for every profile.
The wizard selects and validates the included profiles locally. No profile
download or private-repository login is required.

| Model | Profile | Supported engines | Current validation |
| --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | `deepseek41-flash1017-v1` | TensorFold, vLLM | vLLM covers 85 sites. TensorFold runs 83 text sites with DSpark on four SM120 GPUs; 65,536-token preset, automatic tools only. |
| GLM 5.3 Full | `glm53-full-v1` | TensorFold | 29 input-delta sites. Four-host wizard, cache, auth, concurrent-request and automatic-tool checks passed with native MTP; configured context 8,192. |
| GLM 5.3 Flash | `glm53-flash-v1` | TensorFold, SGLang | 60 sites including MTP. TensorFold passed four-host wizard, cache, auth, harness-hook and grammar checks on DGX Sparks. |
| Qwen3.8-27B | `qwen38-27b-v1` | TensorFold, vLLM, SGLang, llama.cpp | 129 sites. TensorFold on ARM uses MLX 4-bit; on x86_64 it uses EXL3 2.0-bpw. Both TensorFold paths keep DFlash2 active. |
| Gemma 4 26B A4B | `gemma4-26b-a4b-v1` | TensorFold, vLLM, SGLang, llama.cpp | 50 sites. TensorFold uses MLX 4-bit with DFlash; llama.cpp uses the declared Q8_0 GGUF with ngram or MTP. See validation for agent answer-quality limits. |

Use the checkpoint revision and serving format declared by the selected profile
and engine integration. The llama.cpp integration declares its qualified GGUF
files in [backend metadata](../engines/llamacpp/backend.json).
Compatible checkpoint entries pin the complete configuration and the weight
index or shard header. A separate quantization fingerprint is optional and is
checked when supplied; the complete configuration hash also covers its
quantization settings.
See [validation](../docs/validation.md) for measured coverage and answer-quality limits.
Full edited-checkpoint equivalence has not been established. Saved-file checks
cover structure and integrity. Model quality, complete checkpoint parity and
compatibility with other engines require separate tests.

Automatic setup rejects supplied profiles whose artifact validation failed.

See [managed engine setup](../docs/engine-setup.md) for hardware, speculation and
context requirements. Setup preserves active speculative decoding and rejects
unsupported configurations. Additional profiles can be supplied with `--profile`
or `--profiles`; see [setup](../docs/setup.md).

## Attribution and terms

Aevonix fitted and packaged these runtime artifacts from the
[credited edited checkpoints](SOURCES.md). Each manifest identifies its fitting
target and the exact base or serving checkpoints it supports.
The retained model-source terms are stored beside each profile:
[DeepSeek](deepseek41-flash1017-v1/LICENSE),
[GLM Full](glm53-full-v1/LICENSE),
[GLM Flash](glm53-flash-v1/LICENSE),
[Qwen](qwen38-27b-v1/LICENSE), and
[Gemma](gemma4-26b-a4b-v1/LICENSE). GLM Full has its own source license;
it is not relicensed under the FlashAlign software terms.

Manifests record licensing at each source: `source.base` when present,
`source.stock`, and `source.edited`. The `license_notice` field identifies
the retained model-source `LICENSE` shipped with the profile, including its
hash. Keep that notice with redistributed profiles and consult the linked
terms for each source checkpoint.

Gemma 4's base and stock cards declare Apache 2.0, while the pinned OrcaRouter
edited card declares `gemma`. Both declarations are retained in the manifest
and explained in [checkpoint sources](SOURCES.md#gemma-4-26b-a4b).

The starter pack contains runtime parameters. Obtain checkpoint weights
separately under their applicable terms. FlashAlign's
[license](../LICENSE) applies to Aevonix's software and contributions, subject to
retained third-party notices.

Profile-creation tools are not yet released. Runtime use requires only the
supplied profile files.
