# FlashAlign starter profiles

FlashAlign provides five experimental starter profiles. Qwen, Gemma and both
GLM profiles are compact artifacts included in this checkout. The DeepSeek
profile contains exact native weight tensors and uses a 1.09 GB public release
asset. Setup downloads that payload to an external cache and verifies its size
and SHA-256 before staging it.

| Model | Profile | Current runtime |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | `deepseek41-flash1017-v1` | [vLLM](../engines/vllm-deepseek-native/README.md), [TensorFold](../docs/native-tensorfold.md) |
| GLM 5.3 Full | `glm53-full-v1` | TensorFold |
| GLM 5.3 Flash | `glm53-flash-v1` | TensorFold and SGLang |
| Qwen3.8-27B | `qwen38-27b-v1` | TensorFold, vLLM, SGLang and llama.cpp |
| Gemma 4 26B A4B | `gemma4-26b-a4b-v1` | TensorFold, vLLM, SGLang and llama.cpp |

The DeepSeek starter uses 26 WO-B matrices and their scale tensors, with strength
fixed at **1.0**. It requires a native operator backend; compact projection
loaders cannot load it. See [DeepSeek requirements](deepseek41-flash1017-v1/README.md)
for the supported checkpoints and current four-GPU configuration.

Use the exact checkpoint revision and serving format declared by the profile
and backend. See [compatibility](../docs/compatibility.md) and
[validation results](../docs/validation.md).

## Attribution and terms

Source checkpoints and exact revisions are listed in [SOURCES.md](SOURCES.md).
Source edits belong to their checkpoint providers. Fitting, where applicable,
and FlashAlign runtime packaging are by Aevonix.

Each profile retains its model-source license. The DeepSeek source also has a
[responsible-use agreement](deepseek41-flash1017-v1/RESPONSIBLE_USE.md).
These terms are separate from the FlashAlign software's
[AGPL-3.0-only license](../LICENSE) and [commercial option](../COMMERCIAL.md).

Model checkpoints must be obtained separately. Profile creation tools are not
included in this release and are not required for loading the starter profiles.
