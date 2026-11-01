# FlashAlign starter profiles

FlashAlign provides five experimental starter profiles. Qwen, Gemma and both
GLM profiles are compact artifacts included in this checkout. The DeepSeek
profile contains exact native weight tensors and uses a 1.09 GB public release
asset. Setup downloads that payload to an external cache and verifies its size
and SHA-256 before staging it. No private repository access is required.

| Model | Profile | Current runtime |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | `deepseek41-flash1017-v1` | [Pinned native vLLM](../engines/vllm-deepseek-native/README.md), official FP8 checkpoint, four SM120 GPUs |
| GLM 5.3 Full | `glm53-full-v1` | TensorFold |
| GLM 5.3 Flash | `glm53-flash-v1` | TensorFold and SGLang |
| Qwen3.8-27B | `qwen38-27b-v1` | TensorFold, vLLM, SGLang and llama.cpp |
| Gemma 4 26B A4B | `gemma4-26b-a4b-v1` | TensorFold, vLLM, SGLang and llama.cpp |

The current DeepSeek starter uses the Keys source's 26 WO-B matrices and their
26 scale tensors. It applies them only to authorized rows and leaves normal
rows on the stock path. It supports strength **1.0**. It requires the native
operator backend; compact projection loaders and the existing TensorFold/EXL3
path cannot load this artifact. See [DeepSeek details](deepseek41-flash1017-v1/README.md).

Use the exact checkpoint revision and serving format declared by the profile
and backend. [Validation](../docs/validation.md) distinguishes file integrity,
engine behavior and answer quality. Successful loading does not establish
full-model equivalence or a particular refusal rate.

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
