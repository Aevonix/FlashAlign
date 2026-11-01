# Download a model

FlashAlign includes four compact starter profiles and the DeepSeek native
profile manifest. Setup fetches the DeepSeek profile's 1.09 GB payload from a
pinned public release asset. Base model weights and draft models are separate
downloads. Choose the checkpoint for your engine and hardware
before running the wizard. The [compatibility table](compatibility.md#engines-and-models)
lists supported combinations and their requirements.

The commands below use the official [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
If `hf` is not installed, install `huggingface_hub` in your Python environment:

```sh
python -m pip install --upgrade huggingface_hub
```

Store weights outside the FlashAlign checkout. Run this once in the shell where
you will use the commands below:

```sh
FLASHALIGN_MODELS="$HOME/flashalign-models"
mkdir -p "$FLASHALIGN_MODELS"
```

Each download is pinned to a commit. To inspect a download without retrieving
the weight files, append `--dry-run` to its command. If a model page requires
accepting access terms, do that on the linked model page and use `hf auth login`
in your terminal. Do not put access tokens in commands, setup plans or the repo.

## TensorFold checkpoints

Choose your model's section below. These are the checkpoint variants declared
by the bundled TensorFold profiles.
Multi-node model files must be accessible at the configured model path on every
rank. See [TensorFold setup](native-tensorfold.md) for rank configuration and
activation.

| Starter model | Download below | Tested hardware |
| --- | --- | --- |
| [DeepSeek V4.1 Flash 1017](#deepseek-v41-flash-1017-on-tensorfold-120) | EXL3 3.5-bpw plus original Engram tables | Four RTX PRO 6000 GPUs, Linux x86_64 |
| [GLM 5.3 Full](#glm-53-full) | EXL3 2.75-bpw | Four DGX Sparks, Linux aarch64 |
| [GLM 5.3 Flash](#glm-53-flash) | NVIDIA NVFP4 | Four DGX Sparks, Linux aarch64 |
| [Qwen3.8-27B](#qwen38-27b) | EXL3 2.0-bpw or MLX-format 4-bit, plus DFlash2 | One RTX PRO 6000 or one DGX Spark, respectively |
| [Gemma 4 26B A4B](#gemma-4-26b-a4b) | MLX-format 4-bit plus DFlash | One DGX Spark, Linux aarch64 |

The selected bundle enforces its declared GPU architecture and rank count.
MLX-format describes the checkpoint files; TensorFold runs them on NVIDIA GPUs
under Linux.

### DeepSeek V4.1 Flash 1017 on TensorFold 1.2.0

The `deepseek41-flash1017-v1` starter uses the same unchanged native payload
on TensorFold 1.2.0 and [native vLLM](#deepseek-v41-flash-1017-native-vllm).
TensorFold requires the following EXL3 base; vLLM uses the official FP8 checkpoint.

Download the [EXL3 checkpoint](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw/tree/04a492fc3b4d25ddbab8f2d2f07f235ec77c9010):

```sh
hf download LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw \
  --revision 04a492fc3b4d25ddbab8f2d2f07f235ec77c9010 \
  --local-dir "$FLASHALIGN_MODELS/deepseek41-exl3"
```

Its original Engram tables are also required. Download only the original
configuration, index and two required shards from the
[stock checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170):

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  config.json model.safetensors.index.json \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir "$FLASHALIGN_MODELS/deepseek41-engram"
```

Pass the EXL3 folder as `--model-path` and the original-shard folder as
`--engram-source`:

```sh
bash ./install.sh --auto --backend tensorfold \
  --model-path "$FLASHALIGN_MODELS/deepseek41-exl3" \
  --profile ./profiles/deepseek41-flash1017-v1 \
  --engram-source "$FLASHALIGN_MODELS/deepseek41-engram"
```

The wizard requires a TensorFold 1.2.0 bundle declaring this native operator
and exact EXL3 base. This bundle requires four SM120 GPUs and strength 1.0.
Its 1.09 GB profile payload is fetched separately from the runtime bundle.

### GLM 5.3 Full

Download the [EXL3 checkpoint](https://huggingface.co/Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold/tree/cf1fee8923b770b0c47f7a032e1b0f30a57c626a):

```sh
hf download Mia-AiLab/GLM-5.3-EXL3-2.75bpw-TensorFold \
  --revision cf1fee8923b770b0c47f7a032e1b0f30a57c626a \
  --local-dir "$FLASHALIGN_MODELS/glm53-full-exl3"
```

This checkpoint includes the native MTP weights. Prepare it with the wizard:

```sh
bash ./install.sh --auto --backend tensorfold \
  --model-path "$FLASHALIGN_MODELS/glm53-full-exl3"
```

### GLM 5.3 Flash

Download the [NVIDIA NVFP4 checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/09b04e5e74bca08ca8549fc736d4cdd8624bfde3):

```sh
hf download nvidia/GLM-5.3-Flash-NVFP4 \
  --revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3 \
  --local-dir "$FLASHALIGN_MODELS/glm53-flash-nvfp4"
```

This checkpoint includes the native MTP weights:

```sh
bash ./install.sh --auto --backend tensorfold \
  --model-path "$FLASHALIGN_MODELS/glm53-flash-nvfp4"
```

The SGLang alternative uses the original FP8 checkpoint below.

### Qwen3.8-27B

For the Linux x86_64 TensorFold bundle, download the
[EXL3 checkpoint](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw/tree/b4925724092da59826c02d25a7201fb272bf4f34):

```sh
hf download Mia-AiLab/Qwen3.8-27B-EXL3-2.0bpw \
  --revision b4925724092da59826c02d25a7201fb272bf4f34 \
  --local-dir "$FLASHALIGN_MODELS/qwen38-exl3"
```

For the DGX Spark TensorFold bundle, use the declared
[MLX-format checkpoint](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit/tree/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c):

```sh
hf download TensorFold/Qwen3.8-27B-MLX-4bit \
  --revision 22d8d538154e0e5b6f8dcb1fcc74a60a0798104c \
  --local-dir "$FLASHALIGN_MODELS/qwen38-mlx4"
```

Both TensorFold variants use the external
[DFlash2 draft model](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/tree/50307d4c4cde6860d4eee73e2547cd786fe8e8a4):

```sh
hf download z-lab/Qwen3.8-27B-DFlash2 \
  --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir "$FLASHALIGN_MODELS/qwen38-dflash2"
```

For example, prepare the x86_64 variant with:

```sh
bash ./install.sh --auto --backend tensorfold \
  --model-path "$FLASHALIGN_MODELS/qwen38-exl3" \
  --drafter "$FLASHALIGN_MODELS/qwen38-dflash2"
```

Use `qwen38-mlx4` as the model path on DGX Spark. vLLM and SGLang use the
original BF16 checkpoint; llama.cpp uses Q8_0 GGUF. Those downloads are below.

### Gemma 4 26B A4B

For TensorFold on DGX Spark, download the
[MLX-format checkpoint](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit/tree/0d77464eeb233a2da68ebf9d7dc4edaac7db956d)
and its [DFlash draft model](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash/tree/77d4202772dfe50b2396ec7bac9cfffc7b9e7057):

```sh
hf download mlx-community/gemma-4-26b-a4b-it-4bit \
  --revision 0d77464eeb233a2da68ebf9d7dc4edaac7db956d \
  --local-dir "$FLASHALIGN_MODELS/gemma4-mlx4"
hf download z-lab/gemma-4-26B-A4B-it-DFlash \
  --revision 77d4202772dfe50b2396ec7bac9cfffc7b9e7057 \
  --local-dir "$FLASHALIGN_MODELS/gemma4-dflash"
```

```sh
bash ./install.sh --auto --backend tensorfold \
  --model-path "$FLASHALIGN_MODELS/gemma4-mlx4" \
  --drafter "$FLASHALIGN_MODELS/gemma4-dflash"
```

For other supported NVIDIA setups, use the FP8 or Q8_0 alternatives below.

## vLLM and SGLang checkpoints

Use these original checkpoints for the declared backend paths. Do not pass the
TensorFold EXL3, NVFP4 or MLX-format downloads to an integration that expects
these files.

| Model | Engine | Pinned format |
| --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | Pinned native vLLM | Official native FP8 checkpoint |
| GLM 5.3 Flash | SGLang | Original FP8 checkpoint |
| Qwen3.8-27B | vLLM or SGLang | Original BF16 checkpoint |
| Gemma 4 26B A4B | vLLM or SGLang | RedHat FP8-dynamic checkpoint |

### DeepSeek V4.1 Flash 1017 native vLLM

Download the complete official checkpoint, including its Engram and native
DSpark tensors:

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir "$FLASHALIGN_MODELS/deepseek41-original"
```

Prepare the native starter on Linux x86_64 with four SM120 GPUs:

```sh
bash ./install.sh --auto --backend vllm \
  --model-path "$FLASHALIGN_MODELS/deepseek41-original" --no-restart
```

Setup selects the pinned native container, retrieves and verifies the public
profile payload, and stops before activation with `--no-restart`. No edited
full checkpoint or separate draft download is needed. For offline preparation,
place the verified profile payload beside its manifest first. See the
[profile instructions](../profiles/deepseek41-flash1017-v1/README.md).

### Other vLLM and SGLang models

GLM 5.3 Flash, SGLang:

```sh
hf download zai-org/GLM-5.3-Flash \
  --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c \
  --local-dir "$FLASHALIGN_MODELS/glm53-flash-fp8"
```

Qwen3.8-27B, vLLM or SGLang:

```sh
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "$FLASHALIGN_MODELS/qwen38-bf16"
```

Gemma 4 26B A4B, vLLM or SGLang:

```sh
hf download RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic \
  --revision ed35d7abe5d940da41b4ff06eb482feb0be8cb44 \
  --local-dir "$FLASHALIGN_MODELS/gemma4-fp8"
```

Qwen on SGLang also uses the
`qwen38-dflash2` download above. vLLM Qwen's declared default uses ngram
speculation, which needs no separate draft checkpoint. For engine selection,
resource requirements and activation, see [engine setup](engine-setup.md).

## llama.cpp checkpoints

These are the declared Q8_0 files. Their GGUF format differs from the folders
used above.

Qwen3.8-27B:

```sh
hf download ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf \
  --revision a316ea527fde32c12487be7574e2db5ac2ad2a7e \
  --local-dir "$FLASHALIGN_MODELS/qwen38-gguf"
```

The Qwen `draft-simple` path also uses this smaller external draft
model. Download the pinned [Qwen3.5-0.8B Q4_0 GGUF](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/tree/807dfd9a6fadbf703207aac3bfb0b2b3040d077c):

```sh
hf download ggml-org/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_0.gguf \
  --revision 807dfd9a6fadbf703207aac3bfb0b2b3040d077c \
  --local-dir "$FLASHALIGN_MODELS/qwen38-gguf-draft"
bash ./install.sh --engine-only --backend llamacpp \
  --model-path "$FLASHALIGN_MODELS/qwen38-gguf/Qwen3.8-27B-Q8_0.gguf" \
  --drafter "$FLASHALIGN_MODELS/qwen38-gguf-draft/Qwen3.5-0.8B-Q4_0.gguf"
```

Gemma 4 26B A4B:

```sh
hf download ggml-org/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q8_0.gguf \
  --revision 9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2 \
  --local-dir "$FLASHALIGN_MODELS/gemma4-gguf"
```

For example, prepare the Gemma server with:

```sh
bash ./install.sh --engine-only --backend llamacpp \
  --model-path "$FLASHALIGN_MODELS/gemma4-gguf/gemma-4-26B-A4B-it-Q8_0.gguf"
```

Gemma defaults to ngram speculation, which needs no separate draft model. Its
optional official MTP draft is available from the same pinned repository:

```sh
hf download ggml-org/gemma-4-26B-A4B-it-GGUF mtp-gemma-4-26B-A4B-it-Q8_0.gguf \
  --revision 9cf8c8db0f83fb4582f15381f8fb1a7c1b0b9ca2 \
  --local-dir "$FLASHALIGN_MODELS/gemma4-gguf"
```

Pass that file with `--drafter` to select the MTP path. See the
[llama.cpp guide](../engines/llamacpp/README.md) for launch options. Setup preserves
an existing supported speculative configuration rather than disabling it.
