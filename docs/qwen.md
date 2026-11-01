# Qwen3.8-27B

FlashAlign includes the experimental `qwen38-27b-v1` profile.
Select it locally with the wizard or by name in an agent setup plan.

For TensorFold, the wizard selects a pinned 1.1.9 runtime bundle for the model
format and hardware. The catalog includes these tested configurations:

| Model format | Hardware | Draft model |
| --- | --- | --- |
| MLX-format 4-bit | One DGX Spark, Linux aarch64, SM121 | DFlash2 |
| EXL3 2.0-bpw | One RTX PRO 6000, Linux x86_64, SM120 | DFlash2 |

The model and DFlash2 draft model must already be present on the model host.

```sh
bash ./install.sh --auto --backend tensorfold --model-path /models/qwen38-27b \
  --drafter /models/qwen38-dflash2 --devices 0
```

For offline setup, add `--engine-bundle /packages/tensorfold-qwen38` to use an
extracted compatible bundle. Runtime retrieval downloads the engine package
from FlashAlign release assets without a GitHub login; profiles are included
locally and model weights must be provided separately.

Both TensorFold configurations run on one NVIDIA GPU with the external DFlash2
drafter. They support standard and owner requests, concurrent batches, and
separate in-memory prefix caches for each profile and strength. The main model's
embedding, attention and MLP outputs are covered. Native MTP and tensor parallel
owner execution are not
supported by these bundles. Setup stops if the active speculative configuration
is incompatible, preserving that configuration.

The TensorFold serving path is text only. See [validation](validation.md) for
runtime and matrix results, and [TensorFold setup](native-tensorfold.md) for
installation requirements.

For agent setup, use `--config /private/setup.json --apply` to prepare files
without starting the engine or restarting a gateway. See
[agent setup](agent-setup.md) and [managed engine setup](engine-setup.md).
