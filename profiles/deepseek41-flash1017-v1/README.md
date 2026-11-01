# DeepSeek V4.1 Flash 1017 native profile

This starter applies the exact native WO-B weights and scales from
[drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated/tree/de412d2249632903dace23399526e00a145aada5),
revision `de412d2249632903dace23399526e00a145aada5`.
The source edit covers attention layers 10–35. Runtime packaging is by Aevonix.

The profile format is `flashalign.native_wo_b.v1`, transform
`native_matrix_override_v1`. Strength is fixed at **1.0**. The artifact contains
26 native FP8 matrices and 26 UE8M0 scale tensors. It is not a direction-vector
file and cannot be loaded by a projection-only backend.

## Payload

The manifest and download descriptor are included here. The native payload is
`wo_b_l10_35_1017.safetensors`, 1,091,589,280 bytes, with SHA-256:

```text
88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864
```

Setup retrieves the [public payload](https://github.com/Aevonix/FlashAlign/releases/download/v1.1.0/wo_b_l10_35_1017.safetensors)
when needed, verifies it, and stages it outside the source checkout. For an
offline deployment, place that verified file beside `profile.json` before
running setup. The offline inspector never downloads missing files.

## Behavior and compatibility

Authorized requests select the edited WO-B matrices. Normal requests retain
the stock matrices. Mixed batches select the correct operation per row, with
separate cache identities. Speculative decoding stays enabled.

The current backends require Linux x86_64 and four SM120 GPUs. Both load the
same profile payload with these pinned serving checkpoints:

| Backend | Checkpoint | Revision |
| --- | --- | --- |
| [vLLM](../../engines/vllm-deepseek-native/README.md) | `deepseek-ai/DeepSeek-V4.1-Flash-1017` (official FP8) | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| [TensorFold 1.2.0](../../docs/native-tensorfold.md) | [`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw/tree/04a492fc3b4d25ddbab8f2d2f07f235ec77c9010) | `04a492fc3b4d25ddbab8f2d2f07f235ec77c9010` |

TensorFold keeps the EXL3 stock path and selects the profile's FP8 matrices for
owner rows. Its base quantization and arithmetic differ from vLLM. The source's
EXL3 sidecars cannot replace this payload. SGLang and llama.cpp do not implement
this native operator.

See [validation](../../docs/validation.md) for measured results and limits.
The software, model and profile-source terms are separate. Retain the
[MIT notice](LICENSE) and [source agreement](RESPONSIBLE_USE.md).

On this TensorFold build, JSON-schema and guided output requests are rejected
in both modes. Chat and completion requests support stock and authorized profile modes.
