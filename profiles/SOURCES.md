# Starter profile checkpoint sources

Experimental test profiles based on abliterated checkpoints from
[Keys / drowzeys](https://huggingface.co/drowzeys),
[OrcaRouter](https://huggingface.co/orcarouter),
[Dealign](https://huggingface.co/dealignai), and
[MoonMath](https://huggingface.co/MoonMath-ai).
Fitting, where applicable, and runtime packaging by Aevonix.

The revisions below identify the edited fitting targets and matched stock
references. Supported serving checkpoints and quantizations are listed in
each profile manifest and backend. Compact profiles are approximations; the
DeepSeek native profile retains its source tensors. Neither establishes full
model-output equivalence by itself. See [validation](../docs/validation.md).

## DeepSeek V4.1 Flash 1017

Profile: `deepseek41-flash1017-v1`.

Source edit: [drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated/tree/de412d2249632903dace23399526e00a145aada5).
Revision: `de412d2249632903dace23399526e00a145aada5`.

Audited serving base: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
Revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The source edit does not pin its original stock revision.

This native profile packages the source's exact 26 WO-B matrices and 26 scale
tensors. It replaces the earlier compact DeepSeek starter. The source card
declares MIT and includes a separate responsible-use agreement. Retain both the
[MIT notice](deepseek41-flash1017-v1/LICENSE) and
[agreement](deepseek41-flash1017-v1/RESPONSIBLE_USE.md).
The source edit is by drowzeys; runtime packaging is by Aevonix.

## GLM 5.3 Full

Profile: `glm53-full-v1`.

Edited checkpoint: [MoonMath-ai/GLM-5.3-Uncensored-FP8](https://huggingface.co/MoonMath-ai/GLM-5.3-Uncensored-FP8/tree/4aaa244f0a1ca704e70d96b460eb18fb6b94ee76) by MoonMath.
Revision: `4aaa244f0a1ca704e70d96b460eb18fb6b94ee76`.

Stock reference: [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/tree/ba32f8c19c8738e2c57c71798c601d477a7bf844).
Revision: `ba32f8c19c8738e2c57c71798c601d477a7bf844`.

Both pinned source cards declare the custom **GLM-5.3** license (Hugging Face
metadata tag `other`). The complete [retained license](glm53-full-v1/LICENSE)
matches the pinned [stock license](https://huggingface.co/zai-org/GLM-5.3/blob/ba32f8c19c8738e2c57c71798c601d477a7bf844/LICENSE)
and [edited license](https://huggingface.co/MoonMath-ai/GLM-5.3-Uncensored-FP8/blob/4aaa244f0a1ca704e70d96b460eb18fb6b94ee76/LICENSE).
`other` is a metadata category, not an absent license. The manifest identifies
these terms as `LicenseRef-GLM-5.3`; they are separate from FlashAlign's AGPL.

## GLM 5.3 Flash

Profile: `glm53-flash-v1`.

Edited checkpoint: [dealignai/GLM-5.3-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8/tree/d5822b068da8ddb301665b48c58718e6e18a2c3a) by Dealign.
Revision: `d5822b068da8ddb301665b48c58718e6e18a2c3a`.

Stock reference: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/46a050bec13002ad29ee795c16e61cf4fa3c4a9c).
Revision: `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.

Both pinned source cards declare MIT. The retained terms are in the
[profile's MIT notice](glm53-flash-v1/LICENSE).

## Qwen3.8-27B

Profile: `qwen38-27b-v1`.

Edited checkpoint: [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored/tree/0f639a9bde23fe229a7b064b40a5153c5bec0b48) by OrcaRouter.
Revision: `0f639a9bde23fe229a7b064b40a5153c5bec0b48`.

Stock reference: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).
Revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Both pinned source cards declare Apache 2.0. The retained terms are in the
[profile's Apache 2.0 notice](qwen38-27b-v1/LICENSE).

## Gemma 4 26B A4B

Profile: `gemma4-26b-a4b-v1`.

Edited checkpoint: [orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8](https://huggingface.co/orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8/tree/0098988c0a77389b0c1e80754bf410487869ae27) by OrcaRouter.
Revision: `0098988c0a77389b0c1e80754bf410487869ae27`.

Stock reference: [RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic/tree/ed35d7abe5d940da41b4ff06eb482feb0be8cb44).
Revision: `ed35d7abe5d940da41b4ff06eb482feb0be8cb44`.

Base model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52),
revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.

The pinned base and stock cards declare Apache 2.0. Google also links Gemma 4
to its [Apache 2.0 license](https://ai.google.dev/gemma/apache_2), separately
from the [terms for earlier Gemma models](https://ai.google.dev/gemma/terms).
The profile retains its existing [Apache 2.0 notice](gemma4-26b-a4b-v1/LICENSE).

The pinned OrcaRouter edited card instead declares `gemma`. Its
[source declaration](https://huggingface.co/orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8/blob/0098988c0a77389b0c1e80754bf410487869ae27/README.md)
is recorded as `source.edited.license` in the manifest, separately from the
Apache 2.0 declarations for the base and stock models.

## Attribution and model terms

Credit for the abliterated checkpoints belongs to the providers above.
Aevonix fitted and packaged the FlashAlign runtime artifacts. The starter
pack contains profile parameters, not complete model checkpoints.
Keep the applicable [model notices](README.md#attribution-and-terms) with
redistributed profiles. Checkpoint downloads may have separate access terms.

Compact manifests separate source declarations from `license_notice`. The
native DeepSeek directory retains its source declarations and terms alongside
the manifest and download descriptor. FlashAlign software has its own
[AGPL-3.0-only license](../LICENSE) and [commercial licensing option](../COMMERCIAL.md).
