# Third-party software and artifact notices

FlashAlign's software license covers the rights Aevonix owns or is authorized
to license. Other material retains the terms identified below. An Aevonix
commercial agreement does not replace licenses Aevonix has no authority to
replace.

## Inference engines and harnesses

| Component | Distribution and applicable terms |
| --- | --- |
| [TensorFold](https://github.com/ashhart/TensorFold) | Native runtime bundles include source from the exact modified TensorFold revision, with its upstream notices and the separately identified FlashAlign additions. Upstream `LICENSE`, `NOTICE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md` are retained. TensorFold uses Apache-2.0, with earlier and vendored portions retaining MIT or other stated terms. Only the authorized FlashAlign additions carry FlashAlign's software license. |
| [vLLM](https://github.com/vllm-project/vllm) | Separately installed under Apache-2.0. Integration packages contain FlashAlign glue and pinned source checks, not a complete vLLM distribution. The DeepSeek SM120 adaptation identifies its upstream source in the module and provenance file. |
| [SGLang](https://github.com/sgl-project/sglang) | Separately installed source under Apache-2.0. Setup prepares a pinned source copy with FlashAlign additions. Retain the upstream license and notices with that copy. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Separately obtained and built under MIT. The integration includes a patch and a copy of the upstream license at `engines/llamacpp/LICENSE.llama.cpp`; retain it with resulting builds. |
| [Hermes](https://github.com/NousResearch/hermes-agent) | Separately installed host under MIT, copyright Nous Research. The FlashAlign adapter does not include the complete host. |
| [OpenClaw](https://github.com/openclaw/openclaw) | Separately installed host under MIT, copyright OpenClaw Foundation. The FlashAlign adapter uses its native plugin and terminal interfaces. |

TensorFold's nested notices identify its bundled adaptations and sources,
including MLX and mlx-lm, mlx-vlm, z-lab DFlash, DeepSeek encoding helpers,
Transformers-derived helpers and the MiaAI-Lab vision patch. Preserve those
notices when redistributing that runtime. They are not relicensed as exclusively
Aevonix work.

## Starter profiles and model weights

The experimental starter profiles are fitted from abliterated checkpoints by
OrcaRouter, Dealign and MoonMath. Fitting and runtime packaging are by Aevonix.
The [checkpoint source record](profiles/SOURCES.md) credits each provider and
pins the exact edited and stock revisions.

The starter pack contains runtime profile parameters, not complete model or
draft-model checkpoints. Its retained source terms are available beside each
profile:

- [DeepSeek V4.1 Flash 1017](profiles/deepseek41-flash1017-v1/LICENSE): MIT.
- [GLM 5.3 Full](profiles/glm53-full-v1/LICENSE): the custom GLM-5.3 license.
- [GLM 5.3 Flash](profiles/glm53-flash-v1/LICENSE): MIT.
- [Qwen3.8-27B](profiles/qwen38-27b-v1/LICENSE): Apache-2.0.
- [Gemma 4 26B A4B](profiles/gemma4-26b-a4b-v1/LICENSE): Apache-2.0.

The GLM Full source license includes additional conditions, including a
security-review condition for specified commercial uses. Read the full text
before deployment. Model and draft-model checkpoints obtained separately have
their own terms, which can differ from an engine's license. A supported runtime
does not establish permission to redistribute every compatible checkpoint.

## Installed dependencies

The dependency declarations in `pyproject.toml`, `hermes-owner-mode/requirements*.txt`
and each engine package record the installed packages and version constraints.
They include HTTP and configuration libraries, NumPy, safetensors,
Transformers, PyTorch, Triton, FlashInfer and engine-specific packages. These
are installed separately under their respective licenses; their binaries are
not included in the FlashAlign source archive. Preserve their notices when
distributing an environment or container containing them.

NVIDIA drivers, CUDA libraries and related SDK components are separately
installed under NVIDIA's applicable licenses. They are not included in the
FlashAlign source archive. This notice does not grant rights to redistribute
them or change their terms.

## Artwork and fonts

FlashAlign logos use outlined Geist lettering. The font's SIL Open Font
License is retained in the brand kit; no font binary is embedded in the logos.
See the [brand assets](assets/brand/README.md) and
[trademark policy](TRADEMARKS.md). Software licensing does not grant permission
to imply Aevonix endorsement.

If you identify a missing notice, contact **contact@aevonix.com** with the
component and version.
