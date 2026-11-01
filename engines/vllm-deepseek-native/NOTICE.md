# Notices

FlashAlign runtime additions are copyright 2026 Aevonix Inc and the respective contributors, licensed under AGPL-3.0-only. Commercial licensing is described in the repository's `COMMERCIAL.md`.

This backend includes modified source files from [vLLM](https://github.com/local-inference-lab/vllm/tree/6d66eed35), pinned to commit `6d66eed35`. Original copyright and Apache-2.0 headers remain in each file. See `LICENSE.vllm` for the full Apache-2.0 license. The modifications add explicit profile routing, separate graph identities and native matrix bindings.

The pinned image also includes [FlashInfer](https://github.com/local-inference-lab/flashinfer/tree/79819b112f8b), under Apache-2.0. `LICENSE.flashinfer` contains that license. No FlashInfer source is copied into this directory.

The model and profile source checkpoints have their own terms. Those terms and exact revisions are listed with the profile in `profiles/SOURCES.md`. The FlashAlign software license does not relicense model weights.
