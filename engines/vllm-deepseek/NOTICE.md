# FlashAlign software notice

Copyright (c) 2026 Aevonix Inc and the respective contributors.

FlashAlign software is licensed under AGPL-3.0-only; the full terms are in
LICENSE. Aevonix Inc also offers a separate commercial license for rights it
owns or is authorized to license: https://github.com/Aevonix/FlashAlign/blob/main/COMMERCIAL.md
Contact: contact@aevonix.com.

Contributions use the submission-based terms at
https://github.com/Aevonix/FlashAlign/blob/main/CLA.md. No separate signed form is required.
Third-party engine, dependency, model, profile and artwork terms remain separate:
https://github.com/Aevonix/FlashAlign/blob/main/THIRD_PARTY_NOTICES.md

## Integration provenance

# Runtime source notice

This package includes generic FlashAlign request, replay, profile-loading and
engine hooks under the repository license. It contains no profile tensors or
model weights. The native adapter targets the separately installed, Apache-2.0
licensed vLLM 0.32.0 source at the hashes in `flashalign_deepseek/source-pins.json`.
The optional FlashInfer dependency is separately installed into an owned overlay
from the pinned package and hash in `requirements-sm120.txt`; its own license
continues to apply. DeepSeek model weights and supplied profiles retain their
separate source terms.

The `flashalign_vllm/core.py` compatibility module is copied from the same
consumer source revision, without its Qwen plugin. The installed plugin entry
point is `flashalign_deepseek` only. `UPSTREAM_PROVENANCE.json` identifies the upstream
source-only runtime receipt before wizard packaging; `sha256.json` identifies
the complete packaged kit.

## Retained vLLM license

`flashalign_deepseek/sm120.py` adapts the SM12x page-size declarations from
[vLLM PR #60762](https://github.com/vllm-project/vllm/pull/60762). Its target
source is vLLM 0.32.0, revision
`5df63a7203cdb9328d824a27b682e77c7b9928be`; the exact affected file hashes are
in `flashalign_deepseek/source-pins.json`. `LICENSE.vllm` is the complete
Apache-2.0 text copied from that revision's root `LICENSE`. Retain this text
and provenance with redistributed adaptations. FlashAlign's license covers
its authorized additions and does not replace the upstream terms.
