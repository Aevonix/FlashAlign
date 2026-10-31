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

This integration modifies separately installed SGLang source under Apache-2.0.
Retain the upstream license and notices with prepared copies.

## Retained SGLang source fixture

`tests/fixtures/tilelang_kernel.pinned.txt` reproduces
`python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py` from
[SGLang revision 94602c9c2b7cbdb8efd5c52802dac6a1c180089e](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e).
Its SHA-256 is
`ea9afa8b2dfeed08a4840c0d82622909bac15bf8a752f084ba9d260421c573bf`.
`kernel_schedule.py` applies a narrow dispatch change against this preimage.
`LICENSE.sglang` is the complete Apache-2.0 text copied from that revision's
root `LICENSE`. Retain it with the fixture and modified upstream copies.
The upstream fixture remains under its own terms; FlashAlign's license
covers its authorized additions.
