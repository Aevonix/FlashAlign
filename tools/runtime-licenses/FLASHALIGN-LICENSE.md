# Licensing of the accompanying native source

This addendum applies to `@SOURCE_ARCHIVE@`, the exact TensorFold source archive
from Git revision `@SOURCE_REVISION@`. The source archive is unchanged. This
addendum supplies licensing information alongside that original source snapshot.

TensorFold's upstream code and imported upstream commits retain their original
licenses and notices. The source archive's root `LICENSE` and `NOTICE`, and its
`LICENSES/` directory, preserve those terms.

FlashAlign additions marked `SPDX-License-Identifier: AGPL-3.0-only`, and the
FlashAlign modifications to retained upstream files, are distributed under
AGPL-3.0-only while retaining applicable upstream notices. The following
FlashAlign-owned files are also covered by AGPL-3.0-only, including snapshots
that predate their individual SPDX headers:

- `zig/kernels/cuda/flashalign.cu`
- `zig/src/cuda/flashalign.zig`
- `zig/src/families/qwen3_5/cuda_alignment.zig`
- `zig/src/native/cuda_alignment.zig`

The complete license text accompanies this package at
[licenses/FlashAlign/AGPL-3.0-only.txt](../licenses/FlashAlign/AGPL-3.0-only.txt).
Aevonix Inc also offers a [commercial license](../COMMERCIAL.md) for its covered
contributions. That option does not replace third-party obligations or grant
rights without a signed agreement.

No model weights or private profile-creation tools are included in this source
archive. Runtime profiles and model checkpoints retain their own terms.
