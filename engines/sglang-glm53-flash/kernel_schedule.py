# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Select the measured lower-memory GLM attention schedule on SM120/SM121."""
import hashlib

PATH = 'python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py'
PREIMAGE_SHA256 = 'ea9afa8b2dfeed08a4840c0d82622909bac15bf8a752f084ba9d260421c573bf'
BEFORE = '        kernel = kernel_factory(num_heads, d_v, tail_dim, topk, sm_scale=sm_scale)'
AFTER = '''        kernel_options = {"sm_scale": sm_scale}
        if (
            tail_dim == 0
            and d_v == 512
            and num_heads == 16
            and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(q.device) in ((12, 0), (12, 1))
        ):
            # The GLM Flash TP4 no-RoPE BF16 lane was numerically checked on
            # SM120 and SM121 with this unchanged factory and lower-memory schedule.
            kernel_options.update(block_I=64, num_stages=1)
        kernel = kernel_factory(num_heads, d_v, tail_dim, topk, **kernel_options)'''


def apply_schedule_patch(raw):
    if hashlib.sha256(raw).hexdigest() != PREIMAGE_SHA256:
        raise ValueError('Unsupported SGLang attention kernel preimage')
    text = raw.decode()
    if text.count(BEFORE) != 1:
        raise ValueError('Ambiguous SGLang attention dispatch')
    return text.replace(BEFORE, AFTER).encode()
