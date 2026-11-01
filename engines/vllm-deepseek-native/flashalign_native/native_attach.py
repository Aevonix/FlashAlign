# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Native DeepSeek attachment and immutable per-worker coverage receipts."""
import json
import os
from pathlib import Path
import threading
import time

from flashalign_native_wo.execution import install_native_weights
from flashalign_native_wo.loader import NativePayload
from flashalign_native_wo.runtime import NativeRuntime


_RECORDS = {}
_LOCK = threading.Lock()


def _receipt(runtime, rank, layer_receipt):
    directory = Path(os.environ.get('FLASHALIGN_NATIVE_RECEIPTS', ''))
    if (not directory.is_absolute() or not directory.is_dir() or directory.is_symlink()
            or directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077):
        raise ValueError('Precreated native receipt directory required')
    profile = runtime.profile
    pid = os.getpid()
    with _LOCK:
        key = (str(directory), pid, rank, profile.digest)
        record = _RECORDS.setdefault(key, {
            'kind': 'native_wo_b_worker_coverage', 'pid': pid, 'tp_rank': rank,
            'tp_world_size': profile.spec.tp_world_size, 'profile': profile.name,
            'profile_digest': profile.digest, 'manifest_sha256': profile.spec.raw_sha256,
            'payload_sha256': profile.spec.payload_sha256, 'layers': {}, 'complete': False,
        })
        layer = str(layer_receipt['layer'])
        previous = record['layers'].get(layer)
        if previous is not None and previous != layer_receipt:
            raise ValueError('Native worker layer binding changed')
        record['layers'][layer] = layer_receipt
        record['complete'] = set(record['layers']) == {str(n) for n in profile.spec.layers}
        record['updated_unix'] = time.time()
        name = f'rank-{rank}-pid-{pid}.json'
        temp = f'.{name}.{threading.get_ident()}.{time.time_ns()}.tmp'
        dirfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, 'O_NOFOLLOW', 0), 0o600, dir_fd=dirfd)
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump(record, stream, indent=2, allow_nan=False)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp, name, src_dir_fd=dirfd, dst_dir_fd=dirfd)
                os.fsync(dirfd)
            except BaseException:
                try:
                    os.unlink(temp, dir_fd=dirfd)
                except FileNotFoundError:
                    pass
                raise
        finally:
            os.close(dirfd)


def ensure_native_weights(attention):
    runtime = getattr(attention, '_flashalign_native_runtime', None)
    if runtime is None or attention.layer_id not in runtime.profile.spec.layers:
        return
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != runtime.profile.spec.tp_world_size:
        raise ValueError('Native profile tensor-parallel geometry differs')
    rank = get_tensor_model_parallel_rank()
    receipt = install_native_weights(attention, runtime.profile.spec,
                                     runtime._native_payload, rank)
    if receipt is None:
        raise ValueError('Native target did not produce an installation receipt')
    _receipt(runtime, rank, receipt)


def wrap_native_setup(attention_class):
    original = attention_class.setup_wo_projection
    if getattr(original, '_flashalign_native_setup', False):
        return
    from functools import wraps

    @wraps(original)
    def setup(self, *args, **kwargs):
        runtime = getattr(self, '_flashalign_native_runtime', None)
        if runtime is not None and hasattr(self, '_flashalign_native_weights'):
            # Preserve the original immutable packed object on repeat setup.
            ensure_native_weights(self)
            return None
        result = original(self, *args, **kwargs)
        ensure_native_weights(self)
        return result

    setup._flashalign_native_setup = True
    attention_class.setup_wo_projection = setup


def attach_native(runner, profile):
    base = runner.model.language_model.model
    if type(base).__module__ != 'vllm.models.deepseek_v4_1.nvidia.model':
        raise ValueError('Native profile requires the pinned B12X model')
    if getattr(runner, '_flashalign_runtime', None) is not None:
        raise ValueError('Native and output-projection runtimes cannot be combined')
    runner._flashalign_profiles = {profile.name: profile}
    runner._flashalign_selections = {}
    runner._flashalign_mm_types = {}
    runtime = NativeRuntime(runner, profile, runner._flashalign_selections)
    runtime._native_payload = NativePayload(profile.directory, profile.spec)
    runner._flashalign_native_runtime = runtime
    runner.model._flashalign_native_runtime = runtime
    try:
        for index, layer in enumerate(base.layers):
            if getattr(layer, '_flashalign_runtime', None) is not None:
                raise ValueError('Native layer has an output-projection runtime')
            layer._flashalign_native_runtime = runtime
            layer.attn._flashalign_native_runtime = runtime
            layer.attn._flashalign_native_profile = profile.spec
            if layer.attn.layer_id != index:
                raise ValueError('Native main-layer numbering differs from the profile')
            if layer.attn._wo_projection_weights is not None:
                ensure_native_weights(layer.attn)
    except BaseException:
        runtime._native_payload.close()
        raise
    return runtime
