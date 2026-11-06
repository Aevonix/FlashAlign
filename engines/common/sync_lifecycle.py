# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Regenerate the dependency-free engine vendor module from its canonical source."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / 'hermes-owner-mode' / 'profile_lifecycle.py'
    target = root / 'engines/common/flashalign_lifecycle.py'
    if args.check:
        if not target.exists() or source.read_bytes() != target.read_bytes():
            raise SystemExit('Run python engines/common/sync_lifecycle.py')
    else:
        target.write_bytes(source.read_bytes())
    admin_source = root / 'hermes-owner-mode/profile_admin_server.py'
    admin_target = root / 'engines/common/flashalign_admin_server.py'
    if args.check:
        if not admin_target.exists() or admin_source.read_bytes() != admin_target.read_bytes():
            raise SystemExit('Run python engines/common/sync_lifecycle.py')
    else:
        admin_target.write_bytes(admin_source.read_bytes())
    for name, canonical in (('flashalign_lifecycle.py', target), ('flashalign_admin_server.py', admin_target)):
        output = root / 'engines/llamacpp' / name
        if args.check:
            if not output.exists() or output.read_bytes() != canonical.read_bytes():
                raise SystemExit('Run python engines/common/sync_lifecycle.py')
        else:
            output.write_bytes(canonical.read_bytes())
    generation_source = root / 'engines/common/vllm_generations.py'
    for engine in ('vllm', 'vllm-gemma4'):
        output = root / 'engines' / engine / 'flashalign_vllm/generations.py'
        if args.check:
            if not output.exists() or output.read_bytes() != generation_source.read_bytes():
                raise SystemExit('Run python engines/common/sync_lifecycle.py')
        else:
            output.write_bytes(generation_source.read_bytes())
    for engine in ('sglang', 'sglang-gemma4', 'sglang-glm53-flash', 'sglang-qwen4'):
        package = root / 'engines' / engine / 'flashalign_sglang'
        patch_source = root / 'engines/common/sglang_lifecycle_patches.py'
        patch_target = package.parent / 'lifecycle_patches.py'
        if args.check:
            if not patch_target.exists() or patch_source.read_bytes() != patch_target.read_bytes():
                raise SystemExit('Run python engines/common/sync_lifecycle.py')
        else:
            patch_target.write_bytes(patch_source.read_bytes())
        sources = {'flashalign_lifecycle.py': target, 'flashalign_admin_server.py': admin_target, **{f'lifecycle_{part}.py': root / 'engines/common' / f'sglang_lifecycle_{part}.py' for part in ('api', 'worker', 'support')}}
        for name, canonical in sources.items():
            output = package / name
            if args.check:
                if not output.exists() or canonical.read_bytes() != output.read_bytes():
                    raise SystemExit('Run python engines/common/sync_lifecycle.py')
            else:
                output.write_bytes(canonical.read_bytes())


if __name__ == '__main__':
    main()
