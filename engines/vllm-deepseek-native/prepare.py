# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare the pinned Docker image without starting a model or touching a key."""
import argparse
import json
import os
from pathlib import Path
from runpy import run_path
import shutil
import stat
import subprocess

IMAGE = run_path(str(Path(__file__).with_name('engine_config.py')))['IMAGE']


def docker_image():
    result = subprocess.run(['docker', 'image', 'inspect', '--format',
        '{"id":{{json .Id}},"digests":{{json .RepoDigests}},"architecture":{{json .Architecture}},"os":{{json .Os}}}',
        IMAGE], capture_output=True, text=True, timeout=30)
    if result.returncode:
        return None
    try:
        image = json.loads(result.stdout)
    except ValueError as error:
        raise ValueError('Docker returned invalid image metadata') from error
    if (not isinstance(image, dict) or image.get('architecture') != 'amd64'
            or image.get('os') != 'linux' or not isinstance(image.get('id'), str)
            or (IMAGE not in (image.get('digests') or [])
                and image['id'] != IMAGE.rsplit('@', 1)[1])):
        raise ValueError('Docker image does not match the pinned Linux x86_64 runtime')
    return image


def prepare_environment(state_dir, *, pull=False):
    state = Path(state_dir).expanduser().absolute()
    if state.is_symlink():
        raise ValueError('Engine state directory must not be a symlink')
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = state.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('Engine state directory must be owned and private')
    if shutil.which('docker') is None:
        raise ValueError('Install Docker Engine with NVIDIA Container Toolkit for this native backend')
    image = docker_image()
    if image is None:
        if not pull:
            raise ValueError('The pinned native image is not installed; allow dependency installation or pull it explicitly')
        log = state / 'image-install.log'
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'ab') as stream:
            result = subprocess.run(['docker', 'pull', IMAGE], stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError('Pinned image download failed; inspect the private image-install.log')
        image = docker_image()
        if image is None:
            raise ValueError('Pinned image was not available after download')
    return {'container_runtime': 'docker', 'image': IMAGE, 'image_id': image['id']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--pull', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare_environment(args.state_dir, pull=args.pull)))


if __name__ == '__main__':
    main()
