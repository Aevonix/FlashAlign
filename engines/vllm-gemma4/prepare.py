# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Install this backend in a new, owned virtual environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def source_digest():
    digest = hashlib.sha256()
    files = [ROOT/'backend.json', ROOT/'pyproject.toml', ROOT/'requirements.txt', *sorted((ROOT/'flashalign_vllm').glob('*.py'))]
    for path in files:
        digest.update(path.name.encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    receipt = output/'installation.json'
    if output.exists():
        if receipt.exists():
            data = json.loads(receipt.read_text())
            if (data.get('backend') == 'vllm' and data.get('variant') == 'gemma4'
                    and data.get('source_digest') == source_digest()):
                print(json.dumps(data)); return
        raise ValueError('Use a new output directory for this isolated backend')
    output.mkdir(parents=True)
    subprocess.run([sys.executable, '-m', 'venv', str(output/'venv')], check=True)
    python = output/'venv/bin/python'
    environment = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')}
    environment.update(PIP_CACHE_DIR=str(output/'pip-cache'), PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run([str(python), '-m', 'pip', '--isolated', 'install', '--index-url', 'https://pypi.org/simple',
                    '-r', str(ROOT/'requirements.txt'), str(ROOT)], env=environment, check=True)
    data = {'backend': 'vllm', 'variant': 'gemma4', 'python': str(python),
            'launch': str(ROOT/'launch.py'), 'vllm_version': '0.32.0', 'source_digest': source_digest()}
    receipt.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data))


if __name__ == '__main__':
    main()
