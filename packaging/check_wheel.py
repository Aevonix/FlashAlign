# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Install a wheel locally and verify setup without a checkout on its import path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


PROBE = r'''
import hashlib, json, pathlib, sys
installed, expected_file, scratch = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(installed))
import owner_mode
from owner_mode.backend_setup import backend_catalog, _copy_kit
from owner_mode.engine_adapters import engine_adapters
from owner_mode.profiles import bundled_profile_root, discover_profiles
from owner_mode.resources import setup_command, setup_resource
assert pathlib.Path(owner_mode.__file__).is_relative_to(installed)
expected = json.loads(expected_file.read_text())
assert not any(pathlib.Path(item).resolve().is_relative_to(pathlib.Path(expected['source_root']))
               for item in sys.path if item), 'Source checkout is on the installed probe path'
root = pathlib.Path(owner_mode.__file__).parent / '_assets'
actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in root.rglob('*') if p.is_file()}
assert actual == expected['files'], 'Packaged asset bytes differ from the source inventory'
for name, digest in expected['modules'].items():
    assert hashlib.sha256((installed / name).read_bytes()).hexdigest() == digest
catalog = backend_catalog()
identity = lambda row: (row['kind'], row.get('variant'))
assert sorted(map(identity, catalog), key=str) == sorted(map(tuple, expected['backends']), key=str)
assert set(engine_adapters().ids()) == {'tensorfold', 'vllm', 'sglang', 'llamacpp'}
assert all(pathlib.Path(row['kit']).is_relative_to(root) for row in catalog)
for index, row in enumerate(catalog):
    copied = _copy_kit(pathlib.Path(row['kit']), scratch / ('kit-' + str(index)))
    assert copied and 'backend.json' in copied
    assert not any('tests' in pathlib.Path(name).parts or pathlib.Path(name).name.startswith('test_')
                   for name in copied)
profiles, errors = discover_profiles([bundled_profile_root()])
assert not errors, errors
assert sorted(profile['name'] for profile in profiles) == expected['profiles']
assert (setup_resource('openclaw-owner-mode') / 'terminal-runner.mjs').is_file()
resume = setup_command()
assert resume == [sys.executable, str(pathlib.Path(owner_mode.__file__).with_name('setup.py'))]
print(json.dumps({'assets': len(actual), 'backend_kits': len(catalog),
    'runtime_modules': len(expected['modules']),
    'adapters': sorted(engine_adapters().ids()), 'profiles': len(profiles),
    'native_supervisor_sha256': actual['engines/vllm-deepseek-native/supervisor.py'],
    'shared_replay_sha256': actual['engines/common/flashalign_replay.py'],
    'resume': resume}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', default=sys.executable,
                        help='Python with the declared runtime dependencies already installed')
    args = parser.parse_args()
    source = args.source.resolve()
    files = json.loads((source / 'packaging/assets.json').read_text())['files']
    modules = {f'{target}/{path.name}': hashlib.sha256(path.read_bytes()).hexdigest()
               for directory, target in [('flashalign', 'flashalign'), ('hermes-owner-mode', 'owner_mode')]
               for path in (source / directory).glob('*.py')}
    expected = {'files': {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files},
                'modules': modules, 'source_root': str(source), 'backends': [], 'profiles': []}
    for name in files:
        if name.startswith('engines/') and name.endswith('/backend.json'):
            row = json.loads((source / name).read_text())
            expected['backends'].append([row.get('backend', row.get('engine')), row.get('variant')])
        if name.startswith('profiles/') and name.endswith('/profile.json'):
            expected['profiles'].append(json.loads((source / name).read_text())['name'])
    expected['profiles'].sort()
    with zipfile.ZipFile(args.wheel) as wheel:
        prefix = 'owner_mode/_assets/'
        bundled = [name[len(prefix):] for name in wheel.namelist() if name.startswith(prefix)]
        assert len(bundled) == len(set(bundled)), 'Duplicate packaged assets'
        assert set(bundled) == set(files), 'Wheel resource inventory differs'
    with tempfile.TemporaryDirectory(prefix='flashalign-wheel-check-') as directory:
        scratch = Path(directory)
        installed = scratch / 'installed'
        env = {key: value for key, value in os.environ.items() if key not in {'PYTHONPATH', 'PYTHONHOME'}}
        env.update(PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
        subprocess.run([args.python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-compile',
                        '--target', str(installed), str(args.wheel.resolve())],
                       cwd=scratch, env=env, check=True, capture_output=True, text=True, timeout=120)
        receipt = scratch / 'expected.json'
        receipt.write_text(json.dumps(expected))
        # Isolated mode ignores PYTHONPATH and the checkout. Only the installed
        # wheel is added; dependencies come from the selected interpreter.
        probe = subprocess.run([args.python, '-I', '-c', PROBE, str(installed), str(receipt), str(scratch)],
                               cwd=scratch, check=True, capture_output=True, text=True, timeout=60)
        result = json.loads(probe.stdout)
        resume = result.pop('resume')
        version = subprocess.run([*resume, '--version'], cwd=scratch, env=env,
                                 check=True, capture_output=True, text=True, timeout=20)
        bootstrap = ('import sys; sys.path.insert(0, ' + repr(str(installed)) + '); '
                     'from flashalign.__main__ import main; raise SystemExit(main())')
        # The empty PATH prevents probing any installed harness or node binary.
        detected = subprocess.run([args.python, '-I', '-c', bootstrap, 'setup', '--detect',
                                   '--home', str(scratch / 'empty-home')],
                                  cwd=scratch, env={**env, 'PATH': ''}, check=True,
                                  capture_output=True, text=True, timeout=30)
        detection = json.loads(detected.stdout)
        assert len(detection['backends']) == len(expected['backends'])
        assert not detection['profiles']['invalid']
        assert len(detection['profiles']['valid']) == len(expected['profiles'])
        result.update(resume_version=version.stdout.strip(), installed_setup_detect='passed',
                      source_checkout_on_import_path=False)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
