# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Check build prerequisites without installing packages into the host OS."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.engine_setup import prepare_python_headers, private_json
from owner_mode.setup_runtime import SetupError

COMPILER = shutil.which('cc')
RUN = subprocess.run


class PythonHeaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='flashalign-python-headers-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.build = {'version': [3, 12], 'include': str(self.root/'missing'), 'cc': COMPILER, 'cflags': ''}
        self.environment = mock.patch.dict(os.environ, {'CPATH': '', 'CC': COMPILER or 'cc',
            'CFLAGS': '', 'CPPFLAGS': '', 'C_INCLUDE_PATH': ''})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def run_command(self, command, *args, **kwargs):
        if command[:2] == ['python', '-c']:
            return SimpleNamespace(stdout=json.dumps(self.build))
        return RUN(command, *args, **kwargs)

    def headers(self, path, *, config='pyconfig.h', version=(3, 12)):
        path.mkdir(parents=True)
        (path/'Python.h').write_text(f'#include "patchlevel.h"\n#include <{config}>\n')
        (path/'patchlevel.h').write_text(f'#define PY_MAJOR_VERSION {version[0]}\n#define PY_MINOR_VERSION {version[1]}\n')
        if config == 'pyconfig.h':
            (path/config).write_text('/* complete fixture */\n')

    def split_headers(self):
        location = self.root/'usr/include/python3.12'
        # A unique multiarch name prevents host development headers masking a
        # missing transitive header in this synthetic include tree.
        config = f'{self.root.name}-linux-gnu/python3.12/pyconfig.h'
        self.headers(location, config=config)
        return location, self.root/'usr/include'/config

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_existing_matching_headers_are_reused_without_download(self):
        location = self.root/'include'
        self.headers(location)
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command) as run:
            result = prepare_python_headers('python', self.root/'state', environment={'CPATH': str(location)})
        self.assertEqual(result, {'CPATH': str(location)})
        self.assertEqual(run.call_count, 2)
        self.assertIn('-fsyntax-only', run.call_args.args[0])
        self.assertFalse((self.root/'state').exists())

    def test_missing_headers_report_action_before_model_start(self):
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            with self.assertRaisesRegex(SetupError, 'development headers are missing'):
                prepare_python_headers('python', self.root/'state')
        self.assertFalse((self.root/'state').exists())

    def test_cached_header_modification_is_rejected(self):
        state = self.root/'state'
        unpacked = state/'python-headers/unpacked'
        self.headers(unpacked/'usr/include/python3.12')
        private_json(state/'python-headers/receipt.json', {'files': {'usr/include/python3.12/Python.h': 'incorrect'}})
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            with self.assertRaisesRegex(SetupError, 'headers changed'):
                prepare_python_headers('python', state)

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_explicit_cpath_rejects_missing_transitive_multiarch_header(self):
        location, _ = self.split_headers()
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command) as run:
            with self.assertRaisesRegex(SetupError, 'multiarch pyconfig.h'):
                prepare_python_headers('python', self.root/'state', environment={'CPATH': str(location)})
        self.assertEqual(run.call_count, 2)
        self.assertFalse((self.root/'state').exists())

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_split_multiarch_headers_reuse_full_cpath(self):
        location, config = self.split_headers()
        config.parent.mkdir(parents=True)
        config.write_text('/* multiarch configuration */\n')
        cpath = os.pathsep.join([str(location), str(self.root/'usr/include')])
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            result = prepare_python_headers('python', self.root/'state', environment={'CPATH': cpath})
        self.assertEqual(result, {'CPATH': cpath})
        self.assertFalse((self.root/'state').exists())

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_cached_headers_are_compiled_with_the_managed_include_root(self):
        state = self.root/'state'
        unpacked = state/'python-headers/unpacked'
        location = unpacked/'usr/include/python3.12'
        config = f'{self.root.name}-linux-gnu/python3.12/pyconfig.h'
        self.headers(location, config=config)
        receipt = state/'python-headers/receipt.json'
        private_json(receipt, {'files': {str(path.relative_to(unpacked)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in location.iterdir()}})
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            with self.assertRaisesRegex(SetupError, 'multiarch pyconfig.h'):
                prepare_python_headers('python', state)
        configuration = unpacked/'usr/include'/config
        configuration.parent.mkdir(parents=True)
        configuration.write_text('/* managed multiarch configuration */\n')
        private_json(receipt, {'files': {str(path.relative_to(unpacked)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in unpacked.rglob('*') if path.is_file()}})
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            result = prepare_python_headers('python', state)
        self.assertEqual(result, {'CPATH': os.pathsep.join([str(location), str(unpacked/'usr/include')])})

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_platform_include_is_used_for_python_configuration(self):
        location = self.root/'include'
        self.headers(location)
        platform = self.root/'platform'
        platform.mkdir()
        (platform/'platform_config.h').write_text('/* platform configuration */\n')
        (location/'Python.h').write_text('#include "patchlevel.h"\n#include <platform_config.h>\n')
        self.build.update(include=str(location), platinclude=str(platform))
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            self.assertEqual(prepare_python_headers('python', self.root/'state'), {})

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_compiler_and_flags_preserve_transitive_include_search(self):
        location, config = self.split_headers()
        config.parent.mkdir(parents=True)
        config.write_text('#if PYTHON_BUILD_FLAG != 1 || ENGINE_BUILD_FLAG != 2\n#error Missing compiler flags\n#endif\n')
        self.build.update(cc='not-the-selected-compiler', cflags='-DPYTHON_BUILD_FLAG=1')
        environment = {'CPATH': str(location), 'CC': shlex.quote(COMPILER)+' -Werror',
                       'CFLAGS': '-DENGINE_BUILD_FLAG=2',
                       'CPPFLAGS': '-I'+shlex.quote(str(self.root/'usr/include'))}
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            result = prepare_python_headers('python', self.root/'state', environment=environment)
        self.assertEqual(result, {'CPATH': str(location)})

    @unittest.skipUnless(COMPILER, 'C compiler required for syntax-only header check')
    def test_interpreter_include_precedes_cpath_and_checks_compiled_version(self):
        location = self.root/'configured'
        self.headers(location)
        self.headers(Path(self.build['include']), version=(3, 11))
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            with self.assertRaisesRegex(SetupError, 'headers cannot be compiled'):
                prepare_python_headers('python', self.root/'state', environment={'CPATH': str(location)})

    def test_missing_compiler_fails_closed_with_actionable_error(self):
        location = self.root/'include'
        self.headers(location)
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=self.run_command):
            with self.assertRaisesRegex(SetupError, 'install a C compiler or set engine.environment.CC'):
                prepare_python_headers('python', self.root/'state', environment={
                    'CPATH': str(location), 'CC': str(self.root/'missing-compiler')})
        self.assertFalse((self.root/'state').exists())

    def test_header_probe_timeout_is_a_setup_error(self):
        location = self.root/'include'
        self.headers(location)
        responses = [SimpleNamespace(stdout=json.dumps(self.build)), subprocess.TimeoutExpired('cc', 15)]
        with mock.patch('owner_mode.engine_setup.subprocess.run', side_effect=responses):
            with self.assertRaisesRegex(SetupError, 'headers cannot be compiled'):
                prepare_python_headers('python', self.root/'state', environment={'CPATH': str(location)})


if __name__ == '__main__':
    unittest.main()
