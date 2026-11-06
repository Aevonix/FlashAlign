# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Interpreter-local tools must work without a shell-activated environment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_setup import wizard
from owner_mode.backend_setup import select_backend_python
from owner_mode.engine_setup import activate_engine, stop_engine
from owner_mode.runtime_environment import python_environment
from owner_mode.setup_runtime import SetupError


@unittest.skipUnless(os.name == 'posix', 'The managed engine process tests require POSIX')
class RuntimeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='flashalign runtime ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root/'venv/bin'
        self.bin.mkdir(parents=True)
        self.python = self.bin/'python'
        self.python.symlink_to(sys.executable)
        self.explicit = self.root/'explicit-bin'
        self.explicit.mkdir()
        self.inherited = self.root/'inherited-bin'
        self.inherited.mkdir()

    def tool(self, directory, name, output):
        path = directory/name
        path.write_text('#!/bin/sh\nprintf "%s\\n" '+output+'\n')
        path.chmod(0o755)

    def run_tool(self, environment, name='runtime-tool'):
        return subprocess.check_output([str(self.python), '-c',
            'import subprocess; print(subprocess.check_output(['+repr(name)+'], text=True).strip())'],
            env={**os.environ, **environment}, stderr=subprocess.PIPE, text=True).strip()

    def test_symlinked_interpreter_finds_its_own_tool(self):
        self.tool(self.bin, 'runtime-tool', 'selected')
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_tool({'PATH': str(self.inherited)})
        environment = python_environment(self.python, inherited={'PATH': str(self.inherited)})
        self.assertEqual(self.run_tool(environment), 'selected')
        self.assertEqual(environment['PATH'].split(os.pathsep)[0], str(self.bin))
        self.assertNotEqual(self.python.resolve().parent, self.bin)

    def test_explicit_path_and_other_overrides_take_precedence(self):
        self.tool(self.bin, 'runtime-tool', 'selected')
        self.tool(self.explicit, 'runtime-tool', 'explicit')
        self.tool(self.explicit, 'fallback-tool', 'explicit')
        self.tool(self.inherited, 'fallback-tool', 'inherited')
        overrides = {'PATH': str(self.explicit)+os.pathsep, 'SETTING': 'kept'}
        inherited = {'PATH': str(self.inherited), 'SETTING': 'inherited'}
        environment = python_environment(self.python, overrides, inherited=inherited)
        self.assertEqual(self.run_tool(environment), 'selected')
        self.assertEqual(self.run_tool(environment, 'fallback-tool'), 'explicit')
        self.assertEqual(environment['PATH'], str(self.bin)+os.pathsep+overrides['PATH'])
        self.assertEqual(environment['SETTING'], 'kept')
        self.assertEqual(overrides, {'PATH': str(self.explicit)+os.pathsep, 'SETTING': 'kept'})
        self.assertEqual(inherited, {'PATH': str(self.inherited), 'SETTING': 'inherited'})
        self.assertEqual(python_environment(self.python, environment, inherited=inherited), environment)

    def test_empty_path_is_preserved_and_bare_names_never_invent_a_directory(self):
        self.assertEqual(python_environment(self.python, {'PATH': ''})['PATH'], str(self.bin)+os.pathsep)
        path = str(self.bin)+os.pathsep+str(self.explicit)
        self.assertEqual(python_environment('python', inherited={'PATH': path})['PATH'], path)
        self.assertEqual(python_environment('missing-python', inherited={'PATH': str(self.explicit)}), {})

    def test_selection_freezes_bare_name_from_explicit_path(self):
        with mock.patch.dict(os.environ, {'PATH': str(self.inherited)}):
            selected = select_backend_python(
                {'python': 'python', 'environment': {'PATH': str(self.bin)}},
                {'python_min_version': [3, 10]})
        self.assertEqual(selected['executable'], str(self.python))
        self.tool(self.bin, 'runtime-tool', 'selected')
        self.assertEqual(self.run_tool(python_environment(selected['executable'],
                                                         {'PATH': str(self.inherited)})), 'selected')

    def test_selection_freezes_relative_path_before_activation_changes_directory(self):
        relative = os.path.relpath(self.python)
        selected = select_backend_python({'python': relative}, {})
        self.assertEqual(selected['executable'], str(self.python))
        self.assertNotEqual(Path(selected['executable']).resolve().parent, self.bin)

    def test_invalid_environment_fails_before_interpreter_inspection(self):
        for environment in (None, [], 'PATH', {'PATH': None}, {'PATH': 7}, {2: 'value'}):
            with self.subTest(environment=environment), mock.patch('owner_mode.backend_setup._python_version') as version:
                with self.assertRaises(SetupError):
                    select_backend_python({'environment': environment}, {'python_min_version': [3, 12]})
                version.assert_not_called()
        for environment in ([], {'PATH': None}, {'OTHER': False}):
            with self.subTest(environment=environment), self.assertRaises(SetupError):
                python_environment(self.python, environment)

    def test_automatic_search_uses_configured_path(self):
        candidate = self.bin/'python3.12'
        candidate.symlink_to(sys.executable)
        versions = {'/missing/system-python': (3, 9, 1), str(candidate): (3, 12, 1)}
        with (mock.patch('owner_mode.backend_setup.sys._base_executable', '/missing/system-python'),
              mock.patch('owner_mode.backend_setup.sys.executable', '/missing/system-python'),
              mock.patch('owner_mode.backend_setup._python_version', side_effect=versions.get)):
            selected = select_backend_python({'environment': {'PATH': str(self.bin)}},
                                             {'python_min_version': [3, 12]})
        self.assertEqual(selected['executable'], str(candidate))

    def test_activation_repairs_a_stored_python_command_only(self):
        self.tool(self.bin, 'runtime-tool', 'selected')
        marker = self.root/'ready.json'
        program = ('import json,os,subprocess,time; from pathlib import Path; '
                   "value=subprocess.check_output(['runtime-tool'],text=True).strip(); "
                   f'Path({str(marker)!r}).write_text(json.dumps({{"tool":value,"path":os.environ["PATH"]}})); '
                   'time.sleep(60)')
        state = self.root/'state'; state.mkdir()
        deployment = {'python': str(self.python), 'state': str(state), 'source': str(self.root),
                      'base_url': 'http://127.0.0.1:0/v1', 'model': 'fixture', 'profile': 'fixture',
                      'key_file': str(self.root/'unused'), 'install_dependencies': False,
                      'commands': [{'argv': [str(self.python), '-c', program],
                                    'env': {'PATH': str(self.explicit)}}]}
        self.addCleanup(lambda: stop_engine(deployment))
        def models(*_args, **_kwargs):
            if not marker.exists():
                raise SetupError('Fixture is starting')
            return [{'id': 'fixture'}]
        with (mock.patch('owner_mode.engine_setup.model_rows', side_effect=models),
              mock.patch('owner_mode.engine_setup.check_engine', return_value={'status': 'active'}),
              mock.patch('owner_mode.lifecycle_setup.check_lifecycle', return_value={})):
            activate_engine(deployment, timeout=8)
        value = json.loads(marker.read_text())
        self.assertEqual(value['tool'], 'selected')
        self.assertEqual(value['path'], str(self.bin)+os.pathsep+str(self.explicit))
        self.assertEqual(deployment['commands'][0]['env']['PATH'], str(self.explicit))
        self.assertEqual(stop_engine(deployment)['status'], 'stopped')

    def test_native_command_does_not_receive_the_controller_python_path(self):
        marker = self.root/'native.json'
        program = ('import json,os,time; from pathlib import Path; '
                   f'Path({str(marker)!r}).write_text(json.dumps({{"path":os.environ["PATH"]}})); '
                   'time.sleep(60)')
        state = self.root/'state'; state.mkdir()
        deployment = {'python': str(self.python), 'state': str(state), 'source': str(self.root),
                      'base_url': 'http://127.0.0.1:0/v1', 'model': 'fixture', 'profile': 'fixture',
                      'key_file': str(self.root/'unused'), 'install_dependencies': False,
                      'commands': [{'argv': [sys.executable, '-c', program],
                                    'env': {'PATH': str(self.explicit)}}]}
        self.addCleanup(lambda: stop_engine(deployment))
        def models(*_args, **_kwargs):
            if not marker.exists():
                raise SetupError('Fixture is starting')
            return [{'id': 'fixture'}]
        with (mock.patch('owner_mode.engine_setup.model_rows', side_effect=models),
              mock.patch('owner_mode.engine_setup.check_engine', return_value={'status': 'active'}),
              mock.patch('owner_mode.lifecycle_setup.check_lifecycle', return_value={})):
            activate_engine(deployment, timeout=8)
        self.assertEqual(json.loads(marker.read_text())['path'], str(self.explicit))
        self.assertEqual(stop_engine(deployment)['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
