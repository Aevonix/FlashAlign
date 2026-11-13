# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""CLI diagnostics identify local input failures without echoing input contents."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from test_setup import wizard
from flashalign import benchmark, benchmark_cli, profile_evaluation


class BenchmarkDiagnosticsTests(unittest.TestCase):
    def invoke(self, main, argv):
        output = io.StringIO()
        with redirect_stderr(output):
            self.assertEqual(main(argv), 1)
        return output.getvalue()

    def test_each_cli_names_the_malformed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root/'private-config.json'
            config.write_text('{"synthetic-private-content":')
            common = ['--config', str(config), '--manifest', 'manifest.json',
                      '--settings', 'settings.json', '--output', 'output']
            for main, arguments in (
                    (benchmark.main, ['record', *common, '--cases', 'cases.jsonl']),
                    (benchmark_cli.main, ['run', *common, '--prepared', 'prepared']),
                    (profile_evaluation.main, ['evaluate', 'profile', *common, '--prepared', 'prepared'])):
                with self.subTest(command=arguments[0]):
                    message = self.invoke(main, arguments)
                    self.assertIn('--config', message)
                    self.assertIn('Invalid JSON at line 1, column', message)
                    self.assertNotIn(str(config), message)
                    self.assertNotIn('synthetic-private-content', message)

    def test_recorder_names_manifest_validation_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in [('config', {}), ('manifest', {}), ('settings', {}), ('cases', {})]:
                (root/name).write_text(json.dumps(value))
            args = ['record', '--output', str(root/'output')]
            for name in ('config', 'manifest', 'settings', 'cases'):
                args += ['--'+name, str(root/name)]
            message = self.invoke(benchmark.main, args)
            self.assertIn('Manifest must contain model, engine, hardware, speculation and profile', message)

    def test_evaluation_names_corpus_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source.jsonl'
            source.write_text('{}\n')
            message = self.invoke(benchmark_cli.main, ['prepare', '--source', str(source), '--output', str(root/'out')])
            self.assertIn('Evaluation corpus digest mismatch', message)

    def test_profile_evaluation_names_provenance_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = ['evaluate', 'profile', '--prepared', 'prepared', '--output', 'output']
            for name in ('config', 'manifest', 'settings'):
                (root/name).write_text('{}')
                args += ['--'+name, str(root/name)]
            with patch('owner_mode.profiles.validate_profile', return_value={'artifact_valid': True}):
                message = self.invoke(profile_evaluation.main, args)
            self.assertIn('Evaluation needs pinned model provenance', message)

    def test_module_command_reports_its_own_validation_error(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary)
            (inputs/'owner_mode').symlink_to(root/'hermes-owner-mode', target_is_directory=True)
            args = [sys.executable, '-m', 'flashalign.benchmark', 'record', '--output', str(inputs/'out')]
            for name in ('config', 'manifest', 'settings', 'cases'):
                (inputs/name).write_text('{}')
                args += ['--' + name, str(inputs/name)]
            result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=30,
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': str(inputs)})
            self.assertEqual(result.returncode, 1)
            self.assertIn('Manifest must contain model, engine, hardware, speculation and profile', result.stderr)

    def test_external_exception_text_stays_private(self):
        for error in (ValueError('synthetic-private-content'), RuntimeError('synthetic-private-content')):
            with patch.object(benchmark, 'read_json', side_effect=error):
                message = self.invoke(benchmark.main, ['record', '--config', 'config', '--manifest', 'manifest',
                    '--settings', 'settings', '--cases', 'cases', '--output', 'output'])
            self.assertNotIn('synthetic-private-content', message)
