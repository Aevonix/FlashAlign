# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A reader following the guide installs package aliases before importing them."""
from pathlib import Path
import re
import shlex
import unittest


class BenchmarkCheckoutDocsTests(unittest.TestCase):
    def test_install_precedes_every_package_command(self):
        root = Path(__file__).resolve().parents[2]
        guide = (root/'docs/benchmarks.md').read_text()
        installed = False
        imports = 0
        for block in re.findall(r'```(?:sh|python)\n(.*?)```', guide, re.S):
            for line in block.replace('\\\n', ' ').splitlines():
                if line == 'python -m pip install .':
                    installed = True
                if (line.startswith('python -m flashalign.')
                        or line.startswith('python benchmarks/')
                        or line.startswith('from flashalign')):
                    imports += 1
                    self.assertTrue(installed, 'Install the checkout before: ' + line)
                if line.startswith('python benchmarks/'):
                    self.assertTrue((root/shlex.split(line)[1]).is_file())
        self.assertGreater(imports, 0)
