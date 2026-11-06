# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve real OpenSSH options without opening a network connection."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from support import load_cli

load_cli('setup')
from owner_mode.native_cluster import owned_forward_options, ssh_options


@unittest.skipUnless(shutil.which('ssh'), 'OpenSSH is required for option resolution')
class OwnedForwardTests(unittest.TestCase):
    def test_forward_overrides_multiplexing_and_background_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/'ssh-config'
            config.write_text('Host fixture\n'
                              ' HostName 127.0.0.1\n'
                              ' ControlMaster auto\n'
                              ' ControlPath '+folder+'/master\n'
                              ' ControlPersist 900\n'
                              ' ForkAfterAuthentication yes\n')
            config.chmod(0o600)
            def resolve(options):
                run = subprocess.run(['ssh', '-G', '-N', *options, 'fixture'],
                                     capture_output=True, text=True, check=True, timeout=10)
                return dict(line.split(None, 1) for line in run.stdout.splitlines())
            baseline = resolve(ssh_options(config))
            self.assertEqual(baseline['controlmaster'], 'auto')
            self.assertEqual(baseline['controlpersist'], '900')
            self.assertEqual(baseline['forkafterauthentication'], 'yes')
            forward = resolve(owned_forward_options(config))
            self.assertEqual(forward['controlmaster'], 'false')
            self.assertEqual(forward.get('controlpath', 'none'), 'none')
            self.assertEqual(forward['controlpersist'], 'no')
            self.assertEqual(forward['forkafterauthentication'], 'no')
            self.assertEqual(forward['exitonforwardfailure'], 'yes')
            self.assertEqual(forward['forwardagent'], 'no')


if __name__ == '__main__':
    unittest.main()
