# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Private gate integration with actual persistent replay state and restarts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PersistentGate(unittest.TestCase):
    def test_future_timestamp_replay_is_rejected_after_process_restart(self):
        script = r'''
import hashlib,hmac,json,sys
from types import SimpleNamespace
from flashalign_sglang.gate import Gate
from flashalign_sglang.flashalign_replay import ReplayStore
path,now=sys.argv[1],int(sys.argv[2])
key=b'x'*32
body={'owner_mode':'test'}
raw=json.dumps(body).encode()
stamp='1700000050';nonce='a'*32
mac=hmac.new(key,(stamp+'\n'+nonce+'\n'+hashlib.sha256(raw).hexdigest()).encode(),hashlib.sha256).hexdigest()
store=ReplayStore(path,key,clock=lambda:now)
gate=Gate({'test':SimpleNamespace(digest='artifact')},key,clock=lambda:now,replay_store=store)
print(json.dumps({'owner':gate.select(raw,body,['v1:'+stamp+':'+nonce+':'+mac]) is not None}))
'''
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
            path = str(Path(directory)/'nonces.sqlite3')
            first = json.loads(subprocess.check_output([sys.executable, '-c', script, path, '1700000000'], env=env))
            second = json.loads(subprocess.check_output([sys.executable, '-c', script, path, '1700000070'], env=env))
            self.assertTrue(first['owner'])
            self.assertFalse(second['owner'])

    def test_unprepared_source_is_rejected_before_launch(self):
        from installation import validate_prepared_source
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'has not been prepared'):
                validate_prepared_source(directory)
