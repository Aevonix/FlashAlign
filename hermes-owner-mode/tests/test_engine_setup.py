# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Shared native bundle and isolated protocol-server fixtures."""
from pathlib import Path
import sys
from unittest import mock
from urllib.parse import urlsplit

from test_native_setup import make_bundle
from owner_mode.engine_setup import prepare_engine
from owner_mode import native_setup

SERVER = '\nfrom http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\nimport hashlib,hmac,json,os,sys\nfrom pathlib import Path\nkey=Path(os.environ[\'TF_OWNER_MODE_KEY_FILE\']).read_bytes()\nname=json.loads((Path(os.environ[\'TF_OWNER_PROFILES\'])/\'profile.json\').read_text())[\'name\']\nclass Handler(BaseHTTPRequestHandler):\n def log_message(self,*args):pass\n def do_GET(self):\n  data=json.dumps({\'data\':[{\'id\':\'bench\'}]}).encode();self.send_response(200);self.end_headers();self.wfile.write(data)\n def do_POST(self):\n  raw=self.rfile.read(int(self.headers[\'Content-Length\']));body=json.loads(raw);effective=\'off\'\n  auth=self.headers.get(\'X-Owner-Mode-Auth\',\'\').split(\':\')\n  if len(auth)==4:\n   version,ts,nonce,mac=auth\n   expected=hmac.new(key,(ts+\'\\n\'+nonce+\'\\n\'+hashlib.sha256(raw).hexdigest()).encode(),hashlib.sha256).hexdigest()\n   if hmac.compare_digest(mac,expected) and body.get(\'owner_mode\')==name:effective=name\n  data=b\'{"choices":[]}\'\n  self.send_response(200);self.send_header(\'X-Owner-Mode-Effective\',effective);self.end_headers();self.wfile.write(data)\nThreadingHTTPServer((\'127.0.0.1\',int(sys.argv[1])),Handler).serve_forever()\n'


def fixture_deployment(home, spec, profile, *, model_id='bench', key_file=None):
    """Inject a protocol test process after native preparation; not a product escape hatch."""
    command = spec['command']
    native = {k: v for k, v in spec.items() if k != 'command'}
    # This fixture starts a Python protocol server, not a GPU engine. Supply
    # synthetic device evidence so its timeout checks also run on CPU-only CI.
    devices = [str(device) for device in native.get('devices', ['0'])]
    with mock.patch.object(native_setup, 'cuda_inventory',
                           return_value=(devices, {device: 'sm_120' for device in devices})):
        result = prepare_engine(home, native, profile, model_id=model_id, key_file=key_file)
    values = {'port': str(urlsplit(result['base_url']).port)}
    result['commands'] = [{'argv': [part.format_map(values) for part in command],
                           'env': {**spec.get('environment', {}), 'TF_OWNER_MODE_KEY_FILE': result['key_file'],
                                   'TF_OWNER_PROFILES': result['profile_path']}}]
    return result
