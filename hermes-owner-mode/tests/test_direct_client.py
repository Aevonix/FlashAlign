# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Direct calls sign exact bytes and cannot silently lose the requested profile."""
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from test_setup import wizard
from owner_mode.client import Client, ProfileNotActive


class DirectClient(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.key=Path(temporary.name)/'key';self.key.write_bytes(b'x'*32);self.key.chmod(0o600)

    def test_exact_body_is_signed_and_request_options_cannot_override_mode(self):
        def reply(request):
            parts=request.headers['X-Owner-Mode-Auth'].split(':')
            expected=hmac.new(b'x'*32,(parts[1]+'\n'+parts[2]+'\n'+hashlib.sha256(request.content).hexdigest()).encode(),hashlib.sha256).hexdigest()
            self.assertEqual(parts[3],expected)
            body=json.loads(request.content)
            self.assertEqual(body['owner_mode'],'test-profile')
            self.assertEqual(body['owner_mode_scale'],.5)
            self.assertEqual(body['model'],'test-model')
            return httpx.Response(200,headers={'X-Owner-Mode-Effective':'test-profile'},json={'choices':[]})
        with Client('http://localhost/v1','test-model',key_file=self.key,transport=httpx.MockTransport(reply)) as client:
            client.chat([{'role':'user','content':'Unicode: ✓'}],profile='test-profile',scale=.5,
                        owner_mode='off',owner_mode_scale=9,model='other')

    def test_standard_mode_needs_no_key_and_removes_injected_owner_scale(self):
        def reply(request):
            self.assertNotIn('X-Owner-Mode-Auth',request.headers)
            self.assertEqual(json.loads(request.content)['owner_mode'],'off')
            self.assertNotIn('owner_mode_scale',json.loads(request.content))
            return httpx.Response(200,headers={'X-Owner-Mode-Effective':'off'},json={'choices':[]})
        with Client('http://localhost/v1','test-model',transport=httpx.MockTransport(reply)) as client:
            client.chat([],owner_mode='profile',owner_mode_scale=1)

    def test_effective_profile_mismatch_raises_before_stream_events_are_exposed(self):
        transport=httpx.MockTransport(lambda request:httpx.Response(200,headers={'X-Owner-Mode-Effective':'off'},text='data: {}\n\n'))
        with Client('http://localhost/v1','test-model',key_file=self.key,transport=transport) as client:
            with self.assertRaises(ProfileNotActive):
                with client.stream_chat([],profile='test-profile') as events:
                    list(events)

    def test_streaming_uses_confirmed_mode(self):
        transport=httpx.MockTransport(lambda request:httpx.Response(200,headers={'X-Owner-Mode-Effective':'off'},text='data: {"choices":[]}\n\ndata: [DONE]\n\n'))
        with Client('http://localhost/v1','test-model',transport=transport) as client:
            with client.stream_chat([]) as events:
                self.assertEqual(list(events),[{'choices':[]}])

    def test_malformed_scale_or_missing_key_fails_before_network(self):
        with Client('http://localhost/v1','test-model') as client:
            for value in (0,2,True,float('nan')):
                with self.assertRaises(ValueError):client.prepare({},profile='test-profile',scale=value)
            with self.assertRaises(ValueError):client.prepare({},profile='test-profile')


if __name__=='__main__':unittest.main()
