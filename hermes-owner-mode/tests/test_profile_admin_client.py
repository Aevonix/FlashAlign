# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import json
from pathlib import Path
import socket
import tempfile
import unittest

import httpx

from test_setup import wizard
from test_setup import make_profile
from flashalign.profile_admin import AdminClient, LifecycleRequestError


class AdminClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.path = self.root/'admin.sock'
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.bind(str(self.path))
        self.path.chmod(0o600)
        self.addCleanup(self.socket.close)

    def test_mutation_has_no_generation_auth_and_uses_expected_revision(self):
        def respond(request):
            self.assertEqual(request.url.path, '/operations')
            self.assertNotIn('X-Owner-Mode-Auth', request.headers)
            self.assertNotIn('Authorization', request.headers)
            self.assertEqual(json.loads(request.content), {'action': 'activate', 'alias': 'profile',
                                                         'generation': 'g-new', 'expected_revision': 2})
            return httpx.Response(200, json={'ok': True, 'result': {'revision': 3}})
        with AdminClient(self.path, transport=httpx.MockTransport(respond)) as client:
            self.assertEqual(client.request('activate', alias='profile', generation='g-new',
                                            expected_revision=2), {'revision': 3})

    def test_unsafe_socket_and_symlink_are_rejected(self):
        self.path.chmod(0o666)
        with self.assertRaises(ValueError):
            AdminClient(self.path)
        self.path.chmod(0o600)
        alias = self.root/'link.sock'
        alias.symlink_to(self.path)
        with self.assertRaises(ValueError):
            AdminClient(alias)

    def test_error_does_not_reflect_arbitrary_server_message(self):
        response = httpx.MockTransport(lambda request: httpx.Response(409, json={
            'ok': False, 'error': {'code': 'RevisionConflict', 'message': 'PRIVATE-SECRET'}}))
        with AdminClient(self.path, transport=response) as client:
            with self.assertRaises(LifecycleRequestError) as error:
                client.request('recover')
        self.assertIn('RevisionConflict', str(error.exception))
        self.assertNotIn('PRIVATE-SECRET', str(error.exception))

    def test_container_input_is_validated_and_cleaned_after_consumption(self):
        source = make_profile(self.root / 'source')
        staging = self.root / 'incoming'
        staging.mkdir(mode=0o700)
        def respond(request):
            path = Path(json.loads(request.content)['path'])
            self.assertEqual(path.parent, staging)
            self.assertTrue((path / 'profile.safetensors').is_file())
            self.assertEqual((path / 'profile.safetensors').read_bytes(),
                             (source / 'profile.safetensors').read_bytes())
            return httpx.Response(200, json={'ok': True, 'result': {'generation': 'g-test'}})
        with AdminClient(self.path, transport=httpx.MockTransport(respond)) as client:
            self.assertEqual(client.stage(source, staging_directory=staging), {'generation': 'g-test'})
        self.assertEqual(list(staging.iterdir()), [])
        staging.chmod(0o755)
        with AdminClient(self.path, transport=httpx.MockTransport(respond)) as client:
            with self.assertRaises(ValueError):
                client.stage(source, staging_directory=staging)


if __name__ == '__main__':
    unittest.main()
