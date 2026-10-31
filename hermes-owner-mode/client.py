# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Direct engine client for trusted applications, independent of a chat harness."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .auth import read_api_key, read_key, sign


class ProfileNotActive(RuntimeError):
    pass


class Client:
    """A signing capability. Keep it in trusted application code, away from tools.

    The host decides whether a caller is authorized before choosing a profile.
    No identity claim in model input is interpreted by this client.
    """
    def __init__(self, base_url, model, *, key_file=None, api_key_file=None, timeout=300, transport=None,
                 mode='verified', profile=None, gateway=False):
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Use an HTTP(S) model URL without credentials, query or fragment')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('A model ID is required')
        if mode not in ('verified', 'always_on'):
            raise ValueError('Mode must be verified or always_on')
        if type(gateway) is not bool:
            raise ValueError('gateway must be true or false')
        if mode == 'always_on' and (not isinstance(profile, str) or not profile or profile == 'off' or len(profile)>256):
            raise ValueError('Always On requires a configured profile')
        self.base_url, self.model = base_url.rstrip('/'), model
        self.mode, self.profile = mode, profile
        self.gateway = gateway
        self.key_file, self.api_key_file = key_file, api_key_file
        self.http = httpx.Client(timeout=timeout, transport=transport)

    @classmethod
    def from_config(cls, path, **kwargs):
        from .profiles import read_json
        config = read_json(path)
        return cls(config['base_url'], config['model'], key_file=config.get('key_file'),
                   api_key_file=config.get('api_key_file'), mode=config.get('mode', 'verified'),
                   profile=config.get('profile'), gateway=config.get('gateway', False), **kwargs)

    def _profile(self, profile):
        if self.mode == 'always_on':
            if profile is not None and profile != self.profile:
                raise ValueError('Always On uses the configured profile; change the deployment mode to use another profile or off')
            return self.profile
        return 'off' if profile is None else profile

    def prepare(self, payload, *, profile=None, scale=1.):
        if not isinstance(payload, dict):
            raise ValueError('The request must be a JSON object')
        profile = self._profile(profile)
        if not isinstance(profile, str) or not profile or len(profile)>256:
            raise ValueError('A profile name or off is required')
        if type(scale) not in (int,float) or not math.isfinite(scale) or not 0<scale<=1:
            raise ValueError('Scale must be in (0, 1]')
        body = {**payload, 'model':self.model, 'owner_mode':profile}
        body.pop('owner_mode_scale', None)
        if profile!='off':
            body['owner_mode_scale']=scale
        raw=json.dumps(body,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        headers={'Content-Type':'application/json'}
        if self.api_key_file:
            headers['Authorization']='Bearer '+read_api_key(self.api_key_file)
        if profile!='off' and not self.gateway:
            if not self.key_file:
                raise ValueError('An owner request requires the deployment signing-key file')
            headers['X-Owner-Mode-Auth']=sign(read_key(self.key_file),raw)
        return raw,headers

    @staticmethod
    def _check(response, profile):
        response.raise_for_status()
        if response.headers.get('X-Owner-Mode-Effective')!=profile:
            raise ProfileNotActive('The engine did not confirm the requested profile')

    def chat(self, messages, *, profile=None, scale=1., **options):
        profile = self._profile(profile)
        raw,headers=self.prepare({**options,'messages':messages,'stream':False},profile=profile,scale=scale)
        response=self.http.post(self.base_url+'/chat/completions',content=raw,headers=headers)
        self._check(response,profile)
        return response.json()

    @contextmanager
    def stream_chat(self, messages, *, profile=None, scale=1., **options):
        profile = self._profile(profile)
        raw,headers=self.prepare({**options,'messages':messages,'stream':True},profile=profile,scale=scale)
        with self.http.stream('POST',self.base_url+'/chat/completions',content=raw,headers=headers) as response:
            self._check(response,profile)
            def events():
                for line in response.iter_lines():
                    if line=='data: [DONE]':
                        break
                    if line.startswith('data: '):
                        yield json.loads(line[6:])
            yield events()

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
