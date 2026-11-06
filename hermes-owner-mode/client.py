# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Direct engine client for trusted applications, independent of a chat harness."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from urllib.parse import urlsplit

import httpx

from .auth import read_api_key, read_key, sign


class ProfileNotActive(RuntimeError):
    pass


def response_metadata(response, profile):
    """Return only bounded protocol metadata, never authentication headers."""
    value = {'schema': 'flashalign.response.v1', 'effective_profile': profile,
             'status_code': response.status_code}
    for header, field in (('X-FlashAlign-Profile-Digest', 'profile_digest'),
                          ('X-FlashAlign-Model-Digest', 'model_digest')):
        item = response.headers.get(header)
        if item is not None:
            if re.fullmatch(r'[0-9a-f]{64}', item) is None:
                raise ValueError('Invalid engine identity metadata')
            value[field] = item
    generation = response.headers.get('X-FlashAlign-Generation')
    if generation is not None:
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}', generation) is None:
            raise ValueError('Invalid engine generation metadata')
        value['generation'] = generation
    return MappingProxyType(value)


@dataclass(frozen=True)
class ChatResponse:
    data: dict
    metadata: object


class StreamResponse:
    """One SSE response with explicit transport completion and safe metadata."""
    def __init__(self, response, metadata, *, max_event_bytes=None):
        self.metadata = metadata
        self.complete = False
        self.closed = False
        self._response = response
        self._started = False
        if max_event_bytes is not None and (type(max_event_bytes) is not int or not 0 < max_event_bytes <= 16 * 1024 * 1024):
            raise ValueError('SSE event byte limit must be a positive bounded integer')
        self._max_event_bytes = max_event_bytes

    def __iter__(self):
        if self._started:
            raise RuntimeError('A response stream can only be consumed once')
        self._started = True
        data = []
        size = 0
        for line in self._response.iter_lines():
            if not line:
                if not data:
                    continue
                payload = '\n'.join(data)
                data, size = [], 0
                if payload == '[DONE]':
                    self.complete = True
                    return
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError('SSE data must contain a JSON object')
                yield value
            elif line.startswith('data:'):
                item = line[5:]
                if item.startswith(' '):
                    item = item[1:]
                size += len(item.encode('utf-8')) if self._max_event_bytes is not None else len(item)
                if size > (self._max_event_bytes or 16 * 1024 * 1024):
                    raise ValueError('SSE event exceeds the response limit')
                data.append(item)
        # SSE dispatch requires the terminating blank line. EOF does not
        # certify either a partial event or a completed generation.


class Client:
    """A signing capability. Keep it in trusted application code, away from tools.

    The host decides whether a caller is authorized before choosing a profile.
    No identity claim in model input is interpreted by this client.
    """
    def __init__(self, base_url, model, *, key_file=None, api_key_file=None, timeout=300, transport=None,
                 mode='verified', profile=None, gateway=False, keepalive_expiry=1.0, max_sse_event_bytes=None):
        url = urlsplit(base_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('Use an HTTP(S) model URL without credentials, query or fragment')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('A model ID is required')
        if mode not in ('verified', 'always_on'):
            raise ValueError('Mode must be verified or always_on')
        if type(gateway) is not bool:
            raise ValueError('gateway must be true or false')
        if type(keepalive_expiry) not in (int, float) or not math.isfinite(keepalive_expiry) or keepalive_expiry < 0:
            raise ValueError('keepalive_expiry must be a finite nonnegative number')
        if mode == 'always_on' and (not isinstance(profile, str) or not profile or profile == 'off' or len(profile)>256):
            raise ValueError('Always On requires a configured profile')
        self.base_url, self.model = base_url.rstrip('/'), model
        self.mode, self.profile = mode, profile
        self.gateway = gateway
        self.key_file, self.api_key_file = key_file, api_key_file
        if max_sse_event_bytes is not None and (type(max_sse_event_bytes) is not int or not 0 < max_sse_event_bytes <= 16 * 1024 * 1024):
            raise ValueError('SSE event byte limit must be a positive bounded integer')
        self.max_sse_event_bytes = max_sse_event_bytes
        self.http = httpx.Client(timeout=timeout, transport=transport,
                                 limits=httpx.Limits(max_connections=100, max_keepalive_connections=20,
                                                     keepalive_expiry=keepalive_expiry))

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
        return self.chat_response(messages, profile=profile, scale=scale, **options).data

    def chat_response(self, messages, *, profile=None, scale=1., **options):
        profile = self._profile(profile)
        raw,headers=self.prepare({**options,'messages':messages,'stream':False},profile=profile,scale=scale)
        response=self.http.post(self.base_url+'/chat/completions',content=raw,headers=headers)
        self._check(response,profile)
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError('The engine response must be a JSON object')
        return ChatResponse(data, response_metadata(response, profile))

    @contextmanager
    def stream_chat(self, messages, *, profile=None, scale=1., **options):
        with self.stream_chat_response(messages, profile=profile, scale=scale, **options) as response:
            yield response

    @contextmanager
    def stream_chat_response(self, messages, *, profile=None, scale=1., **options):
        profile = self._profile(profile)
        raw,headers=self.prepare({**options,'messages':messages,'stream':True},profile=profile,scale=scale)
        with self.http.stream('POST',self.base_url+'/chat/completions',content=raw,headers=headers) as response:
            self._check(response,profile)
            stream = StreamResponse(response, response_metadata(response, profile), max_event_bytes=self.max_sse_event_bytes)
            try:
                yield stream
            finally:
                stream.closed = True

    def close(self):
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
