# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Small client for a trusted native adapter talking to the loopback bridge."""
from contextlib import contextmanager
from urllib.parse import urlsplit

import httpx

from .auth import read_key


class HarnessClient:
    """Native identity comes from the host's transport, never from model input."""
    def __init__(self, bridge_url, control_key_file, *, timeout=300):
        url=urlsplit(bridge_url)
        if (url.scheme!='http' or url.hostname not in ('127.0.0.1','::1','localhost')
                or url.username or url.password or url.query or url.fragment or url.path not in ('','/')):
            raise ValueError('The control bridge must be a loopback HTTP origin')
        self.url=bridge_url.rstrip('/')
        self.authorization='Bearer '+read_key(control_key_file).hex()
        self.http=httpx.Client(timeout=timeout,follow_redirects=False)

    def control(self, operation, body):
        if operation not in {'claim','claim_session','bind','adopt','end','tool_begin','tool_end','tool'}:
            raise ValueError('Unknown bridge operation')
        response=self.http.post(self.url+'/_owner/'+operation,json=body,
                                headers={'Authorization':self.authorization})
        response.raise_for_status()
        return response.json()

    def claim_native(self, *, session, run, messages):
        """messages: native id/channel/account/contact records attested by host code."""
        return self.control('claim',{'session':session,'run':run,'messages':messages})

    def claim_verified_session(self, *, session, run, channel, account, contact):
        """Host-attested isolated session; protected native identity is required."""
        return self.control('claim_session',{'session':session,'run':run,'channel':channel,
                                             'account':account,'contact':contact})

    def bind_content(self, capability, content):
        return self.control('bind',{'capability':capability,'content':content})

    def end(self, capability):
        return self.control('end',{'capability':capability})

    def adopt(self, parent, *, session, run, tool):
        return self.control('adopt',{'parent':parent,'session':session,'run':run,'tool':tool})

    @contextmanager
    def tool_scope(self, capability, *, call, tool):
        self.control('tool_begin',{'capability':capability,'call':call,'tool':tool})
        try:
            yield
        finally:
            self.control('tool_end',{'capability':capability,'call':call})

    def chat_response(self, capability, messages, *, route='head', **options):
        if not route or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in route):
            raise ValueError('Invalid bridge route')
        response=self.http.post(self.url+'/'+route+'/v1/chat/completions',
                                json={**options,'messages':messages,'stream':False},
                                headers={'X-Owner-Mode-Run':capability} if capability else {})
        response.raise_for_status()
        return response

    def close(self):
        self.http.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
