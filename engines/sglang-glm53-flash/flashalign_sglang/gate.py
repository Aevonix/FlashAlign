# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Signed request selection and internal cache namespaces; no model code."""
from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from .auth_contract import authenticate_header, read_signing_key


AUTH = 'x-owner-mode-auth'
EFFECTIVE = b'x-owner-mode-effective'
PREFIX = 'flashalign.v1.'


class FieldError(ValueError):
    pass


def read_key(path):
    return read_signing_key(path)


@dataclass(frozen=True)
class Selection:
    name: str
    digest: str
    scale: float


class Gate:
    def __init__(self, profiles, key, clock=time.time, model_scope='', *, replay_store=None):
        self.profiles, self.key, self.clock = profiles, key, clock
        self.model_scope=model_scope
        self.replay_store = replay_store
        self.used = {}
        self.lock = threading.Lock()

    def authenticate(self, raw, headers):
        now = self.clock()
        auth = authenticate_header(self.key, raw, headers, now=now)
        if auth is None:
            return False
        nonce, timestamp = auth.nonce, auth.timestamp
        if self.replay_store is not None:
            return self.replay_store.consume(nonce, int(timestamp)+61, now=now)
        with self.lock:
            self.used = {k:v for k,v in self.used.items() if v >= now}
            if nonce in self.used or len(self.used) >= 65536:
                return False
            self.used[nonce] = int(timestamp)+61
        return True

    def select(self, raw, body, headers):
        mode,scale = body.get('owner_mode','off'),body.get('owner_mode_scale',1.)
        mode = 'off' if mode is None else mode
        scale = 1. if scale is None else scale
        if not isinstance(mode,str):
            raise FieldError('owner_mode must be "off" or an owner-mode profile name')
        if type(scale) not in (int,float) or not math.isfinite(scale) or not 0 < scale <= 2:
            raise FieldError('owner_mode_scale must be a number in (0, 2]')
        if mode == 'off':
            return None
        if not self.profiles:
            raise FieldError('this server was started without owner-mode profiles: owner_mode must be "off"')
        if not self.authenticate(raw,headers):
            return None
        if mode not in self.profiles:
            raise FieldError('owner_mode names no profile this server loaded')
        return Selection(mode,self.profiles[mode].digest,float(scale))

    def cache_salt(self, selection, salt):
        # Always wrap the caller's salt. An unsigned request cannot select an
        # owner namespace by copying a salt from another transport or log.
        raw = json.dumps([self.model_scope,salt],sort_keys=True,separators=(',',':')).encode()
        identity = hashlib.sha256(raw).hexdigest()
        if selection is None:
            return PREFIX+'off.'+identity
        data = json.dumps([selection.name,selection.digest,selection.scale.hex(),identity,self.model_scope],
                          separators=(',',':')).encode()
        encoded = base64.urlsafe_b64encode(data).decode().rstrip('=')
        mac = hmac.new(self.key,b'cache\n'+data,hashlib.sha256).hexdigest()
        return PREFIX+encoded+'.'+mac

    def from_salt(self, salt):
        if not isinstance(salt,str) or not salt.startswith(PREFIX) or salt.startswith(PREFIX+'off.'):
            return None
        try:
            value,mac = salt[len(PREFIX):].rsplit('.',1)
            data = base64.urlsafe_b64decode(value+'='*(-len(value)%4))
            expected = hmac.new(self.key,b'cache\n'+data,hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected,mac):
                raise ValueError()
            name,digest,encoded_scale,_,model_scope = json.loads(data)
            scale = float.fromhex(encoded_scale)
            if model_scope!=self.model_scope or name not in self.profiles or digest != self.profiles[name].digest or not 0 < scale <= 2:
                raise ValueError()
            return Selection(name,digest,scale)
        except (ValueError,TypeError,KeyError):
            raise ValueError('Invalid internal FlashAlign cache namespace') from None
