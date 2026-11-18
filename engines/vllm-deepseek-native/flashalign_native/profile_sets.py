# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Authorize one profile or a set of profiles, and name the cache namespace of the result."""
from dataclasses import dataclass
import hashlib
import json
import math

from .protocol import FieldError, Selection, cache_key

MAX_SET = 8


@dataclass(frozen=True)
class Applied:
    """What one request selected. `names` keeps the request's order."""
    primary: Selection | None = None
    low_rank: tuple = ()
    names: tuple = ()


def parse_set(body):
    """Names and strengths of an `owner_mode` array, one strength per name."""
    names = body.get('owner_mode')
    if (not isinstance(names, list) or not 2 <= len(names) <= MAX_SET
            or any(not isinstance(name, str) or not name or name == 'off' for name in names)
            or len(set(names)) != len(names)):
        raise FieldError('owner_mode must be "off", a profile name or two to eight distinct profile names')
    scales = body.get('owner_mode_scale')
    if scales is None:
        scales = [1.] * len(names)
    if (not isinstance(scales, list) or len(scales) != len(names)
            or any(type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1
                   for scale in scales)):
        raise FieldError('owner_mode_scale must hold one number in (0, 1] per profile')
    return names, [float(scale) for scale in scales]


def set_cache_key(applied, profiles, model_identity, original=None):
    """Cache namespace of a request. One matrix or projection profile alone keeps its existing key."""
    if not applied.low_rank:
        return cache_key(applied.primary, profiles, model_identity, original)
    primary = applied.primary
    material = json.dumps([
        'flashalign.profile-set.v1', model_identity,
        None if primary is None else [primary.name, primary.digest, primary.scale.hex(), primary.generation],
        [[selection.name, selection.digest, selection.scale.hex()] for selection in applied.low_rank],
        list(applied.names), original], sort_keys=True, separators=(',', ':'))
    # A different prefix and shape: no caller-supplied salt can name this namespace.
    return 'flashalign-set:' + hashlib.sha256(material.encode()).hexdigest()


def authorize_set(gate, low_rank, raw, body, headers):
    """Authorize one profile or a set. Every named profile applies, or the request is rejected.

    `gate.profiles` holds the profile that replaces an operation; `low_rank` holds the
    profiles that add to one. A set takes at most one of the former.
    """
    mode = body.get('owner_mode', 'off')
    if isinstance(mode, list):
        names, scales = parse_set(body)
    elif isinstance(mode, str) and mode in low_rank:
        scale = body.get('owner_mode_scale', 1.)
        scale = 1. if scale is None else scale
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
            raise FieldError('owner_mode_scale must be a number in (0, 1]')
        names, scales = [mode], [float(scale)]
    else:
        selection = gate.authorize(raw, body, headers)
        return Applied() if selection is None else Applied(selection, (), (selection.name,))
    if not gate.profiles:
        raise FieldError('this server was started without owner-mode profiles: owner_mode must be "off"')
    # The signature covers the raw bytes, not the parsed fields. The gate's own check (header
    # grammar, time window, HMAC, replay) therefore runs unchanged; it only needs a loaded name.
    if gate.authorize(raw, {'owner_mode': next(iter(gate.profiles))}, headers) is None:
        return Applied()
    primary, added = None, []
    for name, scale in zip(names, scales):
        if name in gate.profiles and name in low_rank:
            # A profile update can bring in a name after start. Never guess which one was meant.
            raise FieldError('owner_mode names a profile this server loaded twice')
        if name in gate.profiles:
            if primary is not None:
                raise FieldError('owner_mode names two profiles that replace the same operation; '
                                 'a set holds at most one of them')
            primary = Selection(name, gate.profiles[name].digest, scale)
        elif name in low_rank:
            added.append(Selection(name, low_rank[name].digest, scale))
        else:
            raise FieldError('owner_mode names no profile this server loaded')
    return Applied(primary, tuple(added), tuple(names))
