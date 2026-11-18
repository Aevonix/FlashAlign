# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A route's ordered profile set and each profile's access, failure and strength rules."""
from dataclasses import dataclass
import math
import re


MAX_PROFILES = 8
ACCESS = ("verified", "always")
FAIL = ("optional", "required")
# Names in a set are joined with commas in response headers and audit rows.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset({"name", "access", "fail", "scale", "digest"})


@dataclass(frozen=True)
class ProfileRule:
    """One profile on a route.

    access   "verified": applied to verified turns only. "always": applied to every routed request.
    fail     "optional": a profile the engine cannot apply is dropped and reported.
             "required": the request fails when the engine does not confirm the profile.
    scale    Strength in (0, 1], or None for the engine default.
    digest   Expected profile content digest, or None when the route does not pin one.
    """
    name: str
    access: str = "verified"
    fail: str = "optional"
    scale: float | None = None
    digest: str | None = None


def _scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1.0:
        raise ValueError("invalid profile scale")
    return float(value)


def route_profiles(route):
    """Return a route's ordered rules. An invalid declaration raises ValueError.

    ``profiles`` is a list of profile names or rule objects. A route that names one
    ``profile`` (with an optional ``owner_mode_scale``) is the same as a list of one
    verified, optional profile.
    """
    if "profiles" in route:
        if "profile" in route or "owner_mode_scale" in route:
            raise ValueError("a route declares either profiles or one profile")
        rows = route["profiles"]
        if not isinstance(rows, list) or len(rows) > MAX_PROFILES:
            raise ValueError("invalid profile list")
        rules = []
        for row in rows:
            if isinstance(row, str):
                row = {"name": row}
            if not isinstance(row, dict) or set(row) - _FIELDS:
                raise ValueError("invalid profile rule")
            name, access, fail = row.get("name"), row.get("access", "verified"), row.get("fail", "optional")
            digest = row.get("digest")
            if not isinstance(name, str) or _NAME.fullmatch(name) is None or name == "off":
                raise ValueError("invalid profile name")
            if access not in ACCESS or fail not in FAIL:
                raise ValueError("invalid profile access or failure rule")
            if digest is not None and (not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None):
                raise ValueError("invalid profile digest")
            rules.append(ProfileRule(name, access, fail, _scale(row["scale"]) if "scale" in row else None, digest))
        if len({rule.name for rule in rules}) != len(rules):
            raise ValueError("duplicate profile")
        return tuple(rules)
    scale = _scale(route["owner_mode_scale"]) if "owner_mode_scale" in route else None
    profile = route.get("profile")
    if not isinstance(profile, str) or not profile or profile == "off":
        return ()
    return (ProfileRule(profile, scale=scale),)


def label(rules):
    """The set as the engine reports it: names in order, joined with commas, or ``off``."""
    return ",".join(rule.name for rule in rules) or "off"


def verified_label(rules):
    """What a verified turn's ticket grants on this route."""
    return label(tuple(rule for rule in rules if rule.access == "verified"))


def selected(rules, sent):
    """The declared rules named by a label this module produced."""
    if sent == "off" or not rules:
        return ()
    if sent == label(rules):
        return tuple(rules)
    names = sent.split(",")
    return tuple(rule for rule in rules if rule.name in names)


def applied(rules, effective):
    """The rules an engine's effective-profile header confirms."""
    if not isinstance(effective, str) or effective in ("off", "none", ""):
        return ()
    if effective == label(rules):
        return tuple(rules)
    names = effective.split(",")
    return tuple(rule for rule in rules if rule.name in names)


def digest_check(rules, effective, reported):
    """Compare an engine's reported profile digests with the digests a route pins.

    ``effective`` and ``reported`` are the engine's effective-profile and profile-digest
    header values. Returns (drifted, unchecked): names whose reported digest differs from
    the pin, and pinned names the engine applied without reporting a comparable digest.
    """
    confirmed = applied(rules, effective)
    digests = reported.split(",") if isinstance(reported, str) and reported else []
    if len(digests) != len(confirmed):
        return (), tuple(rule.name for rule in confirmed if rule.digest)
    drifted = tuple(rule.name for rule, digest in zip(confirmed, digests) if rule.digest and digest != rule.digest)
    return drifted, ()


def wire(rules):
    """Owner fields of the final body. One profile keeps the scalar form every engine accepts."""
    if not rules:
        return {}
    if len(rules) == 1:
        fields = {"owner_mode": rules[0].name}
        if rules[0].scale is not None:
            fields["owner_mode_scale"] = rules[0].scale
        return fields
    fields = {"owner_mode": [rule.name for rule in rules]}
    if any(rule.scale is not None for rule in rules):
        fields["owner_mode_scale"] = [1.0 if rule.scale is None else rule.scale for rule in rules]
    return fields


def scales_sent(rules):
    """Audit value: the one strength sent, strengths joined in set order, or None when none was sent."""
    fields = wire(rules)
    value = fields.get("owner_mode_scale")
    if isinstance(value, list):
        return ",".join(format(item, "g") for item in value)
    return value
