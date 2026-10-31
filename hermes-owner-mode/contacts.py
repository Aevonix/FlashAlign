# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Operator-managed, durable approvals for exact channel/account/contact pairs."""
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile

from .channels import DEFAULT_ENABLED, DEFAULT_REGISTRY, adapter_id, opaque_id


def pair(channel, account, contact, *, registry=DEFAULT_REGISTRY):
    channel = channel.strip().lower() if isinstance(channel, str) else ""
    account = account.strip() if isinstance(account, str) else ""
    if not account:
        raise ValueError("An exact account is required")
    contact = registry.canonical_contact(channel, contact)
    return channel, account, contact


def empty():
    return {"version": 1, "enabled": True, "mode": "verified", "grants": [],
            "tools": {"default": True, "overrides": {}}}


def validate(value, *, registry=DEFAULT_REGISTRY):
    required = {"version", "enabled", "grants", "tools"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"mode", "enabled_channels"}:
        raise ValueError("Invalid approval store fields")
    channels = value.get("enabled_channels", DEFAULT_ENABLED)
    if not isinstance(channels, (list, tuple)):
        raise ValueError("Invalid stored enabled channels")
    for channel in channels:
        adapter_id(channel)
    if len(channels) != len(set(channels)):
        raise ValueError("Duplicate stored enabled channel")
    if value.get("mode", "verified") not in ("verified", "always_on"):
        raise ValueError("Invalid activation mode")
    if type(value["version"]) is not int or value["version"] != 1 or type(value["enabled"]) is not bool:
        raise ValueError("Invalid approval store version or enabled flag")
    if not isinstance(value["grants"], list):
        raise ValueError("Invalid grants")
    seen, ids = set(), set()
    for row in value["grants"]:
        required_grant = {"id", "channel", "account", "contact", "enabled"}
        if (not isinstance(row, dict) or not required_grant <= set(row)
                or set(row) - required_grant - {"adapter_version"}):
            raise ValueError("Invalid grant fields")
        # Reading storage never rewrites an identity through newly installed
        # code. Missing adapters and incompatible versions remain inert records.
        channel = adapter_id(row["channel"])
        account = row["account"]
        contact = opaque_id(row["contact"])
        if not isinstance(account, str) or not account or account != account.strip():
            raise ValueError("Invalid stored account")
        version = row.get("adapter_version", "1")
        if not isinstance(version, str) or not version or len(version) > 64 or opaque_id(version) != version:
            raise ValueError("Invalid stored adapter version")
        key = channel, account, contact
        if not isinstance(row["id"], str) or len(row["id"]) != 32 or key in seen or row["id"] in ids:
            raise ValueError("Duplicate or invalid grant")
        if type(row["enabled"]) is not bool or key != (row["channel"], row["account"], row["contact"]):
            raise ValueError("Grant must be canonical")
        seen.add(key)
        ids.add(row["id"])
    tools = value["tools"]
    if not isinstance(tools, dict) or set(tools) != {"default", "overrides"}:
        raise ValueError("Invalid tool policy")
    if type(tools["default"]) is not bool or not isinstance(tools["overrides"], dict):
        raise ValueError("Invalid tool defaults")
    if any(not isinstance(k, str) or not k.strip() or k != k.strip() or type(v) is not bool
           for k, v in tools["overrides"].items()):
        raise ValueError("Tool flags must map exact names to booleans")
    return {**value, "mode": value.get("mode", "verified")}


class ApprovalStore:
    """Reread on authorization so revocation and tool changes need no restart."""
    def __init__(self, path, *, registry=DEFAULT_REGISTRY):
        self.path = Path(path).expanduser()
        self.registry = registry

    def read(self, *, require_existing=False):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            if require_existing:
                raise ValueError("Configured approval store is missing") from None
            return empty()
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size > 1024 * 1024):
                raise ValueError("Approval store must be an owned, private regular file")
            with os.fdopen(fd, "r") as stream:
                fd = None
                return validate(json.load(stream), registry=self.registry)
        finally:
            if fd is not None:
                os.close(fd)

    def approved(self, channel, account, contact):
        channel = channel.strip().lower() if isinstance(channel, str) else ""
        if self.registry.get(channel) is None:
            return None
        key = pair(channel, account, contact, registry=self.registry)
        state = self.read()
        if state["enabled"] and key[0] in state.get("enabled_channels", DEFAULT_ENABLED):
            for row in state["grants"]:
                if row["enabled"] and (row["channel"], row["account"], row["contact"]) == key and self._compatible(row):
                    return row["id"]
        return None

    def active(self, grant_id):
        state = self.read()
        return bool(grant_id and state["enabled"] and any(
            row["id"] == grant_id and row["enabled"]
            and row["channel"] in state.get("enabled_channels", DEFAULT_ENABLED)
            and self._compatible(row) for row in state["grants"]))

    def _compatible(self, row):
        adapter = self.registry.get(row["channel"])
        if adapter is None or row.get("adapter_version", "1") != adapter.identity_version:
            return False
        try:
            return self.registry.canonical_contact(row["channel"], row["contact"]) == row["contact"]
        except (TypeError, ValueError):
            return False

    def channels(self):
        return frozenset(channel for channel in self.read().get("enabled_channels", DEFAULT_ENABLED)
                         if self.registry.get(channel) is not None)

    def channel_enabled(self, channel):
        state = self.read()
        return bool(state["enabled"] and channel in state.get("enabled_channels", DEFAULT_ENABLED)
                    and self.registry.get(channel) is not None)

    def set_channels(self, channels):
        selected = self.registry.select_enabled(channels)
        def change(state):
            removed = set(state.get("enabled_channels", DEFAULT_ENABLED)) - selected
            # Re-enabling a transport must not revive capabilities issued before
            # it was disabled. Keep the contact choice but invalidate its grant.
            for row in state["grants"]:
                if row["channel"] in removed:
                    row["id"] = secrets.token_hex(16)
            state["enabled_channels"] = sorted(selected)
        return self.update(change)

    def tool_enabled(self, name):
        if not isinstance(name, str) or not name or name != name.strip():
            return False
        state = self.read()
        return state["enabled"] and state["tools"]["overrides"].get(name, state["tools"]["default"])

    def always_on(self):
        """Operator policy only; never an assertion about caller identity."""
        state = self.read(require_existing=True)
        return state["enabled"] and state["mode"] == "always_on"

    def set_mode(self, mode):
        if mode not in ("verified", "always_on"):
            raise ValueError("Mode must be verified or always_on")
        return self.update(lambda state: state.update(mode=mode))

    def update(self, change):
        """Local operator action only. Serialize writers and replace the file atomically."""
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lockfd = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        candidate = None
        try:
            info = os.fstat(lockfd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError("Invalid approval-store lock")
            fcntl.flock(lockfd, fcntl.LOCK_EX)
            state = self.read()
            change(state)
            validate(state, registry=self.registry)
            fd, candidate = tempfile.mkstemp(prefix=".owner-approvals-", dir=parent)
            with os.fdopen(fd, "w") as stream:
                json.dump(state, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(candidate, self.path)
            candidate = None
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return state
        finally:
            if candidate is not None:
                os.unlink(candidate)
            os.close(lockfd)

    def set_contact(self, channel, account, contact, enabled):
        key = pair(channel, account, contact, registry=self.registry)
        version = self.registry.get(key[0]).identity_version
        def change(state):
            row = next((r for r in state["grants"] if (r["channel"], r["account"], r["contact"]) == key), None)
            if row is None:
                row = dict(zip(("channel", "account", "contact"), key))
                row.update(id=secrets.token_hex(16), enabled=enabled, adapter_version=version)
                state["grants"].append(row)
            elif row["enabled"] != enabled or not self._compatible(row):
                row.update(id=secrets.token_hex(16), enabled=enabled, adapter_version=version)
        return self.update(change)

    def set_tool(self, name, enabled):
        if not isinstance(name, str) or not name.strip() or name != name.strip() or type(enabled) is not bool:
            raise ValueError("An exact tool name and boolean are required")
        def change(state):
            state["tools"]["overrides"][name] = enabled
        return self.update(change)
