# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Discovery and loading of trusted, installed adapters.

Listing channel entry points does not import them; channel loading requires
local selection and version pins. Engine, model and profile registries load all
their installed entry points. Request bodies cannot select Python code.
"""
from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import stat
import tempfile

from .channels import ChannelAdapter, DEFAULT_REGISTRY, adapter_id, opaque_id


ENTRYPOINT_GROUP = "flashalign.channel_adapters"


def _path(store_path):
    return Path(str(Path(store_path).expanduser()) + ".adapters.json")


def _installed(group=ENTRYPOINT_GROUP):
    entries = {}
    for entry in metadata.entry_points(group=group):
        adapter_id(entry.name)
        if entry.name in entries:
            raise ValueError("Duplicate installed adapter entry point")
        if entry.dist is None:
            raise ValueError("An adapter needs installed distribution metadata")
        name = opaque_id(entry.dist.name)
        version = opaque_id(entry.dist.version)
        entries[entry.name] = (entry, name, version)
    return entries


def _read(store_path):
    try:
        fd = os.open(_path(store_path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return []
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size > 65536):
            raise ValueError("Channel adapter selection must be an owned, private regular file")
        with os.fdopen(fd, "r") as stream:
            fd = None
            value = json.load(stream)
    finally:
        if fd is not None:
            os.close(fd)
    if (not isinstance(value, dict) or set(value) != {"version", "plugins"}
            or type(value["version"]) is not int or value["version"] != 1
            or not isinstance(value["plugins"], list)):
        raise ValueError("Invalid channel adapter selection")
    seen = set()
    for row in value["plugins"]:
        if not isinstance(row, dict) or set(row) != {"name", "distribution", "version", "adapter", "identity_version"}:
            raise ValueError("Invalid channel adapter selection entry")
        adapter_id(row["name"])
        adapter_id(row["adapter"])
        for key in ("distribution", "version", "identity_version"):
            opaque_id(row[key])
        if row["name"] in seen:
            raise ValueError("Duplicate selected channel adapter")
        seen.add(row["name"])
    return value["plugins"]


@contextmanager
def _locked(store_path):
    path = _path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
            raise ValueError("Invalid channel adapter selection lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def _write(path, rows):
    candidate = None
    try:
        fd, candidate = tempfile.mkstemp(prefix=".channel-adapters-", dir=path.parent)
        with os.fdopen(fd, "w") as stream:
            json.dump({"version": 1, "plugins": rows}, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(candidate, path)
        candidate = None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if candidate is not None:
            os.unlink(candidate)


@contextmanager
def entry_point_context(entry):
    """Identify the installed extension when loading or registration fails."""
    try:
        yield
    except Exception as error:
        extension = f"Adapter entry point {entry.name!r} from distribution {entry.dist.name!r}"
        error_type = (ValueError if isinstance(error, ValueError) else
                      OSError if isinstance(error, OSError) else RuntimeError)
        named = error_type(f"{extension}: {error}")
        # The names alone, for a diagnostic that must not repeat the failure's own text.
        named.extension = extension
        raise named from error


def _load(entry, adapter_type=ChannelAdapter):
    with entry_point_context(entry):
        loaded = entry.load()
        adapter = loaded if isinstance(loaded, adapter_type) else loaded() if callable(loaded) else None
        if not isinstance(adapter, adapter_type):
            raise ValueError("An adapter entry point must provide the expected adapter type")
        return adapter


def load_adapters(group, adapter_type, *, with_sources=False):
    """Load trusted installed extensions with the channel loader's validation.

    Load and factory exceptions propagate. Channel identity extensions still
    use their separate explicit selection and distribution pins below.
    """
    loaded = tuple((entry, _load(entry, adapter_type)) for entry, _, _ in _installed(group).values())
    return loaded if with_sources else tuple(adapter for _, adapter in loaded)


def available_channel_adapters(store_path):
    """List installed names and selected pins without executing plugin code."""
    selected = {row["name"]: row for row in _read(store_path)}
    installed = _installed()
    result = []
    for name in sorted(set(selected) | set(installed)):
        current = installed.get(name)
        pin = selected.get(name)
        matches = bool(current and pin and current[1:] == (pin["distribution"], pin["version"]))
        result.append({"name": name, "installed": current is not None, "enabled": pin is not None,
                       "pin_matches": matches, "distribution": current[1] if current else pin["distribution"],
                       "version": current[2] if current else pin["version"]})
    return result


def selected_channel_adapter(store_path, name):
    return next((row["adapter"] for row in _read(store_path) if row["name"] == name), None)


def enable_channel_adapter(store_path, name):
    """Execute an explicitly chosen installed adapter and pin its identity."""
    adapter_id(name)
    installed = _installed().get(name)
    if installed is None:
        raise ValueError("Channel adapter entry point is not installed")
    entry, distribution, version = installed
    adapter = _load(entry)
    # Installed adapters extend built-ins; they cannot replace built-in identity rules.
    with entry_point_context(entry):
        DEFAULT_REGISTRY.with_adapter(adapter)
    row = {"name": name, "distribution": distribution, "version": version,
           "adapter": adapter.id, "identity_version": adapter.identity_version}
    with _locked(store_path) as path:
        previous = _read(store_path)
        rows = [item for item in previous if item["name"] != name]
        if any(item["adapter"] == adapter.id for item in rows):
            raise ValueError("Channel adapter ID is already selected")
        _revoke_selected(store_path, [item for item in previous if item["name"] == name])
        _write(path, [*rows, row])
    return adapter.id


def disable_channel_adapter(store_path, name):
    adapter_id(name)
    with _locked(store_path) as path:
        rows = _read(store_path)
        _revoke_selected(store_path, [row for row in rows if row["name"] == name])
        _write(path, [row for row in rows if row["name"] != name])


def _revoke_selected(store_path, rows):
    if rows:
        from .contacts import ApprovalStore
        # Revocation only reads opaque stored grants; it must work even when
        # the selected plugin has disappeared or can no longer be imported.
        ApprovalStore(store_path, registry=DEFAULT_REGISTRY).revoke_channels(row["adapter"] for row in rows)


def load_channel_registry(store_path):
    """Load only selected, unchanged pins. Missing/upgraded plugins stay inert."""
    selected = _read(store_path)
    if not selected:
        return DEFAULT_REGISTRY
    installed = _installed()
    registry = DEFAULT_REGISTRY
    for row in selected:
        current = installed.get(row["name"])
        if current is None or current[1:] != (row["distribution"], row["version"]):
            continue
        adapter = _load(current[0])
        if adapter.id != row["adapter"] or adapter.identity_version != row["identity_version"]:
            continue
        # Grants bind the distribution pin as well as the plugin's identity
        # version, so explicitly enabling an upgrade cannot revive old grants.
        identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:32]
        bound = ChannelAdapter(adapter.id, adapter.normalize_contact, adapter.capabilities,
                               adapter.label, "installed-" + identity)
        with entry_point_context(current[0]):
            registry = registry.with_adapter(bound)
    return registry
