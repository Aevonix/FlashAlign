# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Transport capabilities installed by trusted application code.

Configuration selects registered adapters. Request fields never register an
adapter or prove that its transport delivered a message.
"""
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Callable


DEFAULT_ENABLED = ("whatsapp", "rcs")
CAPABILITIES = frozenset({"native_message", "native_session", "custom_authority", "local_interactive"})
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_PHONE = re.compile(r"\+?[0-9]{8,15}")
_E164 = re.compile(r"\+[1-9][0-9]{7,14}")
_LID = re.compile(r"[0-9]{5,32}@lid")


def opaque_id(value):
    """An exact, bounded transport identifier; never a display-name lookup."""
    if not isinstance(value, str):
        raise ValueError("Identity must be a string")
    value = value.strip()
    if not value or len(value) > 256 or any(ord(c) < 33 or ord(c) == 127 for c in value):
        raise ValueError("Invalid transport identity")
    return value


def adapter_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("Invalid adapter ID")
    return value


def _phone(value):
    value = opaque_id(value)
    value = "+" + value.lstrip("+") if _PHONE.fullmatch(value) else value
    if not _E164.fullmatch(value):
        raise ValueError("Expected an E.164 phone number")
    return value


def _whatsapp(value):
    value = opaque_id(value)
    if value.endswith("@s.whatsapp.net"):
        value = value[:-len("@s.whatsapp.net")]
    return value if _LID.fullmatch(value) else _phone(value)


def _telegram(value):
    value = opaque_id(value)
    # Native Telegram user IDs are positive decimal integers. Usernames,
    # phone numbers and group/chat IDs are not contact identities.
    if not re.fullmatch(r"[1-9][0-9]{0,19}", value):
        raise ValueError("Expected a native Telegram user ID")
    return value


def _local_user(value):
    value = opaque_id(value)
    if not re.fullmatch(r"uid:(?:0|[1-9][0-9]{0,19})", value):
        raise ValueError("Expected a local uid identity")
    return value


@dataclass(frozen=True)
class ChannelAdapter:
    id: str
    normalize_contact: Callable[[str], str]
    capabilities: frozenset[str]
    label: str = ""
    # Increment when normalization or the meaning of a subject/account changes.
    # Existing grants bind this version and require explicit renewal afterward.
    identity_version: str = "1"

    def __post_init__(self):
        adapter_id(self.id)
        if not isinstance(self.identity_version, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", self.identity_version):
            raise ValueError("Invalid adapter identity version")
        if not callable(self.normalize_contact):
            raise ValueError("An adapter needs an identity normalizer")
        caps = frozenset(self.capabilities)
        if not caps or not caps <= CAPABILITIES:
            raise ValueError("Invalid adapter capabilities")
        object.__setattr__(self, "capabilities", caps)


class AdapterRegistry:
    """Immutable registry assembled by installed code, never request payloads."""
    def __init__(self, adapters):
        entries = {}
        for adapter in adapters:
            if not isinstance(adapter, ChannelAdapter) or adapter.id in entries:
                raise ValueError("Duplicate or invalid registered adapter")
            entries[adapter.id] = adapter
        self._entries = MappingProxyType(entries)

    def get(self, channel):
        return self._entries.get(channel) if isinstance(channel, str) else None

    def ids(self, capability=None):
        return frozenset(key for key, adapter in self._entries.items()
                         if capability is None or capability in adapter.capabilities)

    def supports(self, channel, capability):
        adapter = self.get(channel)
        return adapter is not None and capability in adapter.capabilities

    def canonical_contact(self, channel, value):
        adapter = self.get(channel)
        if adapter is None:
            raise ValueError("Channel has no installed adapter")
        result = opaque_id(adapter.normalize_contact(value))
        if adapter.normalize_contact(result) != result:
            raise ValueError("Adapter identity normalization is not stable")
        return result

    def select_enabled(self, channels):
        if not isinstance(channels, (tuple, list, set, frozenset)):
            raise ValueError("Enabled channels must be a list of registered adapter IDs")
        if any(not isinstance(c, str) or c not in self._entries for c in channels):
            raise ValueError("Enabled channel has no installed adapter")
        if len(channels) != len(set(channels)):
            raise ValueError("Duplicate enabled channel")
        return frozenset(channels)

    def with_adapter(self, adapter):
        return AdapterRegistry((*self._entries.values(), adapter))


DEFAULT_REGISTRY = AdapterRegistry((
    ChannelAdapter("whatsapp", _whatsapp, frozenset({"native_message", "native_session"}), "WhatsApp"),
    ChannelAdapter("sms", _phone, frozenset({"native_message", "native_session"}), "SMS"),
    ChannelAdapter("rcs", _phone, frozenset({"native_message", "native_session"}), "RCS"),
    ChannelAdapter("telegram", _telegram, frozenset({"native_message", "native_session"}), "Telegram"),
    ChannelAdapter("local-terminal", _local_user, frozenset({"local_interactive"}), "Local terminal"),
    # Existing installed identity authorities may prove these transports. They
    # are not native message adapters and are never inferred from prompt text.
    ChannelAdapter("voice", opaque_id, frozenset({"custom_authority"}), "Voice"),
    ChannelAdapter("companion", opaque_id, frozenset({"custom_authority"}), "Companion"),
))
