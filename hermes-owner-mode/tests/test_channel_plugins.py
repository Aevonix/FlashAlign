# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Installed channels use the same management and revocation path as built-ins."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_owner_mode
from support import load_cli
from owner_mode import channel_plugins
from owner_mode.channels import ChannelAdapter, DEFAULT_REGISTRY, opaque_id
from owner_mode.contacts import ApprovalStore
from owner_mode.native_authority import NativeAuthority
from owner_mode.policy import VerifiedTurn


manage = load_cli("manage")


class InstalledChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "approvals.json"
        self.adapter = ChannelAdapter("third-chat", opaque_id, frozenset({"native_message", "native_session"}))
        self.entry = SimpleNamespace(name="third-chat-plugin", dist=SimpleNamespace(name="third-chat", version="1.0"),
                                     load=mock.Mock(return_value=lambda: self.adapter))
        self.points = mock.patch.object(channel_plugins.metadata, "entry_points", return_value=[self.entry])
        self.points.start()
        self.addCleanup(self.points.stop)

    def run_cli(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            manage.main(["--store", str(self.path), *args])
        return json.loads(output.getvalue())

    def test_discovery_does_not_execute_or_enable_installed_code(self):
        self.assertIs(ApprovalStore(self.path).registry, DEFAULT_REGISTRY)
        result = self.run_cli("adapters", "list")
        self.assertFalse(result["plugins"][0]["enabled"])
        self.entry.load.assert_not_called()
        with self.assertRaises(ValueError):
            self.run_cli("approve", "third-chat", "user-1")

    def test_third_channel_cli_approve_revoke_and_disable_revoke_live_grants(self):
        self.run_cli("adapters", "enable", "third-chat-plugin")
        self.run_cli("channels", "--set", "third-chat")
        self.run_cli("approve", "third-chat", "user-1", "--account", "bot-1")
        running_store = ApprovalStore(self.path)
        grant = running_store.approved("third-chat", "bot-1", "user-1")
        self.assertTrue(grant)
        self.assertIsNone(running_store.approved("third-chat", "bot-2", "user-1"))
        self.run_cli("revoke", "third-chat", "user-1", "--account", "bot-1")
        self.assertFalse(running_store.active(grant))
        self.run_cli("approve", "third-chat", "user-1", "--account", "bot-1")
        renewed = running_store.approved("third-chat", "bot-1", "user-1")
        self.assertTrue(renewed)
        self.assertNotEqual(grant, renewed)
        self.run_cli("adapters", "disable", "third-chat-plugin")
        self.assertFalse(running_store.active(renewed))
        self.assertIsNone(ApprovalStore(self.path).registry.get("third-chat"))

    def test_upgraded_distribution_is_inert_until_enabled_and_grants_renewed(self):
        self.run_cli("adapters", "enable", "third-chat-plugin")
        self.run_cli("channels", "--set", "third-chat")
        self.run_cli("approve", "third-chat", "user-1")
        grant = ApprovalStore(self.path).approved("third-chat", "default", "user-1")
        self.entry.dist.version = "2.0"
        self.entry.load.reset_mock()
        self.assertIsNone(ApprovalStore(self.path).registry.get("third-chat"))
        self.entry.load.assert_not_called()
        self.run_cli("adapters", "enable", "third-chat-plugin")
        self.run_cli("channels", "--set", "third-chat")
        self.assertFalse(ApprovalStore(self.path).active(grant))
        self.assertIsNone(ApprovalStore(self.path).approved("third-chat", "default", "user-1"))
        self.run_cli("approve", "third-chat", "user-1")
        self.assertTrue(ApprovalStore(self.path).approved("third-chat", "default", "user-1"))

    def test_changed_identity_version_cannot_revive_approval(self):
        channel_plugins.enable_channel_adapter(self.path, self.entry.name)
        store = ApprovalStore(self.path)
        store.set_channels(["third-chat"])
        store.set_contact("third-chat", "default", "user-1", True)
        grant = store.approved("third-chat", "default", "user-1")
        self.adapter = ChannelAdapter("third-chat", opaque_id, frozenset({"native_message"}), identity_version="2")
        self.assertIsNone(ApprovalStore(self.path).registry.get("third-chat"))
        channel_plugins.enable_channel_adapter(self.path, self.entry.name)
        self.assertFalse(ApprovalStore(self.path).active(grant))

    def test_operator_can_disable_a_broken_plugin_without_importing_it(self):
        self.run_cli("adapters", "enable", self.entry.name)
        self.run_cli("channels", "--set", "third-chat")
        self.run_cli("approve", "third-chat", "user-1")
        running = ApprovalStore(self.path)
        grant = running.approved("third-chat", "default", "user-1")
        self.entry.load.side_effect = RuntimeError("broken installed plugin")
        self.run_cli("adapters", "list")
        self.run_cli("adapters", "disable", self.entry.name)
        self.assertFalse(running.active(grant))
        self.assertIsNone(ApprovalStore(self.path).registry.get("third-chat"))

    def test_selection_permissions_and_symlinks_are_rejected(self):
        channel_plugins.enable_channel_adapter(self.path, self.entry.name)
        selection = channel_plugins._path(self.path)
        selection.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private regular"):
            ApprovalStore(self.path)
        selection.chmod(0o600)
        target = selection.with_suffix(".saved")
        selection.rename(target)
        selection.symlink_to(target)
        with self.assertRaises(OSError):
            ApprovalStore(self.path)

    def test_builtin_collision_and_invalid_plugin_are_rejected(self):
        self.entry.load.return_value = DEFAULT_REGISTRY.get("whatsapp")
        with self.assertRaises(ValueError):
            channel_plugins.enable_channel_adapter(self.path, self.entry.name)
        self.entry.load.return_value = object()
        with self.assertRaises(ValueError):
            channel_plugins.enable_channel_adapter(self.path, self.entry.name)
        self.assertFalse(channel_plugins._path(self.path).exists())

    def test_explicit_registry_is_preserved(self):
        registry = DEFAULT_REGISTRY.with_adapter(self.adapter)
        with mock.patch.object(channel_plugins, "load_channel_registry", side_effect=AssertionError("not selected")):
            self.assertIs(ApprovalStore(self.path, registry=registry).registry, registry)

    def test_native_authority_uses_the_same_installed_registry(self):
        channel_plugins.enable_channel_adapter(self.path, self.entry.name)
        authority = NativeAuthority(self.path, VerifiedTurn)
        self.assertTrue(authority.registry.supports("third-chat", "native_message"))
        self.assertIs(authority.registry, authority.store.registry)


if __name__ == "__main__":
    unittest.main()
