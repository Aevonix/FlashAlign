# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Manage Aevonix FlashAlign contacts and tool access locally."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("owner_mode", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from owner_mode.contacts import ApprovalStore
from owner_mode.channels import DEFAULT_REGISTRY
from owner_mode.channel_plugins import (available_channel_adapters, disable_channel_adapter,
                                        enable_channel_adapter)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("approve", "revoke"):
        p = commands.add_parser(name)
        p.add_argument("channel")
        p.add_argument("contact")
        p.add_argument("--account", default="default")
    p = commands.add_parser("tool")
    p.add_argument("name")
    p.add_argument("state", choices=("on", "off"))
    p = commands.add_parser("mode", help="Choose verified contacts or Always On for all routed requests")
    p.add_argument("value", choices=("verified", "always_on"))
    p = commands.add_parser("channels", help="List installed adapters or replace the enabled selection")
    p.add_argument("--set", dest="replace_channels", action="store_true",
                   help="Replace enabled channels with the listed installed adapter IDs")
    p.add_argument("adapters", nargs="*")
    p = commands.add_parser("adapters", help="Explicitly enable or disable installed channel adapter entry points")
    p.add_argument("action", choices=("list", "enable", "disable"))
    p.add_argument("name", nargs="?")
    commands.add_parser("list")
    commands.add_parser("pause", help="Pause profile selection without changing saved access settings")
    commands.add_parser("resume", help="Resume the saved access mode and settings")
    args = parser.parse_args(argv)
    store = ApprovalStore(args.store, registry=DEFAULT_REGISTRY if args.command == "adapters" else None)
    if args.command == "adapters":
        if args.action == "list":
            if args.name:
                parser.error("Adapter list does not accept a name")
        else:
            if not args.name:
                parser.error("Adapter enable/disable requires an installed entry-point name")
            if args.action == "enable":
                enable_channel_adapter(args.store, args.name)
            else:
                disable_channel_adapter(args.store, args.name)
        state = {"plugins": available_channel_adapters(args.store)}
    elif args.command in ("pause", "resume"):
        state = store.set_enabled(args.command == "resume")
    elif args.command in ("approve", "revoke"):
        state = store.set_contact(args.channel, args.account, args.contact, args.command == "approve")
    elif args.command == "tool":
        state = store.set_tool(args.name, args.state == "on")
    elif args.command == "mode":
        state = store.set_mode(args.value)
    elif args.command == "channels":
        if args.replace_channels:
            state = store.set_channels(args.adapters)
        elif args.adapters:
            parser.error("Use channels --set to change enabled adapters")
        else:
            enabled = store.channels()
            state = {"adapters": [{"id": name, "label": store.registry.get(name).label,
                       "capabilities": sorted(store.registry.get(name).capabilities),
                       "enabled": name in enabled}
                      for name in sorted(store.registry.ids())]}
    else:
        state = store.read()
    print(json.dumps(state, indent=2))


def cli(argv=None):
    """Main CLI adapter: accept the store option before or after the command."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--store", type=Path, default=Path.home() / '.config/owner-mode/approvals.json')
    args, remaining = parser.parse_known_args(argv)
    return main(["--store", str(args.store), *remaining])


if __name__ == "__main__":
    main()
