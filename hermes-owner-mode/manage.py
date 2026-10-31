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
    commands.add_parser("list")
    args = parser.parse_args(argv)
    store = ApprovalStore(args.store)
    if args.command in ("approve", "revoke"):
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


if __name__ == "__main__":
    main()
