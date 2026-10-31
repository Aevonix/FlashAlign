# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Run the loopback provider bridge for a trusted native harness adapter."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("owner_mode", root / "__init__.py", submodule_search_locations=[str(root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from owner_mode.bridge import Bridge
from owner_mode.contacts import ApprovalStore
from owner_mode.proxy import ProviderProxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    bridge = Bridge(ApprovalStore(cfg["approvals_file"]), cfg["control_key_file"], cfg["routes"],
                    instance_id=os.environ.get('FLASHALIGN_BRIDGE_INSTANCE'))
    proxy = ProviderProxy(bridge.policy, cfg["routes"], cfg["key_file"], port=cfg["port"], bridge=bridge,
                          approvals=bridge.store)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    proxy.start()
    try:
        stop.wait()
    finally:
        proxy.close()


if __name__ == "__main__":
    main()
