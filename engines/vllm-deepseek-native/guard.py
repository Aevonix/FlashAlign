# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Stop only the bound container if its owning launcher disappears."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

from supervisor import private_directory, process_identity, read_receipt, save, stop_owned

STOP = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    private_directory(args.record.parent)
    raw = read_receipt(args.record)
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError('Guard record changed')
    record = json.loads(raw)
    def stopped(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    save(args.record.parent / 'guard-ready.json', {'record_sha256': args.sha256})
    while not STOP and process_identity(record['parent_pid']) == record['parent_start']:
        time.sleep(1)
    stop_owned(record)
    save(args.record.parent / 'guard-finished.json', {'container_stopped': True})


if __name__ == '__main__':
    main()
