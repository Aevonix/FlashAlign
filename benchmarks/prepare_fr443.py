# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Verify the separately obtained FR443 corpus into a private run directory."""
import argparse
from pathlib import Path

from flashalign.benchmark_suites import prepare_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_suite("fr443", args.source, args.output)
    print("Prepared the pinned 443-row corpus. No model requests were sent.")


if __name__ == "__main__":
    main()
