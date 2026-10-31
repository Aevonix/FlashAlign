# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Convert a separately obtained, pinned XSTest CSV to private benchmark cases."""
import argparse
import csv
import hashlib
from pathlib import Path
import random

from flashalign.benchmark import private_directory, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private output directory")
    parser.add_argument("--group", choices=("safe", "unsafe", "all"), default="safe")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    cases = []
    for index, row in enumerate(source):
        group = "unsafe" if row["type"].startswith("contrast_") else "safe"
        if args.group != "all" and group != args.group:
            continue
        cases.append({"id": "xstest:" + str(row.get("id", index)), "group": group,
                      "demo_safe": False, "messages": [{"role": "user", "content": row["prompt"]}]})
    random.Random(args.seed).shuffle(cases)
    if args.limit:
        cases = cases[:args.limit]
    target = private_directory(args.output)
    write_private(target / "cases.jsonl", cases, lines=True)
    write_private(target / "source.json", {
        "repository": "https://github.com/paul-rottger/xstest",
        "expected_revision": "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
        "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "license": "CC-BY-4.0", "citation": "Röttger et al., XSTest, NAACL 2024, https://aclanthology.org/2024.naacl-long.301/",
        "seed": args.seed, "group": args.group, "selected_cases": len(cases),
        "revision_note": "Verify the input file against the named upstream checkout; expected_revision is not a download verification."
    })
    print(f"Prepared {len(cases)} cases. No responses were generated.")


if __name__ == "__main__":
    main()
