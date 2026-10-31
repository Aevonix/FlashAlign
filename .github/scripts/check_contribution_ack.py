# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Check the existing PR declaration without treating PR text as executable code."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re


TERMS_VERSION = "1.0"
DECLARATION = (
    "I agree to the FlashAlign contribution terms version 1.0 for this "
    "contribution and have authority to grant the rights described there."
)


def acknowledgment_checked(body):
    """Accept one checked declaration outside comments, quotations and code."""
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    items = []
    current = None
    fence = None
    for line in body.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if marker:
            char = marker.group(1)[0]
            if fence is None:
                fence = (char, len(marker.group(1)))
            elif fence[0] == char and len(marker.group(1)) >= fence[1] and not line[marker.end():].strip():
                fence = None
            current = None
            continue
        if fence is not None or line.lstrip().startswith(">"):
            current = None
            continue
        match = re.match(r"^ {0,3}[-*+] \[([ xX])\] (.+)$", line)
        if match:
            current = [match.group(1).lower() == "x", match.group(2)]
            items.append(current)
        elif current is not None and line.startswith(("  ", "\t")) and line.strip():
            current[1] += " " + line.strip()
        else:
            current = None
    matches = [checked for checked, text in items
               if " ".join(text.split()) == DECLARATION]
    return matches == [True]


def build_record(event, terms, observed_at=None):
    pr = event.get("pull_request") or {}
    if not acknowledgment_checked(pr.get("body")):
        raise ValueError(
            "Check the existing contribution-terms acknowledgment in the pull "
            "request body after reading legal/CLA-TERMS.md."
        )
    if not re.search(r"^Version 1\.0(?:,|$)", terms.decode("utf-8"), re.M):
        raise ValueError("The acknowledgment checker must match the reviewed terms version.")
    head = (pr.get("head") or {}).get("sha", "")
    base = (pr.get("base") or {}).get("sha", "")
    if not all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (head, base)):
        raise ValueError("The pull request must identify its complete head and base commits.")
    number = pr.get("number")
    if type(number) is not int or number <= 0:
        raise ValueError("The pull request number is missing.")
    return {
        "record_type": "pull_request_acknowledgment",
        "repository": (event.get("repository") or {}).get("full_name"),
        "pull_request": number,
        "acceptance_link": pr.get("html_url"),
        "submitting_account": (pr.get("user") or {}).get("login"),
        "observed_event_actor": (event.get("sender") or {}).get("login"),
        "observed_at": observed_at or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "covered_head_commit": head,
        "reviewed_base_commit": base,
        "terms_version": TERMS_VERSION,
        "terms_sha256": hashlib.sha256(terms).hexdigest(),
        "declaration": DECLARATION,
        "body_sha256": hashlib.sha256((pr.get("body") or "").encode()).hexdigest(),
        "maintainer_review_required": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--terms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = build_record(json.loads(args.event.read_text()), args.terms.read_bytes())
    except (ValueError, KeyError, TypeError) as exc:
        parser.exit(1, str(exc) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("Contribution acknowledgment recorded. Maintainer rights review is still required.")


if __name__ == "__main__":
    main()
