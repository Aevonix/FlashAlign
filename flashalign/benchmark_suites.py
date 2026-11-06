# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Trusted evaluation-suite registrations; input files never select Python code."""
from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .benchmark_io import atomic_bytes, atomic_json, digest, loads, private_directory, read_bytes, read_json


FR443_REVISION = "a2a8051a3f4ff2593f6b6e9c22f9929b3244c742"
FR443_SHA256 = "ba6431ca900cf41eb4f621cbf04d7f084187a16ec1ba6d1b8b6663967302d2e6"


@dataclass(frozen=True)
class Suite:
    name: str
    corpus_id: str
    revision: str
    source_sha256: str
    row_count: int
    datasets: tuple
    labels: tuple
    unique_prompts: int
    scorer: str
    source_url: str
    validator: object = None
    purpose: str = "evaluation"

    def identity(self):
        return {"name": self.name, "corpus_id": self.corpus_id, "revision": self.revision,
                "source_sha256": self.source_sha256, "row_count": self.row_count,
                "datasets": dict(self.datasets), "labels": dict(self.labels),
                "unique_prompts": self.unique_prompts, "scorer": self.scorer,
                "source_url": self.source_url, "purpose": self.purpose}

    def validate(self, raw):
        if hashlib.sha256(raw).hexdigest() != self.source_sha256:
            raise ValueError("Evaluation corpus digest mismatch")
        rows = [loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) != self.row_count:
            raise ValueError("Evaluation corpus row count mismatch")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"id", "dataset", "label", "prompt"}:
                raise ValueError("Invalid evaluation row fields")
            if any(not isinstance(value, str) or not value.strip() for value in row.values()):
                raise ValueError("Evaluation row values must be nonempty strings")
            if row["id"] in ids:
                raise ValueError("Duplicate evaluation row ID")
            ids.add(row["id"])
        if Counter(row["dataset"] for row in rows) != Counter(dict(self.datasets)):
            raise ValueError("Evaluation dataset counts mismatch")
        if Counter(row["label"] for row in rows) != Counter(dict(self.labels)):
            raise ValueError("Evaluation label counts mismatch")
        if len({row["prompt"] for row in rows}) != self.unique_prompts:
            raise ValueError("Evaluation duplicate-text contract mismatch")
        if self.validator:
            self.validator(rows)
        return rows


class SuiteRegistry:
    def __init__(self, suites=()):
        self._suites = {}
        for suite in suites:
            self.register(suite)

    def register(self, suite):
        if not isinstance(suite, Suite) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", suite.name):
            raise ValueError("Invalid trusted suite registration")
        if suite.purpose not in {"evaluation", "calibration"}:
            raise ValueError("Invalid suite purpose")
        if suite.name in self._suites:
            raise ValueError("Duplicate suite registration")
        self._suites[suite.name] = suite

    def get(self, name):
        try:
            return self._suites[name]
        except KeyError:
            raise ValueError("Unknown evaluation suite") from None

    def names(self):
        return tuple(sorted(self._suites))


def _fr443_rows(rows):
    for index, row in enumerate(rows, 1):
        if row["id"] != f"r1fr443-{index:04d}":
            raise ValueError("FR443 source order or ID mismatch")
        expected = ("advbench", "harmful") if index <= 146 else (("strongreject", "harmful") if index <= 295 else None)
        if expected and (row["dataset"], row["label"]) != expected:
            raise ValueError("FR443 source classification mismatch")
        if index > 295 and (row["dataset"] != "xstest" or row["label"] not in ("safe", "unsafe")):
            raise ValueError("FR443 XSTest classification mismatch")


FR443 = Suite("fr443", "R1-FALSE-REFUSAL-443-v1", FR443_REVISION, FR443_SHA256, 443,
              (("advbench", 146), ("strongreject", 149), ("xstest", 148)),
              (("harmful", 295), ("safe", 75), ("unsafe", 73)), 441, "fourway-refusal-v1",
              "https://github.com/Blackfrost-AI/r1-false-refusal-443", _fr443_rows)
SUITES = SuiteRegistry((FR443,))


def prepare_suite(name, source, output, *, registry=SUITES):
    suite = registry.get(name)
    raw = read_bytes(source)
    rows = suite.validate(raw)
    root = private_directory(output, create=True)
    atomic_bytes(root / "source.jsonl", raw, replace=False)
    # A JSON array preserves the exact source rows without bundling third-party data.
    atomic_json(root / "cases.json", rows, replace=False)
    atomic_json(root / "suite.json", {"schema": "flashalign.prepared-suite.v1", "suite": suite.identity(),
                                       "cases_sha256": digest(rows), "contains_responses": False}, replace=False)
    return root


def load_prepared(path, *, registry=SUITES):
    path = private_directory(path)
    declaration = read_json(path / "suite.json")
    if set(declaration) != {"schema", "suite", "cases_sha256", "contains_responses"} or declaration["schema"] != "flashalign.prepared-suite.v1":
        raise ValueError("Unsupported prepared suite")
    if not isinstance(declaration["suite"], dict):
        raise ValueError("Invalid prepared suite descriptor")
    suite = registry.get(declaration["suite"].get("name"))
    if declaration["suite"] != suite.identity() or declaration["contains_responses"] is not False:
        raise ValueError("Prepared suite declaration mismatch")
    rows = read_json(path / "cases.json")
    if not isinstance(rows, list) or digest(rows) != declaration["cases_sha256"] or len(rows) != suite.row_count:
        raise ValueError("Prepared suite changed")
    if rows != suite.validate(read_bytes(path / "source.jsonl")):
        raise ValueError("Prepared rows differ from the pinned corpus")
    return suite, rows, declaration
