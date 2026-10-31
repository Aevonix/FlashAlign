# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Create an original, configurable long-reading fixture without model downloads."""
import argparse
from pathlib import Path

from flashalign.benchmark import private_directory, write_private


def make_case(paragraphs):
    if type(paragraphs) is not int or not 1 <= paragraphs <= 100_000:
        raise ValueError("paragraphs must be between one and 100,000")
    records = []
    for index in range(paragraphs):
        records.append(
            f"Log entry {index:06d}. The fictional observatory checked its weather instruments, "
            "reviewed a chart of harmless cloud observations, and filed the completed inventory. "
            "The staff put spare notebooks on the blue shelf and left the viewing platform clear. "
            "This routine entry does not change any appointment or calibration schedule.")
        if index == paragraphs // 2:
            records.append("Calibration appointment: the observatory's spectrometer is calibrated in the Cedar room on Tuesday.")
    question = "Read these fictional records. What room and weekday are specified for the spectrometer calibration? Answer with only the room and weekday."
    return {"id": f"observatory-{paragraphs}-records", "group": "capability", "demo_safe": False,
            "messages": [{"role": "user", "content": "\n\n".join(records) + "\n\n" + question}],
            "expected_answer": "Cedar, Tuesday"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paragraphs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private directory")
    args = parser.parse_args()
    case = make_case(args.paragraphs)
    directory = private_directory(args.output)
    write_private(directory / "cases.jsonl", [case], lines=True)
    write_private(directory / "selection.json", {"generator": "observatory-v1", "paragraphs": args.paragraphs,
                                                "expected_answer": "Cedar, Tuesday",
                                                "token_count": "Tokenizer-dependent; use the recorded prompt_tokens."})
    print("Original long-context fixture created; token count depends on the selected tokenizer.")


if __name__ == "__main__":
    main()
