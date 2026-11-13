# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Commands for pinned corpus preparation, strict collection and scoring."""
import argparse
import json
from pathlib import Path
import sys

from .benchmark_io import atomic_json, canonical, private_directory, read_json
from .benchmark_judge import judge_run, score_run
from .benchmark_protocol import collect_run
from .benchmark_reports import compare_reports, evidence_card, performance_summary, public_report
from .benchmark_suites import SUITES, prepare_suite


COMMANDS = {"prepare", "run", "judge", "report", "compare"}


class InputError(ValueError):
    """A CLI input role with a diagnostic safe to display."""


def input_error(error):
    if isinstance(error, InputError):
        return str(error)
    if isinstance(error, json.JSONDecodeError):
        return f'Invalid JSON at line {error.lineno}, column {error.colno}'
    if isinstance(error, UnicodeError):
        return 'Input is not valid UTF-8'
    if isinstance(error, OSError):
        return error.strerror or type(error).__name__
    if isinstance(error, ValueError):
        trace = error.__traceback__
        while trace is not None and trace.tb_next is not None:
            trace = trace.tb_next
        # Only fixed validation messages from our code are public. Dynamic
        # exception text can include a filename, response body or input value.
        if (trace is not None
                and trace.tb_frame.f_globals.get('__package__') in ('flashalign', 'owner_mode')
                and str(error) in trace.tb_frame.f_code.co_consts):
            return str(error)
    return type(error).__name__


def read_input(label, path, reader=read_json):
    try:
        return reader(path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise InputError(label + ': ' + input_error(error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Verify an operator-supplied pinned corpus")
    prepare.add_argument("--suite", choices=SUITES.names(), default="fr443")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Collect strict off/profile request pairs")
    for name in ("prepared", "config", "manifest", "settings", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--scale", type=float, default=1.)
    run.add_argument("--attempts", type=int, default=1)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limits", type=Path, help="Explicit bounded resource budgets; starts a strict-completion v2 run")
    judge = commands.add_parser("judge", help="Grade completed answers using an isolated judge")
    for name in ("run", "config", "output"):
        judge.add_argument("--" + name, type=Path, required=True)
    judge.add_argument("--resume", action="store_true")
    report = commands.add_parser("report", help="Validate scoring receipts and export aggregates")
    for name in ("run", "judgments", "output"):
        report.add_argument("--" + name, type=Path, required=True)
    report.add_argument("--public", action="store_true")
    report.add_argument("--card", action="store_true")
    compare = commands.add_parser("compare", help="Compare compatible measured profile reports")
    compare.add_argument("--reports", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_suite(args.suite, args.source, args.output)
            print(canonical({"suite": args.suite, "prepared": True, "model_requests": 0}))
            return 0
        if args.command == "run":
            result = collect_run(args.prepared, read_input("--config", args.config), read_input("--manifest", args.manifest),
                                 read_input("--settings", args.settings), args.output, scale=args.scale,
                                 attempts=args.attempts, resume=args.resume,
                                 limits=read_input("--limits", args.limits) if args.limits is not None else None)
            print(canonical(result))
            return 0 if result["collection_valid"] else 1
        if args.command == "judge":
            result = judge_run(args.run, read_input("--config", args.config), args.output, resume=args.resume)
            print(canonical({"final_behavioral_gate_valid": result["final_behavioral_gate_valid"], "modes": result["modes"]}))
            return 0 if result["final_behavioral_gate_valid"] else 1
        if args.command == "compare":
            result = compare_reports([read_input(f"--reports input {index + 1}", path) for index, path in enumerate(args.reports)])
        else:
            result = score_run(args.run, args.judgments)
            performance = performance_summary(args.run)
            if args.card:
                result = evidence_card(result, performance)
            elif args.public:
                result = public_report(result, performance)
            else:
                result["performance"] = performance
        private_directory(args.output.parent)
        atomic_json(args.output, result, replace=False)
        complete = result.get("final_behavioral_gate_valid", result.get("evaluation", {}).get("final_behavioral_gate_valid"))
        if args.command == "compare":
            complete = all(row["complete"] for row in result["candidates"])
        print(canonical({"written": True, "schema": result["schema"], "complete": complete}))
        return 0 if complete is not False else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("Evaluation stopped (" + args.command + "): " + input_error(exc) + ". Inspect private inputs and receipts.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
