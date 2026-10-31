# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Controlled, signed off/profile comparisons. Raw recordings stay private."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from . import Client


SCHEMA = "flashalign-benchmark-v1"
RESERVED_OPTIONS = {"messages", "model", "stream", "owner_mode", "owner_mode_scale", "n"}
LABELS = {"full_compliance", "partial_refusal", "full_refusal"}
SENSITIVE_FIELDS = {"authorization", "api_key", "api_key_file", "key", "key_file", "secret", "password", "headers", "user", "metadata"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def private_directory(path):
    """Never put model responses in a source checkout, including a linked worktree."""
    path = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("Choose a private output directory outside a Git checkout")
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.chmod(0o700)
    return path


def write_private(path, value, *, lines=False):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        if lines:
            for row in value:
                handle.write(canonical(row) + "\n")
        else:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_cases(cases):
    seen = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in seen:
            raise ValueError("Cases need unique nonempty string IDs")
        seen.add(case["id"])
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Each case needs a nonempty messages array")
        if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
               or not isinstance(m.get("content"), str) or set(m) != {"role", "content"}
               for m in messages):
            raise ValueError("Benchmark cases accept text-only role/content messages")
        if case.get("group", "custom") not in {"safe", "unsafe", "capability", "custom"}:
            raise ValueError("Unknown case group")
        if type(case.get("demo_safe", False)) is not bool:
            raise ValueError("demo_safe must be a boolean")
        if case.get("demo_safe") and case.get("group") != "safe":
            raise ValueError("Only reviewed safe cases may be marked demo_safe")
    if not cases:
        raise ValueError("At least one case is required")


def validate_manifest(manifest):
    """A narrow allowlist prevents client credentials and URLs entering reports."""
    fields = {
        "model": {"id", "served_id", "revision", "weights_sha256", "quantization", "tokenizer_revision"},
        "engine": {"name", "version", "commit", "settings_sha256"},
        "hardware": {"accelerator", "accelerator_count", "memory_gib", "power_limit_watts", "cpu"},
        "speculation": {"enabled", "method", "draft_model", "draft_revision", "draft_tokens"},
        "profile": {"name", "artifact_sha256"},
    }
    if not isinstance(manifest, dict) or set(manifest) != set(fields):
        raise ValueError("Manifest must contain model, engine, hardware, speculation and profile")
    for section, allowed in fields.items():
        values = manifest[section]
        if not isinstance(values, dict) or not values or set(values) - allowed:
            raise ValueError("Manifest has an unknown or empty section")
        if any(type(v) not in (str, int, float, bool) or isinstance(v, str) and
               (len(v) > 512 or "://" in v or "\n" in v) for v in values.values()):
            raise ValueError("Manifest values must be short, nonsecret scalar descriptions")
    for section, field in (("model", "id"), ("model", "revision"), ("model", "quantization"),
                           ("engine", "name"), ("engine", "version"),
                           ("profile", "name"), ("profile", "artifact_sha256")):
        if not isinstance(manifest[section].get(field), str) or not manifest[section][field]:
            raise ValueError("Manifest is missing required model, engine or profile provenance")
    if type(manifest["speculation"].get("enabled")) is not bool:
        raise ValueError("Declare whether speculative decoding is enabled")
    profile_hash = manifest["profile"]["artifact_sha256"]
    if len(profile_hash) != 64 or any(c not in "0123456789abcdef" for c in profile_hash):
        raise ValueError("profile.artifact_sha256 must be the SHA-256 of profile.safetensors")


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_options(value):
    if isinstance(value, dict):
        if any(str(key).lower() in SENSITIVE_FIELDS for key in value):
            raise ValueError("Generation settings must not contain credentials or caller metadata")
        for nested in value.values():
            validate_options(nested)
    elif isinstance(value, list):
        for nested in value:
            validate_options(nested)


def metrics(events, elapsed):
    deltas = [event["at_s"] for event in events
              if event.get("content") or event.get("reasoning")]
    content_deltas = [event["at_s"] for event in events if event.get("content")]
    usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
    timings = next((e["timings"] for e in reversed(events) if e.get("timings")), {})
    native = next((e["tensorfold"] for e in reversed(events) if e.get("tensorfold")), {})
    speculative = next((e["speculative"] for e in reversed(events) if e.get("speculative")), {})
    cached = next((e["cached_prompt_tokens"] for e in reversed(events)
                   if type(e.get("cached_prompt_tokens")) is int and e["cached_prompt_tokens"] >= 0), None)
    count = usage.get("completion_tokens")
    count = count if type(count) is int and count >= 0 else None
    decode = None
    decode_source = None
    if number(timings.get("predicted_n")) and number(timings.get("predicted_ms")) and timings["predicted_ms"] > 0:
        decode = timings["predicted_n"] / (timings["predicted_ms"] / 1000)
        decode_source = "timings.predicted_n / (timings.predicted_ms / 1000)"
    elif number(native.get("tokens_per_second")) and number(native.get("seconds")) and native["seconds"] > 0:
        decode = native["tokens_per_second"]
        decode_source = "tensorfold.tokens_per_second; tensorfold.seconds decode interval"
    # SSE events can contain several tokens, including speculative bursts. This is
    # an explicit estimate, never the engine's measured decode rate.
    estimate = None
    if count is not None and count > 1 and len(deltas) > 1 and deltas[-1] > deltas[0]:
        estimate = (count - 1) / (deltas[-1] - deltas[0])
    return {
        "ttft_s": deltas[0] if deltas else None,
        "first_answer_text_s": content_deltas[0] if content_deltas else None,
        "total_s": elapsed,
        "completion_tokens": count,
        "prompt_tokens": usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int else None,
        "cached_prompt_tokens": cached,
        "output_tokens_per_total_s": count / elapsed if count is not None and elapsed > 0 else None,
        "engine_decode_tps": decode,
        "engine_decode_tps_source": decode_source,
        "engine_decode_seconds": native.get("seconds") if number(native.get("seconds")) else None,
        "speculative": {key: value for key, value in speculative.items()
                        if key in {"rounds", "drafted", "accepted"} and type(value) is int and value >= 0},
        "stream_decode_tps_estimate": estimate,
        "finish_reason": next((e["finish_reason"] for e in reversed(events) if e.get("finish_reason")), None),
    }


def capture(client, case, profile, options, *, scale=1., clock=time.perf_counter):
    events = []
    started = clock()
    status = "ok"
    error_type = None
    try:
        with client.stream_chat(case["messages"], profile=profile, scale=scale, **options) as stream:
            for item in stream:
                event = {"at_s": clock() - started}
                choices = item.get("choices", [])
                if len(choices) > 1:
                    raise ValueError("The benchmark supports one completion per request")
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content", delta.get("reasoning"))
                    if content is not None and not isinstance(content, str):
                        raise ValueError("Non-text completion delta")
                    if reasoning is not None and not isinstance(reasoning, str):
                        raise ValueError("Non-text reasoning delta")
                    if content:
                        event["content"] = content
                    if reasoning:
                        event["reasoning"] = reasoning
                    if choices[0].get("finish_reason") in {"stop", "length", "tool_calls", "content_filter", "function_call"}:
                        event["finish_reason"] = choices[0]["finish_reason"]
                # Keep only numeric usage/timing fields; never copy arbitrary
                # response metadata, model headers, IDs or server error bodies.
                for field in ("usage", "timings", "tensorfold", "speculative"):
                    if isinstance(item.get(field), dict):
                        event[field] = {k: v for k, v in item[field].items() if number(v)}
                usage = item.get("usage")
                details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
                cached = details.get("cached_tokens") if isinstance(details, dict) else None
                if type(cached) is int and cached >= 0:
                    event["cached_prompt_tokens"] = cached
                native = item.get("tensorfold")
                if isinstance(native, dict):
                    ids = native.get("token_ids")
                    if isinstance(ids, list) and len(ids) <= 1_000_000 and all(type(x) is int and x >= 0 for x in ids):
                        event["token_ids"] = ids
                    token_hash = native.get("token_sha256")
                    if isinstance(token_hash, str) and len(token_hash) == 64 and all(c in "0123456789abcdef" for c in token_hash):
                        event["token_sha256"] = token_hash
                events.append(event)
        if not any(event.get("finish_reason") for event in events):
            status, error_type = "incomplete", "MissingFinishReason"
    except Exception as exc:
        status, error_type = "error", type(exc).__name__
    elapsed = clock() - started
    return {"status": status, "error_type": error_type, "events": events, "metrics": metrics(events, elapsed)}


def aggregate(records, manifest, options, *, cases_sha256, labels=None):
    measured = [r for r in records if r["phase"] == "measured"]
    modes = {}
    for mode in ("off", "profile"):
        rows = [r for r in measured if r["mode"] == mode]
        valid = [r for r in rows if r["status"] == "ok"]
        stats = {}
        for metric in ("ttft_s", "first_answer_text_s", "total_s", "completion_tokens", "cached_prompt_tokens",
                       "output_tokens_per_total_s", "engine_decode_tps", "stream_decode_tps_estimate"):
            values = sorted(r["metrics"][metric] for r in valid if r["metrics"].get(metric) is not None)
            stats[metric] = {"n": len(values), "mean": statistics.mean(values) if values else None,
                             "median": statistics.median(values) if values else None,
                             "p95": values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None}
        modes[mode] = {"requests": len(rows), "completed": len(valid),
                       "failures": dict(Counter(r["error_type"] for r in rows if r["status"] != "ok")),
                       "metrics": stats}
    report = {
        "schema": SCHEMA, "manifest": manifest, "manifest_provenance": "operator_supplied",
        "request_settings": options, "request_settings_sha256": digest(options),
        "profile_scales": sorted({r["requested_scale"] for r in measured if r["mode"] == "profile"}),
        "cases_sha256": cases_sha256, "unique_cases": len({r["case_id"] for r in measured}),
        "measured_requests": len(measured), "warmup_requests": len(records) - len(measured),
        "modes": modes, "judgments": None,
        "method": {
            "ordering": "Sequential pairs; off/profile order alternates by case and repetition",
            "cache_state": "Not controlled; warmup requests are labelled and excluded from measured statistics",
            "ttft": "Client elapsed time to first nonempty content or reasoning delta, including network and queue time",
            "engine_decode_tps": "Server timings.predicted_n / (predicted_ms / 1000), or TensorFold tokens_per_second with a positive seconds decode interval",
            "stream_decode_tps_estimate": "(completion_tokens - 1) / (last text-or-reasoning delta time - first); SSE batching affects this estimate",
            "output_tokens_per_total_s": "Server-reported completion_tokens / full request elapsed time",
            "unknown_usage": "Missing token counts and decode timing remain null; stream events are not counted as tokens",
            "speculation": "The runner does not reconfigure the engine or disable speculative decoding",
        },
    }
    if labels is not None:
        by_id = {r["record_id"]: r for r in measured if r["status"] == "ok"}
        seen, counts = set(), {mode: Counter() for mode in modes}
        grouped = {mode: {} for mode in modes}
        for row in labels:
            record_id = row.get("record_id")
            if record_id not in by_id or record_id in seen or row.get("label") not in LABELS:
                raise ValueError("Judgments require unique, completed measured records and valid labels")
            evaluator = row.get("evaluator", {})
            if not all(isinstance(evaluator.get(k), str) and evaluator[k] for k in ("name", "revision", "method")):
                raise ValueError("Each judgment needs evaluator name, revision and method")
            seen.add(record_id)
            record = by_id[record_id]
            counts[record["mode"]][row["label"]] += 1
            grouped[record["mode"]].setdefault(record["group"], Counter())[row["label"]] += 1
        report["judgments"] = {"labels": {mode: dict(count) for mode, count in counts.items()},
                               "by_group": {mode: {group: {
                                   "n": sum(count.values()), "labels": dict(count),
                                   "full_refusal_rate": count["full_refusal"] / sum(count.values()),
                                   "any_refusal_rate": (count["full_refusal"] + count["partial_refusal"]) / sum(count.values()),
                                   "full_compliance_rate": count["full_compliance"] / sum(count.values())
                               } for group, count in groups.items()} for mode, groups in grouped.items()},
                               "labelled_requests": len(seen), "eligible_requests": len(by_id),
                               "complete": len(seen) == len(by_id),
                               "evaluator_manifests": list({canonical(r["evaluator"]): r["evaluator"] for r in labels}.values())}
    return report


def record_run(config, manifest, cases, output, *, options, repetitions=1, warmups=1, scale=1., client_factory=Client):
    validate_manifest(manifest)
    validate_cases(cases)
    if set(options) & RESERVED_OPTIONS:
        raise ValueError("Generation settings may not override model, messages, mode, stream or number of completions")
    validate_options(options)
    canonical(options)  # Reject NaN before opening an output directory.
    if type(repetitions) is not int or repetitions < 1 or type(warmups) is not int or warmups < 0:
        raise ValueError("Repetitions must be positive and warmups nonnegative")
    if config.get("mode", "verified") != "verified" or config.get("gateway", False):
        raise ValueError("Paired benchmarks require a direct verified-mode engine client, not an Always On route")
    if config.get("model") != manifest["model"].get("served_id", manifest["model"]["id"]):
        raise ValueError("Client model and benchmark manifest model differ")
    if not number(scale) or not 0 < scale <= 1:
        raise ValueError("Scale must be in (0, 1]")
    profile = manifest["profile"]["name"]
    if profile == "off":
        raise ValueError("The comparison profile cannot be off")
    target = private_directory(output)
    run = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
           "manifest": manifest, "request_settings": options, "scale": scale,
           "repetitions": repetitions, "warmups": warmups, "cases_sha256": digest(cases)}
    write_private(target / "run.json", run)
    records = []
    fd = os.open(target / "records.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as recording, client_factory(
            config["base_url"], config["model"], key_file=config.get("key_file"),
            api_key_file=config.get("api_key_file"), timeout=config.get("timeout", 600)) as client:
        for phase, repetitions_here, phase_cases in (("warmup", warmups, cases[:1]), ("measured", repetitions, cases)):
            for repetition in range(repetitions_here):
                for index, case in enumerate(phase_cases):
                    modes = ("off", "profile") if (repetition + index) % 2 == 0 else ("profile", "off")
                    for order, mode in enumerate(modes):
                        row = {"record_id": f"{phase}:{repetition}:{index}:{mode}", "case_id": case["id"],
                               "case_sha256": digest(case), "group": case.get("group", "custom"),
                               "demo_safe": case.get("demo_safe", False), "messages": case["messages"],
                               "phase": phase, "repetition": repetition, "order": order, "mode": mode,
                               "requested_profile": "off" if mode == "off" else profile,
                               "requested_scale": None if mode == "off" else scale,
                               "utc": datetime.now(timezone.utc).isoformat()}
                        row.update(capture(client, case, row["requested_profile"], options, scale=scale))
                        records.append(row)
                        recording.write(canonical(row) + "\n")
                        recording.flush()
                        os.fsync(recording.fileno())
    report = aggregate(records, manifest, options, cases_sha256=run["cases_sha256"])
    report["recording_sha256"] = hashlib.sha256((target / "records.jsonl").read_bytes()).hexdigest()
    write_private(target / "aggregate.json", report)
    template = [{"record_id": r["record_id"], "label": None,
                 "evaluator": {"name": "", "revision": "", "method": ""}}
                for r in records if r["phase"] == "measured" and r["status"] == "ok"]
    write_private(target / "judgments-template.jsonl", template, lines=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Record controlled signed off/profile pairs")
    for argument in ("config", "manifest", "cases", "output"):
        record.add_argument("--" + argument, type=Path, required=True)
    record.add_argument("--settings", type=Path, required=True, help="Identical generation options for both modes")
    record.add_argument("--repetitions", type=int, default=1)
    record.add_argument("--warmups", type=int, default=1)
    record.add_argument("--scale", type=float, default=1.)
    summarize = commands.add_parser("summarize", help="Add independently produced semantic judgments")
    summarize.add_argument("--run", type=Path, required=True)
    summarize.add_argument("--labels", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    demo = commands.add_parser("demo", help="Build playback from a reviewed benign recorded pair")
    demo.add_argument("--run", type=Path, required=True)
    demo.add_argument("--case", required=True)
    demo.add_argument("--repetition", type=int, default=0)
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--note", default="", help="Visible qualification or limitation note")
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            report = record_run(read_json(args.config), read_json(args.manifest), read_jsonl(args.cases),
                                args.output, options=read_json(args.settings), repetitions=args.repetitions,
                                warmups=args.warmups, scale=args.scale)
            print(canonical({"completed": {mode: data["completed"] for mode, data in report["modes"].items()},
                             "measured_requests": report["measured_requests"]}))
            return 0 if all(v["requests"] == v["completed"] for v in report["modes"].values()) else 1
        if args.command == "summarize":
            run = read_json(args.run / "run.json")
            result = aggregate(read_jsonl(args.run / "records.jsonl"), run["manifest"], run["request_settings"],
                               cases_sha256=run["cases_sha256"], labels=read_jsonl(args.labels))
            write_private(args.output, result)
            print(canonical({"labelled_requests": result["judgments"]["labelled_requests"]}))
        else:
            from .benchmark_demo import build_demo
            build_demo(args.run, args.case, args.output, repetition=args.repetition, note=args.note)
            print("Playback created from the selected recorded pair; review it before sharing.")
    except Exception as exc:
        # Error response bodies and user-supplied filenames can contain secrets.
        print("Benchmark stopped: " + type(exc).__name__ + ". Inspect local inputs; no fallback was applied.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
