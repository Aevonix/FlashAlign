# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Allowlisted aggregate reports and model/profile evidence cards."""
from collections import Counter
import ipaddress
import math
import re
import statistics

from .benchmark_io import digest
from .benchmark_protocol import read_collection
from .evaluation_limits import validate_limits


METRICS = ("ttft_s", "first_answer_text_s", "total_s", "completion_tokens", "prompt_tokens",
           "cached_prompt_tokens", "engine_decode_tps", "output_tokens_per_total_s", "stream_decode_tps_estimate")


def performance_summary(root, *, registry=None):
    args = {} if registry is None else {"registry": registry}
    _, _, entries = read_collection(root, **args)
    result = {}
    for mode in ("off", "profile"):
        rows = [entry["record"] for entry in entries if entry["expected"]["mode"] == mode]
        valid = [row for row in rows if row and row["status"] == "valid"]
        values = {}
        for metric in METRICS:
            measured = sorted(row["metrics"][metric] for row in valid
                              if type(row["metrics"].get(metric)) in (int, float) and math.isfinite(row["metrics"][metric]))
            values[metric] = {"n": len(measured), "mean": statistics.mean(measured) if measured else None,
                              "median": statistics.median(measured) if measured else None,
                              "p95": measured[max(0, math.ceil(.95*len(measured))-1)] if measured else None}
        result[mode] = {"assigned": len(rows), "completed": len(valid), "metrics": values,
                        "finish_reasons": dict(Counter(row.get("metrics", {}).get("finish_reason") or "missing" for row in rows if row))}
    return result


def _public_scalar(value):
    if type(value) in (int, float, bool) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite public value")
        return value
    if not isinstance(value, str) or len(value) > 512 or any(ord(ch) < 32 for ch in value):
        raise ValueError("Invalid public metadata value")
    if value.startswith(("/", "~", "\\")) or "\\" in value or "://" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Public metadata must not contain deployment paths or URLs")
    if any(re.fullmatch(pattern, value, re.I) for pattern in (r"sk-[A-Za-z0-9_-]{12,}", r"gh[pousr]_[A-Za-z0-9]{12,}")):
        raise ValueError("Credential-like public metadata")
    for candidate in re.findall(r"[0-9]+(?:\.[0-9]+){3}", value):
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            pass
        else:
            raise ValueError("Public metadata must not identify a deployment address")
    return value


def public_manifest(manifest):
    fields = {"model": ("id", "revision", "weights_sha256", "quantization", "tokenizer_revision"),
              "profile": ("name", "artifact_sha256", "content_digest", "format"),
              "engine": ("name", "version", "commit", "settings_sha256", "build_sha256"),
              "hardware": ("accelerator", "accelerator_count", "memory_gib", "power_limit_watts", "cpu"),
              "speculation": ("enabled", "method", "draft_model", "draft_revision", "draft_tokens")}
    return {section: {key: _public_scalar(values[key]) for key in allowed if key in values}
            for section, allowed in fields.items() if isinstance(values := manifest.get(section), dict)}


def public_settings(settings):
    numeric = {"temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "max_completion_tokens",
               "frequency_penalty", "presence_penalty", "repetition_penalty"}
    result = {key: value for key, value in settings.items() if key in numeric and type(value) in (int, float)}
    if settings.get("reasoning_effort") in {"none", "low", "medium", "high", "max"}:
        result["reasoning_effort"] = settings["reasoning_effort"]
    for name, allowed in (("chat_template_kwargs", {"thinking", "enable_thinking", "thinking_budget", "preserve_thinking"}),
                          ("stream_options", {"include_usage"})):
        value = settings.get(name)
        if isinstance(value, dict):
            result[name] = {key: item for key, item in value.items() if key in allowed and type(item) in (bool, int, float)}
    return result


def public_report(report, performance=None):
    """Deliberate export: no raw rows, endpoints, receipt IDs or judge reasons."""
    if report.get("schema") != "flashalign.refusal-report.v1":
        raise ValueError("Unsupported public report")
    suite = report["suite"]
    judge = report["judge"]
    value = {"schema": report["schema"], "protocol": _public_scalar(report["protocol"]),
             "judge_protocol": _public_scalar(report.get("judge_protocol", "flashalign.fourway-judge.v1")),
             "suite": {key: suite[key] for key in ("name", "corpus_id", "revision", "source_sha256", "row_count", "datasets", "labels", "unique_prompts", "scorer", "purpose")},
             "rubric": report["rubric"], "judge": {key: _public_scalar(judge[key]) for key in ("model", "revision")},
             "judge_parameters": judge["parameters"], "manifest": public_manifest(report["manifest"]),
             "settings": public_settings(report["settings"]), "settings_sha256": digest(report["settings"]),
             "settings_complete": public_settings(report["settings"]) == report["settings"], "profile_strength": report["profile_strength"],
             "final_behavioral_gate_valid": report["final_behavioral_gate_valid"], "modes": report["modes"],
             "paired_refusal_change": report["paired_refusal_change"],
             "interpretation": "Judge-rated refusal behavior, not accuracy or successful task completion."}
    if performance is not None:
        value["performance"] = performance
        value["performance_scope"] = "Sequential paired requests; not maximum server throughput. Missing metrics remain null."
    if "evaluation_limits" in report:
        value["evaluation_limits"] = validate_limits(report["evaluation_limits"])
    # The report shape above is built from internal aggregation, not arbitrary
    # records. Ensure future extensions cannot accidentally include text fields.
    forbidden = {"prompt", "answer", "reasoning", "messages", "headers", "key_file", "api_key_file", "base_url", "run_id", "record_id", "served_id"}
    def walk(item):
        if isinstance(item, dict):
            if forbidden & set(item):
                raise ValueError("Raw or deployment data in public report")
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)
    return value


def evidence_card(report, performance=None):
    clean = public_report(report, performance)
    return {"schema": "flashalign.profile-evidence.v1", "profile": clean["manifest"]["profile"],
            "model": clean["manifest"]["model"], "engine": clean["manifest"]["engine"],
            "hardware": clean["manifest"]["hardware"], "speculation": clean["manifest"]["speculation"],
            "evaluation": {key: clean[key] for key in ("suite", "protocol", "judge_protocol", "rubric", "judge", "judge_parameters", "settings_sha256", "profile_strength", "final_behavioral_gate_valid", "modes", "paired_refusal_change", "evaluation_limits") if key in clean},
            "performance": performance, "capability_assessment": {"status": "not_assessed", "results": None},
            "evidence_sha256": digest(clean)}


def compare_reports(reports):
    """Compare measured candidates without claiming a calibration winner."""
    if not reports:
        raise ValueError("At least one report is required")
    reference = reports[0]
    identities = (reference["suite"], reference["protocol"], reference["rubric"], reference["judge"], reference["settings"],
                  reference.get("judge_protocol", "flashalign.fourway-judge.v1"), reference.get("evaluation_limits"))
    for report in reports:
        if (report["suite"], report["protocol"], report["rubric"], report["judge"], report["settings"],
                report.get("judge_protocol", "flashalign.fourway-judge.v1"), report.get("evaluation_limits")) != identities:
            raise ValueError("Report corpus, protocol, judge and generation settings must match")
        for section in ("model", "engine", "hardware", "speculation"):
            if report["manifest"][section] != reference["manifest"][section]:
                raise ValueError("Candidate reports require the same model and serving configuration")
    return {"schema": "flashalign.profile-comparison.v1", "candidates": [
        {"profile": public_manifest(report["manifest"])["profile"], "strength": report["profile_strength"],
         "complete": report["final_behavioral_gate_valid"], "modes": report["modes"],
         "paired_refusal_change": report["paired_refusal_change"]} for report in reports],
        "selection": None, "reason": "Evaluation results are not calibration data; no automatic winner is selected."}
