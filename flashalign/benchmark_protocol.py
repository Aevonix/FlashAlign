# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Strict paired collection with immutable attempts and resumable provenance."""
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import secrets
import time

from . import Client
from .benchmark import RESERVED_OPTIONS, metrics, validate_manifest, validate_options
from .benchmark_io import atomic_json, canonical, digest, private_directory, read_json, run_lock, text_digest
from .benchmark_suites import SUITES, load_prepared
from .evaluation_limits import LEGACY_LIMITS, LIMITED_PROTOCOL, UPPER_LIMITS, limits_for_run, validate_limits


PROTOCOL = "flashalign.strict-completion.v1"
RECORD_SCHEMA = "flashalign.evaluation-record.v1"
MAX_ANSWER_BYTES = 4 * 1024 * 1024
IDENTITY_FIELDS = {"effective_profile", "profile_digest", "model_digest", "generation"}


def validate_evaluation_manifest(value):
    """Keep v1 manifests usable while declaring optional runtime identity."""
    if not isinstance(value, dict) or not isinstance(value.get("profile"), dict):
        raise ValueError("Invalid evaluation manifest")
    extra = {key: value["profile"][key] for key in ("content_digest", "format", "allowed_strengths") if key in value["profile"]}
    clean = {**value, "profile": {key: item for key, item in value["profile"].items() if key not in extra}}
    validate_manifest(clean)
    if "content_digest" in extra and not re.fullmatch(r"[0-9a-f]{64}", str(extra["content_digest"])):
        raise ValueError("Invalid runtime profile digest")
    if "format" in extra and (not isinstance(extra["format"], str) or len(extra["format"]) > 128):
        raise ValueError("Invalid profile format")
    allowed = extra.get("allowed_strengths")
    if allowed is not None and (not isinstance(allowed, list) or not allowed or any(
            type(item) not in (int, float) or not math.isfinite(item) or not 0 < item <= 1 for item in allowed)):
        raise ValueError("Invalid declared profile strengths")
    return value


def validate_settings(options):
    if not isinstance(options, dict) or set(options) & (RESERVED_OPTIONS | {"evaluation_limits", *UPPER_LIMITS}):
        raise ValueError("Settings may not override requests, model or profile selection")
    validate_options(options)
    canonical(options)
    budget = options.get("max_tokens")
    if type(budget) is not int or budget <= 0:
        raise ValueError("Strict evaluation requires an explicit positive output-token budget")


def endpoint_fingerprint(url):
    from urllib.parse import urlsplit, urlunsplit
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use an endpoint without embedded credentials, query or fragment")
    port = parsed.port
    authority = parsed.hostname.lower() + (":" + str(port) if port else "")
    return text_digest(urlunsplit((parsed.scheme.lower(), authority, parsed.path.rstrip("/"), "", "")))


def collect(client, case, profile, options, *, scale, expected_digest=None, clock=time.perf_counter, limits=None):
    declared_limits = validate_limits(limits) if limits is not None else None
    budgets = declared_limits or LEGACY_LIMITS
    started = clock()
    events, contents, reasoning, models = [], [], [], set()
    metadata = {}
    status, error = "invalid", None
    finish = None
    saw_finish = False
    response_complete = False
    response_bytes = 0
    try:
        with client.stream_chat_response([{"role": "user", "content": case["prompt"]}],
                                         profile=profile, scale=scale, **options) as stream:
            if not isinstance(stream.metadata, Mapping) or stream.metadata.get("effective_profile") != profile:
                raise ValueError("EffectiveProfileMismatch")
            metadata = {key: stream.metadata[key] for key in IDENTITY_FIELDS if key in stream.metadata}
            if expected_digest is not None and metadata.get("profile_digest") != expected_digest:
                raise ValueError("ProfileDigestMismatch")
            for item in stream:
                if not isinstance(item, dict):
                    raise ValueError("MalformedEvent")
                item_bytes = len(canonical(item).encode())
                if declared_limits is not None and item_bytes > declared_limits["max_sse_event_bytes"]:
                    raise ValueError("ResponseEventLimit")
                response_bytes += item_bytes
                if response_bytes > budgets["max_response_bytes"] or len(events) >= budgets["max_events"]:
                    raise ValueError("ResponseWorkLimit")
                model = item.get("model")
                if model is not None:
                    if not isinstance(model, str) or model != client.model:
                        raise ValueError("ResponseModelMismatch")
                    models.add(model)
                choices = item.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("MultipleOrMalformedChoices")
                event = {"at_s": clock() - started}
                if choices:
                    choice = choices[0]
                    if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                        raise ValueError("MalformedChoice")
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("function_call"):
                        raise ValueError("NonTextAnswer")
                    content = delta.get("content")
                    thought = delta.get("reasoning_content", delta.get("reasoning"))
                    if (content is not None and not isinstance(content, str)) or (thought is not None and not isinstance(thought, str)):
                        raise ValueError("NonTextDelta")
                    if delta.get("reasoning_content") is not None and delta.get("reasoning") is not None and delta["reasoning_content"] != delta["reasoning"]:
                        raise ValueError("ConflictingReasoningFields")
                    if saw_finish and (content or thought):
                        raise ValueError("ContentAfterFinish")
                    if content:
                        contents.append(content)
                        event["content"] = content
                    if thought:
                        reasoning.append(thought)
                        event["reasoning"] = thought
                    if choice.get("finish_reason") is not None:
                        if saw_finish:
                            raise ValueError("DuplicateFinish")
                        finish = choice["finish_reason"]
                        saw_finish = True
                        event["finish_reason"] = finish
                for field in ("usage", "timings", "tensorfold", "speculative"):
                    if isinstance(item.get(field), dict):
                        event[field] = {key: value for key, value in item[field].items()
                                        if type(value) in (int, float) and math.isfinite(value) and value >= 0}
                usage = item.get("usage")
                details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
                cached = details.get("cached_tokens") if isinstance(details, dict) else None
                if type(cached) is int and cached >= 0:
                    event["cached_prompt_tokens"] = cached
                events.append(event)
            response_complete = stream.complete is True
        answer = "".join(contents)
        if not response_complete:
            error = "IncompleteStream"
        elif not models:
            error = "MissingResponseModel"
        elif finish != "stop":
            error = "NonterminalFinish"
        elif not answer.strip():
            error = "MissingFinalAnswer"
        elif answer.lstrip().startswith(("<think>", "<analysis>", "</think>")):
            error = "UnseparatedReasoning"
        else:
            status = "valid"
    except Exception as exc:
        # Messages and response bodies can contain prompts or credentials.
        error = str(exc) if type(exc) is ValueError and re.fullmatch(r"[A-Za-z]{1,64}", str(exc)) else type(exc).__name__
    elapsed = clock() - started
    answer = "".join(contents)
    result = {"status": status, "error": error, "answer": answer, "reasoning": "".join(reasoning),
            "answer_sha256": text_digest(answer), "events": events, "metadata": metadata,
            "transport_complete": response_complete, "response_models": sorted(models), "metrics": metrics(events, elapsed)}
    if declared_limits is not None:
        result["collection_work"] = {"response_bytes": response_bytes, "events": len(events)}
    return result


def records_for(suite, rows, profile):
    """Ordering is fixed in provenance and shared by fresh and resumed runs."""
    records = []
    for index, row in enumerate(rows):
        modes = ("off", "profile") if index % 2 == 0 else ("profile", "off")
        for mode in modes:
            records.append({"record_id": row["id"] + ":" + mode, "case_id": row["id"],
                            "case_index": index, "dataset": row["dataset"], "label": row["label"],
                            "mode": mode, "requested_profile": "off" if mode == "off" else profile,
                            "prompt_sha256": text_digest(row["prompt"]), "prompt": row["prompt"],
                            "corpus_id": suite.corpus_id})
    return records


def _attempts(root, ordinal, expected, fingerprint, run_id, model, *, limits=None, limited=False):
    directory = root / "attempts" / f"{ordinal:06d}"
    if not directory.exists():
        return []
    private_directory(directory)
    files = list(directory.iterdir())
    if any(not re.fullmatch(r"[0-9]{3}\.(begin|end)\.json", item.name) for item in files):
        raise ValueError("Unexpected attempt artifact")
    starts = sorted(directory.glob("*.begin.json"))
    if len(files) - len(starts) > len(starts):
        raise ValueError("Orphan attempt result")
    output = []
    for index, start in enumerate(starts, 1):
        if start.name != f"{index:03d}.begin.json":
            raise ValueError("Noncontiguous attempt ledger")
        begin = read_json(start)
        if begin.get("fingerprint") != fingerprint or begin.get("run_id") != run_id or begin.get("record") != expected or begin.get("attempt") != index:
            raise ValueError("Attempt provenance mismatch")
        end = directory / f"{index:03d}.end.json"
        if not end.exists():
            output.append({"status": "interrupted", "attempt": index})
            continue
        record = read_json(end, limit=(limits or LEGACY_LIMITS)["max_record_bytes"])
        checksum = record.get("record_sha256")
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if checksum != digest(body) or any(record.get(key) != value for key, value in expected.items()) or record.get("fingerprint") != fingerprint or record.get("run_id") != run_id or record.get("attempt") != index:
            raise ValueError("Collection record binding mismatch")
        if record.get("answer_sha256") != text_digest(record.get("answer", "")):
            raise ValueError("Collection answer binding mismatch")
        if record.get("schema") != RECORD_SCHEMA or record.get("status") not in {"valid", "invalid"}:
            raise ValueError("Invalid collection record")
        if limited:
            work = record.get("collection_work")
            if (not isinstance(work, dict) or set(work) != {"response_bytes", "events"}
                    or any(type(value) is not int or value < 0 for value in work.values())
                    or work["events"] != len(record.get("events", []))
                    or work["events"] > limits["max_events"]
                    or record["status"] == "valid" and work["response_bytes"] > limits["max_response_bytes"]):
                raise ValueError("Invalid collection resource accounting")
        if (record.get("answer") != "".join(event.get("content", "") for event in record.get("events", [])) or
                record.get("reasoning") != "".join(event.get("reasoning", "") for event in record.get("events", []))):
            raise ValueError("Collection text differs from its event trace")
        if record["status"] == "valid" and (not record.get("answer", "").strip() or
                record.get("transport_complete") is not True or record.get("metrics", {}).get("finish_reason") != "stop" or
                record.get("response_models") != [model] or
                record.get("metadata", {}).get("effective_profile") != expected["requested_profile"]):
            raise ValueError("Invalid completed collection record")
        output.append(record)
    if {item.name for item in files if item.name.endswith(".end.json")} - {start.name.replace(".begin.", ".end.") for start in starts}:
        raise ValueError("Orphan attempt result")
    return output


def read_collection(root, *, registry=SUITES):
    root = private_directory(root)
    run = read_json(root / "run.json")
    if run.get("schema") != "flashalign.evaluation-run.v1":
        raise ValueError("Unsupported evaluation run")
    budgets = limits_for_run(run)
    descriptor = run.get("descriptor")
    if not isinstance(descriptor, dict) or run.get("fingerprint") != digest(descriptor):
        raise ValueError("Run fingerprint mismatch")
    if not isinstance(run.get("run_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", run["run_id"]):
        raise ValueError("Missing run identity")
    validate_evaluation_manifest(descriptor["manifest"])
    validate_settings(descriptor["settings"])
    suite, cases, prepared = load_prepared(root / "corpus", registry=registry)
    if descriptor.get("suite") != suite.identity() or descriptor.get("cases_sha256") != prepared["cases_sha256"]:
        raise ValueError("Run corpus mismatch")
    expected = records_for(suite, cases, descriptor["manifest"]["profile"]["name"])
    attempts_root = root / "attempts"
    if any(not re.fullmatch(r"[0-9]{6}", item.name) or int(item.name) >= len(expected) for item in attempts_root.iterdir()):
        raise ValueError("Unexpected collection record")
    rows = []
    for index, template in enumerate(expected):
        manifest = descriptor["manifest"]
        model = manifest["model"].get("served_id", manifest["model"]["id"])
        attempts = _attempts(root, index, template, run["fingerprint"], run["run_id"], model,
                             limits=budgets, limited=run["protocol"] == LIMITED_PROTOCOL)
        valid = [item for item in attempts if item.get("status") == "valid"]
        if len(valid) > 1 or (valid and valid[0] is not attempts[-1]):
            raise ValueError("Completed answers cannot be retried")
        rows.append({"expected": template, "attempts": attempts, "record": valid[0] if valid else (attempts[-1] if attempts else None)})
    completed = [row["record"] for row in rows if row["record"] and row["record"].get("status") == "valid"]
    for field in ("profile_digest", "generation"):
        active = [row for row in completed if row["mode"] == "profile"]
        values = [row["metadata"].get(field) for row in active]
        if len(set(values)) > 1:
            raise ValueError("Profile identity changed during evaluation")
    if len({row["metadata"]["model_digest"] for row in completed if row["metadata"].get("model_digest") is not None}) > 1:
        raise ValueError("Model identity changed during evaluation")
    return run, suite, rows


def collect_run(prepared_path, config, manifest, settings, output, *, scale=1., attempts=1,
                resume=False, registry=SUITES, client_factory=Client, limits=None):
    from .benchmark_io import atomic_bytes, read_bytes
    validate_evaluation_manifest(manifest)
    validate_settings(settings)
    declared_limits = validate_limits(limits) if limits is not None else None
    protocol = LIMITED_PROTOCOL if declared_limits is not None else PROTOCOL
    if config.get("mode", "verified") != "verified" or config.get("gateway", False):
        raise ValueError("Paired collection requires an isolated direct dual-mode endpoint")
    if config.get("model") != manifest["model"].get("served_id", manifest["model"]["id"]):
        raise ValueError("Requested model differs from the manifest")
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("Declare one to three collection attempts before starting")
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Invalid requested profile strength")
    declared_strengths = manifest["profile"].get("allowed_strengths")
    if declared_strengths is not None and scale not in declared_strengths:
        raise ValueError("Requested strength is outside the artifact declaration")
    if manifest["profile"].get("format") == "flashalign.native_wo_b.v1" and scale != 1:
        raise ValueError("Native matrix profiles require strength one")
    suite, cases, prepared = load_prepared(prepared_path, registry=registry)
    descriptor = {"suite": suite.identity(), "cases_sha256": prepared["cases_sha256"], "manifest": manifest,
                  "settings": settings, "scale": scale, "maximum_attempts": attempts,
                  "subject_endpoint_sha256": endpoint_fingerprint(config["base_url"]),
                  "order": "alternating-pairs-source-order-v1", "protocol": protocol}
    if declared_limits is not None:
        descriptor["evaluation_limits"] = declared_limits
    root = private_directory(output, create=not resume)
    with run_lock(root):
        if resume:
            old = read_json(root / "run.json")
            if old.get("descriptor") != descriptor or old.get("fingerprint") != digest(descriptor):
                raise ValueError("Resume configuration differs from the frozen run")
        else:
            corpus = private_directory(root / "corpus", create=True)
            for name in ("suite.json", "cases.json", "source.jsonl"):
                atomic_bytes(corpus / name, read_bytes(Path(prepared_path) / name), replace=False)
            private_directory(root / "attempts", create=True)
            atomic_json(root / "run.json", {"schema": "flashalign.evaluation-run.v1", "protocol": protocol,
                                            "run_id": secrets.token_hex(16), "descriptor": descriptor,
                                            "fingerprint": digest(descriptor),
                                            **({"evaluation_limits": declared_limits} if declared_limits is not None else {})}, replace=False)
        run, _, entries = read_collection(root, registry=registry)
        client_options = {"max_sse_event_bytes": declared_limits["max_sse_event_bytes"]} if declared_limits is not None else {}
        with client_factory(config["base_url"], config["model"], key_file=config.get("key_file"),
                            api_key_file=config.get("api_key_file"), timeout=config.get("timeout", 600), **client_options) as client:
            for ordinal, entry in enumerate(entries):
                previous = entry["attempts"]
                if entry["record"] and entry["record"].get("status") == "valid":
                    continue
                if entry["record"] and entry["record"].get("status") != "interrupted" and entry["record"].get("error") not in {
                        "ConnectError", "ReadError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}:
                    continue
                folder = root / "attempts" / f"{ordinal:06d}"
                if not folder.exists():
                    private_directory(folder, create=True)
                for attempt in range(len(previous) + 1, attempts + 1):
                    expected = entry["expected"]
                    begin = {"fingerprint": run["fingerprint"], "run_id": run["run_id"], "record": expected, "attempt": attempt,
                             "started_utc": datetime.now(timezone.utc).isoformat()}
                    atomic_json(folder / f"{attempt:03d}.begin.json", begin, replace=False)
                    result = collect(client, expected, expected["requested_profile"], settings, scale=scale,
                                     expected_digest=manifest["profile"].get("content_digest") if expected["mode"] == "profile" else None,
                                     limits=declared_limits)
                    record = {"schema": RECORD_SCHEMA, "run_id": run["run_id"], "fingerprint": run["fingerprint"],
                              **expected, "attempt": attempt, "scale": scale if expected["mode"] == "profile" else None,
                              **result}
                    record["record_sha256"] = digest(record)
                    if declared_limits is not None and len((canonical(record) + "\n").encode()) > declared_limits["max_record_bytes"]:
                        raise ValueError("Collection record exceeds its declared resource budget")
                    atomic_json(folder / f"{attempt:03d}.end.json", record, replace=False)
                    if result["status"] == "valid":
                        break
                    # Retry transport failures only; malformed/finished outputs
                    # are outcomes, not opportunities to select a nicer answer.
                    if result["error"] not in {"ConnectError", "ReadError", "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}:
                        break
        return collection_summary(root, registry=registry)


def collection_summary(root, *, registry=SUITES):
    from collections import Counter
    run, suite, entries = read_collection(root, registry=registry)
    modes = {}
    for mode in ("off", "profile"):
        selected = [entry for entry in entries if entry["expected"]["mode"] == mode]
        valid = [entry for entry in selected if entry["record"] and entry["record"].get("status") == "valid"]
        failures = Counter(entry["record"].get("error", entry["record"].get("status")) if entry["record"] else "not_collected"
                           for entry in selected if entry not in valid)
        modes[mode] = {"assigned": suite.row_count, "valid": len(valid), "invalid_or_missing": len(selected)-len(valid),
                       "failure_counts": dict(failures), "attempts": sum(len(entry["attempts"]) for entry in selected)}
    return {"schema": "flashalign.collection-summary.v1", "protocol": run["protocol"], "suite": suite.name,
            "collection_valid": all(value["valid"] == suite.row_count for value in modes.values()),
            "modes": modes, "run_fingerprint": run["fingerprint"],
            **({"evaluation_limits": run["evaluation_limits"]} if run["protocol"] == LIMITED_PROTOCOL else {})}


def collect_calibration_sweep(prepared, config, manifest, settings, output, *, strengths,
                              registry=SUITES, client_factory=Client):
    """Private workbench hook; held-out evaluation corpora cannot select strength."""
    suite, _, _ = load_prepared(prepared, registry=registry)
    if suite.purpose != "calibration":
        raise ValueError("Strength calibration requires a separate calibration corpus")
    declared = manifest.get("profile", {}).get("allowed_strengths")
    if not isinstance(strengths, list) or not strengths or not isinstance(declared, list):
        raise ValueError("Calibration needs explicit artifact-supported candidate strengths")
    if len(strengths) > 32 or any(type(value) not in (int, float) or not math.isfinite(value) for value in strengths):
        raise ValueError("Invalid calibration strength list")
    if len(set(strengths)) != len(strengths) or any(value not in declared for value in strengths):
        raise ValueError("Calibration strengths differ from the artifact capability")
    validate_evaluation_manifest(manifest)
    validate_settings(settings)
    root = private_directory(output, create=True)
    runs = []
    for index, strength in enumerate(strengths):
        summary = collect_run(prepared, config, manifest, settings, root/f"candidate-{index:03d}",
                              scale=strength, registry=registry, client_factory=client_factory)
        runs.append({"index": index, "strength": strength, "collection": summary})
    result = {"schema": "flashalign.calibration-collection.v1", "suite": suite.identity(), "runs": runs,
              "selection": None, "requires": "Independent quality scoring and untouched final evaluation"}
    atomic_json(root/"sweep.json", result, replace=False)
    return result
