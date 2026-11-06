# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Offline end-to-end checks; synthetic inputs and HTTP transports only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "owner_mode" not in sys.modules:
    spec = importlib.util.spec_from_file_location("owner_mode", ROOT / "hermes-owner-mode/__init__.py",
                                                 submodule_search_locations=[str(ROOT / "hermes-owner-mode")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

from flashalign import Client
from flashalign.benchmark_io import atomic_json, canonical, digest, private_directory, read_json, run_lock
from flashalign.benchmark_judge import FOURWAY, JUDGE_PARAMETERS, RUBRIC, Scorer, ScorerRegistry, judge_run, score_run
from flashalign.benchmark_protocol import collect_calibration_sweep, collect_run, collection_summary, read_collection
from flashalign.benchmark_reports import compare_reports, evidence_card, performance_summary, public_report
from flashalign.benchmark_suites import FR443, Suite, SuiteRegistry, load_prepared, prepare_suite


MANIFEST = {"model": {"id": "org/model", "served_id": "model", "revision": "fixed", "quantization": "BF16"},
            "engine": {"name": "fixture", "version": "1"}, "hardware": {"accelerator": "synthetic"},
            "speculation": {"enabled": True, "method": "preserved"},
            "profile": {"name": "profile", "artifact_sha256": "a"*64}}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root.chmod(0o700)
        self.key = self.root / "key"
        self.key.write_bytes(b"k"*32)
        self.key.chmod(0o600)
        self.config = {"base_url": "http://subject.test/v1", "model": "model", "key_file": self.key}
        self.judge_config = {"base_url": "http://judge.test/v1", "model": "judge", "revision": "fixed", "independent_deployment": True}
        self.cases = [{"id": "f-1", "dataset": "fixture", "label": "safe", "prompt": "A synthetic harmless prompt."},
                      {"id": "f-2", "dataset": "fixture", "label": "unsafe", "prompt": "Another synthetic harmless prompt."}]
        self.raw = "".join(canonical(row)+"\n" for row in self.cases).encode()
        self.source = self.root / "source.jsonl"
        self.source.write_bytes(self.raw)
        self.suite = Suite("fixture", "fixture-v1", "fixed", hashlib.sha256(self.raw).hexdigest(), 2,
                           (("fixture", 2),), (("safe", 1), ("unsafe", 1)), 2, FOURWAY.name, "https://example.test")
        self.registry = SuiteRegistry((self.suite,))
        self.prepared = prepare_suite("fixture", self.source, self.root / "prepared", registry=self.registry)
        self.options = {"max_tokens": 128, "temperature": 0, "seed": 42}
        self.requests = []
        self.judge_requests = []
        self.fault = None
        self.judge_verdict = None
        self.judge_fault = None

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        if self.fault == "interrupt":
            self.fault = None
            raise KeyboardInterrupt()
        if self.fault == "transport":
            self.fault = None
            raise httpx.ConnectError("synthetic", request=request)
        mode = payload["owner_mode"]
        model = "other" if self.fault == "model" else "model"
        finish = self.fault if self.fault in {"length", "tool_calls", "content_filter"} else "stop"
        content = "" if self.fault in {"empty", "reasoning"} else mode+": synthetic final answer"
        if self.fault == "unseparated":
            content = "<think>synthetic analysis</think>synthetic final"
        delta = {"content": content}
        if self.fault == "reasoning":
            delta["reasoning_content"] = "synthetic reasoning only"
        events = [{"model": model, "choices": [{"index": 0, "delta": delta}]},
                  {"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                   "usage": {"completion_tokens": 5, "prompt_tokens": 6}}]
        text = "".join("data: "+json.dumps(event)+"\n\n" for event in events)
        if self.fault != "missing_done":
            text += "data: [DONE]\n\n"
        headers = {"X-Owner-Mode-Effective": "wrong" if self.fault == "mode" else mode}
        return httpx.Response(200, headers=headers, text=text)

    def client(self, *args, **kwargs):
        return Client(*args, **kwargs, transport=httpx.MockTransport(self.respond))

    def run_subject(self, name="run", **kwargs):
        return collect_run(self.prepared, self.config, copy.deepcopy(MANIFEST), self.options,
                           self.root/name, registry=self.registry, client_factory=self.client, **kwargs)

    def judge_reply(self, request):
        payload = json.loads(request.content)
        self.judge_requests.append(payload)
        if self.judge_fault == "transport":
            raise httpx.ConnectError("synthetic", request=request)
        evidence = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        verdict = self.judge_verdict or ("true_refusal" if evidence["candidate_response"].startswith("off:") else "not_refusal")
        content = json.dumps({"verdict": verdict, "reason": "Synthetic fixture judgment."})
        if self.judge_fault == "malformed":
            content = '{"verdict":"not_refusal","verdict":"true_refusal","reason":"Duplicate."}'
        return httpx.Response(200, json={"model": "judge", "choices": [{"finish_reason": "stop", "message": {"content": content}}]})

    def judge_client(self, **kwargs):
        return httpx.Client(**kwargs, transport=httpx.MockTransport(self.judge_reply))

    def score_fixture(self, *, resume=False):
        return judge_run(self.root/"run", self.judge_config, self.root/"judged", resume=resume,
                         registry=self.registry, client_factory=self.judge_client)

    def test_trusted_registration_and_frozen_fr443_contract(self):
        self.assertEqual(FR443.row_count, 443)
        self.assertEqual(dict(FR443.labels), {"harmful": 295, "safe": 75, "unsafe": 73})
        self.assertEqual(FR443.unique_prompts, 441)
        self.assertEqual(self.registry.names(), ("fixture",))
        with self.assertRaises(ValueError):
            self.registry.register(self.suite)
        with self.assertRaises(ValueError):
            self.registry.get("not-installed")
        scorer = Scorer("new-scorer", (("yes", "not_refuse"), ("no", "refuse")), ("unknown",))
        self.assertEqual(ScorerRegistry((scorer,)).get("new-scorer").map("yes"), "not_refuse")

    def test_source_and_prepared_tampering_rejected(self):
        self.source.write_bytes(self.raw+b"\n")
        with self.assertRaises(ValueError):
            prepare_suite("fixture", self.source, self.root/"bad", registry=self.registry)
        self.assertFalse((self.root/"bad").exists())
        rows = read_json(self.prepared/"cases.json")
        rows[0]["prompt"] = "Changed"
        atomic_json(self.prepared/"cases.json", rows)
        meta = read_json(self.prepared/"suite.json")
        meta["cases_sha256"] = digest(rows)
        atomic_json(self.prepared/"suite.json", meta)
        with self.assertRaises(ValueError):
            load_prepared(self.prepared, registry=self.registry)

    def test_pairing_signing_and_identical_generation_options(self):
        result = self.run_subject()
        self.assertTrue(result["collection_valid"])
        self.assertEqual([r["owner_mode"] for r in self.requests], ["off", "profile", "profile", "off"])
        for request in self.requests:
            self.assertEqual({key: request[key] for key in self.options}, self.options)
        run, _, entries = read_collection(self.root/"run", registry=self.registry)
        self.assertTrue(all(e["record"]["transport_complete"] for e in entries))
        self.assertTrue(all(e["record"]["metadata"]["effective_profile"] == e["expected"]["requested_profile"] for e in entries))
        self.assertEqual((self.root/"run/run.json").stat().st_mode & 0o777, 0o600)
        self.assertNotIn("key_file", canonical(run))
        self.assertNotIn("base_url", canonical(run))

    def test_strict_completion_failures(self):
        for fault in ("length", "tool_calls", "content_filter", "empty", "reasoning", "missing_done", "model", "mode", "unseparated"):
            with self.subTest(fault=fault):
                self.fault = fault
                result = self.run_subject(name=fault)
                self.assertFalse(result["collection_valid"])
                self.assertEqual(result["modes"]["off"]["valid"], 0)
                self.assertEqual(result["modes"]["profile"]["valid"], 0)

    def test_resume_never_reissues_valid_or_semantic_failure(self):
        self.run_subject(attempts=2)
        total = len(self.requests)
        self.run_subject(attempts=2, resume=True)
        self.assertEqual(len(self.requests), total)
        self.fault = "length"
        self.run_subject(name="bad-run", attempts=2)
        total = len(self.requests)
        self.fault = None
        self.run_subject(name="bad-run", attempts=2, resume=True)
        self.assertEqual(len(self.requests), total)

    def test_retry_policy_and_interrupted_attempt_receipts(self):
        self.fault = "transport"
        result = self.run_subject(attempts=2)
        self.assertTrue(result["collection_valid"])
        self.assertEqual(len(self.requests), 5)
        self.fault = "interrupt"
        with self.assertRaises(KeyboardInterrupt):
            self.run_subject(name="interrupted", attempts=2)
        result = self.run_subject(name="interrupted", attempts=2, resume=True)
        self.assertTrue(result["collection_valid"])
        _, _, entries = read_collection(self.root/"interrupted", registry=self.registry)
        self.assertEqual(entries[0]["attempts"][0]["status"], "interrupted")
        self.assertEqual(entries[0]["record"]["attempt"], 2)

    def test_changed_resume_settings_rejected_before_request(self):
        self.run_subject()
        total = len(self.requests)
        self.options["max_tokens"] = 129
        with self.assertRaises(ValueError):
            self.run_subject(resume=True)
        self.assertEqual(len(self.requests), total)

    def test_record_and_judgment_tamper_rejected(self):
        self.run_subject()
        self.score_fixture()
        receipt = self.root/"judged/judgments/000000.json"
        item = read_json(receipt)
        item["answer_sha256"] = "b"*64
        item["receipt_sha256"] = digest({k:v for k,v in item.items() if k != "receipt_sha256"})
        atomic_json(receipt, item)
        with self.assertRaises(ValueError):
            score_run(self.root/"run", self.root/"judged", registry=self.registry)
        record = self.root/"run/attempts/000000/001.end.json"
        item = read_json(record)
        item["answer"] = "Tampered"
        atomic_json(record, item)
        with self.assertRaises(ValueError):
            read_collection(self.root/"run", registry=self.registry)

    def test_identical_configuration_runs_still_have_distinct_receipts(self):
        self.run_subject()
        self.run_subject(name="other")
        for name in ("001.begin.json", "001.end.json"):
            shutil.copyfile(self.root/"run/attempts/000000"/name, self.root/"other/attempts/000000"/name)
        with self.assertRaises(ValueError):
            read_collection(self.root/"other", registry=self.registry)

    def test_raw_judge_verdict_cannot_be_relabelled(self):
        self.run_subject()
        self.score_fixture()
        path = self.root/"judged/judgments/000000.json"
        item = read_json(path)
        item["verdict"], item["mapped_verdict"] = "not_refusal", "not_refuse"
        item["receipt_sha256"] = digest({key: value for key, value in item.items() if key != "receipt_sha256"})
        atomic_json(path, item)
        with self.assertRaises(ValueError):
            score_run(self.root/"run", self.root/"judged", registry=self.registry)

    def test_exact_fourway_mapping_and_primary_score(self):
        self.assertEqual(FOURWAY.map("true_refusal"), "refuse")
        self.assertEqual(FOURWAY.map("not_refusal"), "not_refuse")
        self.assertEqual(FOURWAY.map("capability_limitation"), "not_refuse")
        self.assertIsNone(FOURWAY.map("uncertain"))
        self.run_subject()
        score = self.score_fixture()
        self.assertTrue(score["final_behavioral_gate_valid"])
        self.assertEqual(score["modes"]["off"]["refusal_rate"], 1)
        self.assertEqual(score["modes"]["profile"]["refusal_rate"], 0)
        self.assertEqual(score["paired_refusal_change"]["profile_minus_off"], -1)
        for request in self.judge_requests:
            for key, value in JUDGE_PARAMETERS.items():
                self.assertEqual(request[key], value)
        self.assertEqual(RUBRIC["sha256"], hashlib.sha256(self.judge_requests[0]["messages"][0]["content"].encode()).hexdigest())

    def test_registered_scorer_runs_without_central_dispatch_changes(self):
        scorer = Scorer("other-refusal-scorer", (("refused", "refuse"), ("answered", "not_refuse")),
                        ("unknown",), {"id": "test-rubric", "sha256": "f"*64}, "Synthetic rubric", JUDGE_PARAMETERS)
        suite = Suite("other", "other-v1", "fixed", hashlib.sha256(self.raw).hexdigest(), 2,
                      self.suite.datasets, self.suite.labels, 2, scorer.name, "https://example.test")
        self.registry = SuiteRegistry((suite,))
        self.prepared = prepare_suite("other", self.source, self.root/"prepared-other", registry=self.registry)
        self.run_subject()
        self.judge_verdict = "answered"
        report = judge_run(self.root/"run", self.judge_config, self.root/"judged-other", registry=self.registry,
                           scorers=ScorerRegistry((scorer,)), client_factory=self.judge_client)
        self.assertTrue(report["final_behavioral_gate_valid"])
        self.assertEqual(report["modes"]["profile"]["refusal_rate"], 0)

    def test_uncertain_stays_invalid_and_not_rejudged(self):
        self.run_subject()
        self.judge_verdict = "uncertain"
        score = self.score_fixture()
        self.assertFalse(score["final_behavioral_gate_valid"])
        self.assertIsNone(score["modes"]["off"]["refusal_rate"])
        total = len(self.judge_requests)
        self.judge_verdict = "not_refusal"
        again = self.score_fixture(resume=True)
        self.assertFalse(again["final_behavioral_gate_valid"])
        self.assertEqual(len(self.judge_requests), total)

    def test_judge_transport_attempts_bounded_and_malformed_answer_preserved(self):
        self.run_subject()
        self.judge_fault = "transport"
        report = self.score_fixture()
        self.assertFalse(report["final_behavioral_gate_valid"])
        self.assertEqual(len(self.judge_requests), 12)
        self.score_fixture(resume=True)
        self.assertEqual(len(self.judge_requests), 12)
        self.judge_fault = "malformed"
        report = judge_run(self.root/"run", self.judge_config, self.root/"bad-judge", registry=self.registry,
                           client_factory=self.judge_client)
        self.assertFalse(report["final_behavioral_gate_valid"])
        raw = read_json(self.root/"bad-judge/judgments/000000.json")
        self.assertIn('"Duplicate."', raw["raw_judge_answer"])
        self.assertEqual(len(raw["attempts"]), 1)

    def test_capability_limitations_are_not_task_success(self):
        self.run_subject()
        self.judge_verdict = "capability_limitation"
        report = self.score_fixture()
        self.assertEqual(report["modes"]["profile"]["capability_limitations"], 2)
        self.assertEqual(report["modes"]["profile"]["refusal_rate"], 0)
        self.assertNotIn("success_rate", canonical(report))

    def test_judging_requires_complete_collection_and_separate_deployment(self):
        self.fault = "length"
        self.run_subject()
        with self.assertRaises(ValueError):
            self.score_fixture()
        self.fault = None
        self.run_subject(name="complete")
        config = {**self.judge_config, "base_url": self.config["base_url"]}
        with self.assertRaises(ValueError):
            judge_run(self.root/"complete", config, self.root/"same", registry=self.registry, client_factory=self.judge_client)
        self.assertFalse((self.root/"same").exists())

    def test_public_export_has_no_raw_text_or_deployment_data(self):
        self.run_subject()
        report = self.score_fixture()
        performance = performance_summary(self.root/"run", registry=self.registry)
        public = public_report(report, performance)
        self.assertEqual(public['judge_protocol'], report['judge_protocol'])
        text = canonical(public)
        for forbidden in ("synthetic final answer", "A synthetic harmless prompt", "subject.test", "judge.test", str(self.key), "served_id", "endpoint_sha256"):
            self.assertNotIn(forbidden, text)
        card = evidence_card(report, performance)
        self.assertEqual(card['evaluation']['judge_protocol'], report['judge_protocol'])
        self.assertEqual(card["capability_assessment"]["status"], "not_assessed")
        self.assertEqual(card["evidence_sha256"], digest(public))
        report["manifest"]["hardware"]["cpu"] = "/private/deployment"
        with self.assertRaises(ValueError):
            public_report(report)

    def test_partial_judgments_have_fixed_denominators(self):
        self.run_subject()
        self.score_fixture()
        (self.root/"judged/judgments/000000.json").unlink()
        score = score_run(self.root/"run", self.root/"judged", registry=self.registry)
        self.assertFalse(score["final_behavioral_gate_valid"])
        self.assertEqual(score["modes"]["off"]["assigned"], 2)
        self.assertEqual(score["modes"]["off"]["ungraded_or_invalid"], 1)
        self.assertIsNone(score["modes"]["profile"]["refusal_rate"])

    def test_reports_compare_only_matched_settings_without_selecting_winner(self):
        self.run_subject()
        report = self.score_fixture()
        compared = compare_reports([report, copy.deepcopy(report)])
        self.assertIsNone(compared["selection"])
        other = copy.deepcopy(report)
        other["settings"]["max_tokens"] += 1
        with self.assertRaises(ValueError):
            compare_reports([report, other])
        other = copy.deepcopy(report)
        other['judge_protocol'] = 'flashalign.fourway-judge.v1'
        with self.assertRaises(ValueError):
            compare_reports([report, other])

    def test_private_output_and_concurrent_run_lock(self):
        (self.root/".git").write_text("fixture")
        with self.assertRaises(ValueError):
            private_directory(self.root/"not-private", create=True)
        (self.root/".git").unlink()
        with run_lock(self.root):
            with self.assertRaises(ValueError):
                with run_lock(self.root):
                    pass

    def test_strength_sweep_requires_disjoint_calibration_data_and_capabilities(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["profile"]["allowed_strengths"] = [.5, 1.]
        with self.assertRaises(ValueError):
            collect_calibration_sweep(self.prepared, self.config, manifest, self.options,
                                      self.root/"forbidden-sweep", strengths=[.5, 1.], registry=self.registry)
        suite = Suite("calibration", "calibration-v1", "fixed", hashlib.sha256(self.raw).hexdigest(), 2,
                      self.suite.datasets, self.suite.labels, 2, FOURWAY.name, "https://example.test", purpose="calibration")
        registry = SuiteRegistry((suite,))
        prepared = prepare_suite("calibration", self.source, self.root/"calibration", registry=registry)
        result = collect_calibration_sweep(prepared, self.config, manifest, self.options, self.root/"sweep",
                                          strengths=[.5, 1.], registry=registry, client_factory=self.client)
        self.assertEqual([row["strength"] for row in result["runs"]], [.5, 1.])
        self.assertIsNone(result["selection"])
        self.assertTrue(all(row["collection"]["collection_valid"] for row in result["runs"]))

    def test_complete_443_pair_preserves_source_duplicates_and_subsets(self):
        cases = []
        for index in range(1, 444):
            dataset, label = ("advbench", "harmful") if index <= 146 else (("strongreject", "harmful") if index <= 295 else ("xstest", "safe" if index <= 370 else "unsafe"))
            cases.append({"id": f"r1fr443-{index:04d}", "dataset": dataset, "label": label,
                          "prompt": "Synthetic fixture " + str(index if index <= 441 else index-441)})
        raw = "".join(canonical(row)+"\n" for row in cases).encode()
        self.source.write_bytes(raw)
        suite = Suite("fixture443", "synthetic-443", "test", hashlib.sha256(raw).hexdigest(), 443,
                      FR443.datasets, FR443.labels, 441, FOURWAY.name, "https://example.test", FR443.validator)
        self.registry = SuiteRegistry((suite,))
        self.prepared = prepare_suite("fixture443", self.source, self.root/"prepared443", registry=self.registry)
        self.run_subject()
        report = self.score_fixture()
        self.assertEqual(len(self.requests), 886)
        self.assertEqual(len(self.judge_requests), 886)
        self.assertTrue(report["final_behavioral_gate_valid"])
        self.assertEqual(report["modes"]["off"]["assigned"], 443)
        self.assertEqual(report["modes"]["profile"]["by_label"]["safe"]["assigned"], 75)
        self.assertEqual(report["paired_refusal_change"]["clusters"], 441)


if __name__ == "__main__":
    unittest.main()
