# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import asyncio
import copy
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

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
from flashalign.benchmark import aggregate, capture, metrics, private_directory, record_run, read_jsonl
from flashalign.benchmark_demo import build_demo
from flashalign.benchmark_lm_eval import SignedEvaluationMixin


MANIFEST = {"model": {"id": "model", "revision": "pinned", "quantization": "BF16"},
            "engine": {"name": "synthetic", "version": "test"},
            "hardware": {"accelerator": "none"}, "speculation": {"enabled": True, "method": "unchanged"},
            "profile": {"name": "profile", "artifact_sha256": "0" * 64}}
CASE = {"id": "safe", "group": "safe", "demo_safe": True,
        "messages": [{"role": "user", "content": "A benign synthetic fixture."}]}


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp" if Path("/var/tmp").is_dir() else None)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = self.root / "key"
        self.key.write_bytes(b"x" * 32)
        self.key.chmod(0o600)
        self.requests = []

    def reply(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        profile = payload["owner_mode"]
        if profile != "off":
            parts = request.headers["X-Owner-Mode-Auth"].split(":")
            expected = hmac.new(b"x" * 32, (parts[1] + "\n" + parts[2] + "\n" + hashlib.sha256(request.content).hexdigest()).encode(), hashlib.sha256).hexdigest()
            self.assertEqual(parts[3], expected)
        else:
            self.assertNotIn("X-Owner-Mode-Auth", request.headers)
        events = [{"choices": [{"delta": {"content": "Safe "}}]},
                  {"choices": [{"delta": {"content": "answer."}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 7, "completion_tokens": 3}}]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"X-Owner-Mode-Effective": profile}, text=body)

    def factory(self, *args, **kwargs):
        return Client(*args, **kwargs, transport=httpx.MockTransport(self.reply))

    def run_fixture(self, **kwargs):
        return record_run({"base_url": "http://localhost/v1", "model": "model", "key_file": self.key},
                          MANIFEST, [CASE], self.root / "run", options={"temperature": 0, "max_tokens": 12},
                          client_factory=self.factory, **kwargs)

    def test_same_generation_settings_and_signed_effective_profile(self):
        report = self.run_fixture(repetitions=2, warmups=1)
        self.assertEqual([x["owner_mode"] for x in self.requests], ["off", "profile", "off", "profile", "profile", "off"])
        clean = [{k: v for k, v in p.items() if k not in ("owner_mode", "owner_mode_scale")} for p in self.requests]
        self.assertTrue(all(p == clean[0] for p in clean))
        self.assertEqual(report["measured_requests"], 4)
        self.assertEqual(report["warmup_requests"], 2)
        self.assertEqual(report["modes"]["off"]["metrics"]["completion_tokens"]["mean"], 3)
        self.assertEqual((self.root / "run/records.jsonl").stat().st_mode & 0o777, 0o600)
        self.assertNotIn("Safe answer", json.dumps(report))
        self.assertNotIn(str(self.key), json.dumps(report))
        self.assertNotIn("base_url", json.dumps(report))

    def test_no_usage_does_not_count_chunks_as_tokens(self):
        result = metrics([{"at_s": .1, "content": "many tokens in one chunk"}, {"at_s": .2, "content": "more"}], .3)
        self.assertEqual(result["ttft_s"], .1)
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["engine_decode_tps"])
        self.assertIsNone(result["stream_decode_tps_estimate"])

    def test_serving_alias_keeps_the_checkpoint_identity_in_provenance(self):
        manifest = copy.deepcopy(MANIFEST)
        manifest["model"].update(id="organization/checkpoint", served_id="model")
        report = record_run({"base_url": "http://localhost/v1", "model": "model", "key_file": self.key},
                            manifest, [CASE], self.root / "alias-run", options={}, warmups=0,
                            client_factory=self.factory)
        self.assertEqual(report["manifest"]["model"]["id"], "organization/checkpoint")
        self.assertEqual(self.requests[0]["model"], "model")

    def test_reasoning_ttft_usage_and_server_decode_are_distinct(self):
        result = metrics([{"at_s": .2, "reasoning": "private"}, {"at_s": 1.2, "content": "answer"},
                          {"at_s": 1.3, "usage": {"completion_tokens": 21},
                           "timings": {"predicted_n": 20, "predicted_ms": 500}}], 1.5)
        self.assertEqual(result["ttft_s"], .2)
        self.assertEqual(result["first_answer_text_s"], 1.2)
        self.assertEqual(result["engine_decode_tps"], 40)
        self.assertEqual(result["output_tokens_per_total_s"], 14)
        self.assertEqual(result["stream_decode_tps_estimate"], 20)

    def test_bad_effective_mode_and_incomplete_stream_are_failures(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, headers={"X-Owner-Mode-Effective": "off"}, text="data: {}\n\n"))
        with Client("http://localhost/v1", "model", key_file=self.key, transport=transport) as client:
            result = capture(client, CASE, "profile", {})
            self.assertEqual(result["error_type"], "ProfileNotActive")
            self.assertFalse(result["events"])
            result = capture(client, CASE, "off", {})
            self.assertEqual(result["status"], "incomplete")

    def test_cache_usage_keeps_only_the_reported_nonnegative_count(self):
        def reply(_):
            event = {"choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}],
                     "usage": {"completion_tokens": 1, "prompt_tokens_details": {
                         "cached_tokens": 1024, "private_field": "not recorded"}}}
            return httpx.Response(200, headers={"X-Owner-Mode-Effective": "off"},
                                  text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")
        with Client("http://localhost/v1", "model", transport=httpx.MockTransport(reply)) as client:
            result = capture(client, CASE, "off", {})
        self.assertEqual(result["metrics"]["cached_prompt_tokens"], 1024)
        self.assertNotIn("not recorded", json.dumps(result))
        self.assertIsNone(metrics([{"at_s": .1, "cached_prompt_tokens": -1}], .2)["cached_prompt_tokens"])

    def test_bundled_context_fixture_is_original_data_with_a_late_question(self):
        rows = read_jsonl(ROOT / "benchmarks/long-context.jsonl")
        self.assertEqual(len(rows), 1)
        text = rows[0]["messages"][0]["content"]
        self.assertGreater(len(text), 8000)
        self.assertTrue(text.endswith("What is Cedar Station's weekly calibration day? Answer in one sentence."))
        self.assertEqual(rows[0]["group"], "safe")

    def test_requested_native_token_trace_stays_in_private_records(self):
        def reply(_):
            event = {"choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}],
                     "tensorfold": {"token_ids": [0, 9], "token_sha256": "a" * 64, "secret": "discard"}}
            return httpx.Response(200, headers={"X-Owner-Mode-Effective": "off"}, text="data: " + json.dumps(event) + "\n\n")
        with Client("http://localhost/v1", "model", transport=httpx.MockTransport(reply)) as client:
            result = capture(client, CASE, "off", {"return_token_ids": True})
        self.assertEqual(result["events"][0]["token_ids"], [0, 9])
        self.assertEqual(result["events"][0]["token_sha256"], "a" * 64)
        self.assertNotIn("discard", json.dumps(result))

    def test_native_decode_telemetry_is_distinct_from_client_stream_estimate(self):
        result = metrics([{"at_s": .2, "content": "first"}, {"at_s": .6, "content": "last"},
                          {"at_s": .7, "usage": {"completion_tokens": 21},
                           "tensorfold": {"tokens_per_second": 50, "seconds": .4},
                           "speculative": {"rounds": 5, "drafted": 25, "accepted": 18, "extra": "private"}}], .8)
        self.assertEqual(result["engine_decode_tps"], 50)
        self.assertEqual(result["engine_decode_seconds"], .4)
        self.assertEqual(result["speculative"], {"rounds": 5, "drafted": 25, "accepted": 18})
        self.assertIn("tensorfold", result["engine_decode_tps_source"])
        invalid = metrics([{"at_s": .1, "tensorfold": {"tokens_per_second": 999, "seconds": 0}}], .2)
        self.assertIsNone(invalid["engine_decode_tps"])

    def test_no_always_on_route_or_credential_metadata(self):
        with self.assertRaises(ValueError):
            record_run({"mode": "always_on"}, MANIFEST, [CASE], self.root / "invalid", options={})
        with self.assertRaises(ValueError):
            record_run({}, MANIFEST, [CASE], self.root / "invalid", options={"user": "private-contact"})
        self.assertFalse((self.root / "invalid").exists())
        (self.root / ".git").write_text("gitdir: private")
        with self.assertRaises(ValueError):
            private_directory(self.root / "inside-repo")

    def test_semantic_labels_require_provenance_and_no_duplicate_denominator(self):
        self.run_fixture(warmups=0)
        rows = read_jsonl(self.root / "run/records.jsonl")
        labels = [{"record_id": r["record_id"], "label": "full_compliance",
                   "evaluator": {"name": "human", "revision": "rubric-v1", "method": "blind semantic review"}} for r in rows]
        report = aggregate(rows, MANIFEST, {}, cases_sha256="test", labels=labels)
        self.assertTrue(report["judgments"]["complete"])
        self.assertEqual(report["judgments"]["labelled_requests"], 2)
        with self.assertRaises(ValueError):
            aggregate(rows, MANIFEST, {}, cases_sha256="test", labels=labels + labels[:1])

    def test_html_uses_real_pair_and_escapes_model_output(self):
        self.run_fixture(warmups=0)
        path = self.root / "run/records.jsonl"
        rows = read_jsonl(path)
        rows[0]["events"][0]["content"] = "</script><script>alert('x')</script>"
        rows[0]["events"][0]["reasoning"] = "PRIVATE REASONING"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        payload = build_demo(self.root / "run", "safe", self.root / "demo.html", note="Development recording: one selected fixture.")
        html = (self.root / "demo.html").read_text()
        self.assertNotIn("</script><script>alert", html)
        self.assertNotIn("PRIVATE REASONING", html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(payload["note"], "Development recording: one selected fixture.")
        rows[0]["demo_safe"] = False
        path.write_text("\n".join(json.dumps(r) for r in rows))
        with self.assertRaises(ValueError):
            build_demo(self.root / "run", "safe", self.root / "bad.html")


class FakeAPI:
    def __init__(self, **kwargs):
        self.base_url = kwargs["base_url"]
        self._seed = 7
        self.eos_string = "end"
        self.options = kwargs
    def create_message(self, messages):
        return messages
    def _create_payload(self, messages, **kwargs):
        return {"messages": messages, **kwargs["gen_kwargs"]}
    def parse_generations(self, outputs):
        return [outputs["choices"][0]["message"]["content"]]


class EvaluationAdapterTests(unittest.TestCase):
    def test_sync_and_async_adapter_sign_and_preserve_generation_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            key = Path(tmp) / "key"
            key.write_bytes(b"x" * 32)
            key.chmod(0o600)
            calls = []
            def reply(request):
                body = json.loads(request.content)
                calls.append(body)
                self.assertEqual(body["owner_mode"], "profile")
                self.assertIn("X-Owner-Mode-Auth", request.headers)
                return httpx.Response(200, headers={"X-Owner-Mode-Effective": "profile"},
                                      json={"choices": [{"message": {"content": "4"}}]})
            cls = type("Adapter", (SignedEvaluationMixin, FakeAPI), {})
            with Client("http://localhost/v1", "model", key_file=key, transport=httpx.MockTransport(reply)) as client:
                adapter = cls(flashalign_client=client, flashalign_profile="profile",
                              base_url="http://localhost/v1/chat/completions")
                gen = {"temperature": 0, "max_tokens": 10}
                adapter.model_call(CASE["messages"], gen_kwargs=gen)
                self.assertEqual(gen, {"temperature": 0, "max_tokens": 10})
                async def run():
                    return await adapter.amodel_call(None, asyncio.Semaphore(1), CASE["messages"], gen_kwargs=gen)
                self.assertEqual(asyncio.run(run()), ["4"])
                self.assertEqual(adapter.options["num_concurrent"], 1)
                self.assertEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
