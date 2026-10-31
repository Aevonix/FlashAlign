# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Synthetic engines and test-only loading for command-line entrypoints."""
from collections import OrderedDict
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time


def load_cli(name):
    """Load a standalone CLI without replacing the suite's package object.

    The entrypoints establish their own ``owner_mode`` alias when run directly.
    Repeating that bootstrap inside a test process must preserve the existing
    package and the usual parent/child bindings for already imported modules.
    Python 3.10's mock.patch resolves dotted targets through those bindings.
    """
    if name not in {"setup", "manage"}:
        raise ValueError("Unknown test CLI")
    path = Path(__file__).resolve().parents[1] / (name + ".py")
    spec = importlib.util.spec_from_file_location("flashalign_" + name + "_test", path)
    module = importlib.util.module_from_spec(spec)
    package = sys.modules.get("owner_mode")
    try:
        spec.loader.exec_module(module)
    finally:
        if package is not None:
            sys.modules["owner_mode"] = package
        for fullname, child in list(sys.modules.items()):
            if fullname.startswith("owner_mode."):
                parent_name, _, attribute = fullname.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and child is not None:
                    setattr(parent, attribute, child)
    return module


OWNER_400_MESSAGES = {
    "a": 'this server was started without owner-mode profiles: owner_mode must be "off"',
    "b": "owner_mode_scale must be a number in (0, 2]",
    "c": "owner_mode names no profile this server loaded",
    "d": 'owner_mode must be "off" or an owner-mode profile name',
}


class Engine:
    def __init__(self, key, profiles=("owner-test",)):
        self.key, self.profiles = key, set(profiles)
        self.records, self.nonces = [], OrderedDict()
        self.lock = threading.Lock()
        self.scripts = {}
        self.fail_models = {}
        self.required_authorization = None
        self.refuse_owner_mode = False  # head started with an unusable key: 400 on ANY non-off owner_mode
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                payload = json.dumps({"object": "list", "data": [{"id": "bench", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.loads(raw)
                if engine.required_authorization and self.headers.get("Authorization") != engine.required_authorization:
                    self.send_response(401)
                    self.end_headers()
                    return
                effective, reason = engine.verify(raw, body, self.headers.get("X-Owner-Mode-Auth", ""))
                with engine.lock:
                    engine.records.append({"body": body, "effective": effective, "auth_reason": reason,
                                           "body_sha256": hashlib.sha256(raw).hexdigest()})
                case = engine.owner_400(body, effective)
                if case:
                    # Preserve the engine contract's exact owner-mode error shape and messages.
                    payload = {"error": {"message": OWNER_400_MESSAGES[case], "type": "invalid_request_error",
                                         "param": "owner_mode", "code": "invalid_owner_mode"}}
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-Owner-Mode-Effective", "off")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode())
                    return
                if body.get("model") in engine.fail_models:
                    self.send_response(engine.fail_models[body['model']])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("X-Owner-Mode-Effective", effective)
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"model unavailable","type":"invalid_request_error","code":"model_not_found"}}')
                    return
                self.send_response(200)
                self.send_header("X-Owner-Mode-Effective", effective)
                with engine.lock:
                    script = engine.scripts.get(body.get("model"), [])
                    action = script.pop(0) if script else None
                msg = action or {"role": "assistant", "content": "Bench response. Effective=" + effective}
                common = {"id": "chatcmpl-bench", "model": body.get("model", "bench"), "created": int(time.time())}
                finish = "tool_calls" if msg.get("tool_calls") else "stop"
                if body.get("stream"):
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    delta = dict(msg)
                    if delta.get("tool_calls"):
                        delta["tool_calls"] = [dict(t, index=i) for i, t in enumerate(delta["tool_calls"])]
                    chunks = [dict(common, object="chat.completion.chunk", choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
                              dict(common, object="chat.completion.chunk", choices=[{"index": 0, "delta": {}, "finish_reason": finish}],
                                   usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})]
                    for chunk in chunks:
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(dict(common, object="chat.completion", choices=[{
                        "index": 0, "message": msg, "finish_reason": finish}],
                        usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def owner_400(self, body, effective):
        """Checks in the live engine's order d, b, a, c."""
        mode = body.get("owner_mode", "off")
        if not isinstance(mode, str):
            return "d"
        if "owner_mode_scale" in body:
            scale = body["owner_mode_scale"]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) \
                    or not 0 < scale <= 2.0:
                return "b"
        if mode != "off" and self.refuse_owner_mode:
            return "a"
        if effective != "off" and effective not in self.profiles:
            return "c"
        return None

    def verify(self, raw, body, header):
        mode = body.get("owner_mode", "off")
        if mode == "off":
            return "off", "off"
        try:
            version, ts, nonce, mac = header.split(":")
            if version != "v1" or abs(time.time() - int(ts)) > 60 or not nonce:
                return "off", "expired_or_format"
            expected = hmac.new(self.key, (ts + "\n" + nonce + "\n" + hashlib.sha256(raw).hexdigest()).encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, mac):
                return "off", "invalid"
            with self.lock:
                now = time.time()
                for used, expiry in list(self.nonces.items()):
                    if expiry < now:
                        del self.nonces[used]
                if nonce in self.nonces:
                    return "off", "replay"
                if len(self.nonces) >= 4096:
                    return "off", "nonce_capacity"
                self.nonces[nonce] = int(ts) + 60
            return mode, "valid"
        except Exception:
            return "off", "missing_or_format"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
