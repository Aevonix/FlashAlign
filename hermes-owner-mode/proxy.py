# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Loopback provider wrapper. Only this boundary can sign a final wire body."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import threading
import time
from urllib.parse import urlsplit

import httpx

from .auth import read_api_key, read_key, sign
from .policy import TICKET_HEADER, CONTEXT_HEADER, canonical, strict_json
from .model_metadata import compatible_request, checked_request_compatibility


logger = logging.getLogger("owner_mode.proxy")
MAX_BODY = 64 * 1024 * 1024
OWNER_FIELDS = ("owner_mode", "owner_mode_scale")
# The engine's owner-related 400s share type/param/code; only `message` tells them apart. Exactly one of them means
# "owner mode is unavailable on this server" (started without a usable key/profiles): that one alone is retried as
# stock. Any other 400 (bad scale, unknown profile, bad type) is a configuration error and is surfaced, never demoted.
# Exact full-string match, pinned to the engine build this was verified against. Any new engine build must re-check
# these strings; a changed message means no retry and a loud surfaced 400 (the safe direction).
UNAVAILABLE_MESSAGE = 'this server was started without owner-mode profiles: owner_mode must be "off"'
OWNER_400_CASES = {
    UNAVAILABLE_MESSAGE: "no_profiles",
    "owner_mode_scale must be a number in (0, 2]": "bad_scale",
    "owner_mode names no profile this server loaded": "unknown_profile",
    'owner_mode must be "off" or an owner-mode profile name': "bad_owner_mode_type",
}
MAX_ERROR_BODY = 64 * 1024


def owner_400_case(raw):
    """Map an engine 400 body to a fixed label (never echoes engine text)."""
    try:
        error = json.loads(raw)["error"]
        if error.get("code") != "invalid_owner_mode":
            return "other"
        return OWNER_400_CASES.get(error.get("message"), "other_owner_mode")
    except Exception:
        return "other"


def route_scale(route):
    """Configured per-route gain, or None. Invalid values disable owner mode on the route (fail closed)."""
    if "owner_mode_scale" not in route:
        return None
    value = route["owner_mode_scale"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1.0:
        raise ValueError("invalid owner_mode_scale")
    return float(value)


def loopback_request(headers):
    """Keep a loopback-only signing service outside foreign browser origins."""
    hosts = headers.get_all('Host', [])
    if len(hosts) != 1:
        return False

    def authority(value, *, origin=False):
        parsed = urlsplit(value if origin else 'http://' + value)
        if (parsed.scheme not in ('http', 'https') or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment):
            raise ValueError('Invalid loopback authority')
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)

    try:
        host = authority(hosts[0])
        origins = headers.get_all('Origin', [])
        if origins and (len(origins) != 1 or authority(origins[0], origin=True) != host):
            return False
    except ValueError:
        return False
    sites = headers.get_all('Sec-Fetch-Site', [])
    return not sites or len(sites) == 1 and sites[0] in ('none', 'same-origin')


class ProviderProxy:
    def __init__(self, policy, routes, key_file, *, port=0, audit=None, bridge=None, approvals=None):
        self.policy, self.routes, self.key_file, self.audit = policy, dict(routes), key_file, audit
        self.bridge = bridge
        self.approvals = approvals if approvals is not None else getattr(bridge, "store", None)
        self.scales, self.disabled = {}, set()
        for name, route in self.routes.items():
            if not name or not name.replace("-", "").isalnum():
                raise ValueError("invalid route")
            url = urlsplit(route["upstream"])
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("invalid upstream")
            checked_request_compatibility(route.get('request_compatibility'))
            try:
                self.scales[name] = route_scale(route)
            except ValueError:
                # Keep serving stock traffic; this route can never sign.
                self.disabled.add(name)
                self.scales[name] = None
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def parse_request(self):
                if not super().parse_request():
                    return False
                if not loopback_request(self.headers):
                    self.send_error(403, 'Loopback gateway origin rejected')
                    return False
                return True

            def log_message(self, *_):
                pass  # Default logging can expose request data.

            def do_GET(self):
                # Read-only model listing passthrough (health checks, metadata probes). No body, no owner mode.
                path = self.path.split("/")
                if len(path) != 4 or path[2:] != ["v1", "models"] or path[1] not in proxy.routes:
                    self.send_error(404)
                    return
                try:
                    headers = {}
                    if self.headers.get("Authorization"):
                        headers["Authorization"] = self.headers["Authorization"]
                    if proxy.routes[path[1]].get('api_key_file'):
                        headers['Authorization'] = 'Bearer ' + read_api_key(proxy.routes[path[1]]['api_key_file'])
                    upstream = proxy.routes[path[1]]["upstream"].rstrip("/") + "/models"
                    with httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=False) as client:
                        response = client.get(upstream, headers=headers)
                    self.send_response(response.status_code)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(response.content)))
                    self.end_headers()
                    self.wfile.write(response.content)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception:
                    self.send_error(502, "provider wrapper unavailable")

            def do_POST(self):
                if self.path.startswith("/_owner/") and proxy.bridge:
                    proxy.bridge.http(self)
                    return
                path = self.path.split("/")
                if len(path) != 5 or path[2:] != ["v1", "chat", "completions"] or path[1] not in proxy.routes:
                    self.send_error(404)
                    return
                route_name = path[1]
                started = time.monotonic()
                info, effective, actual, status = {}, "off", "none", 0
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                        raise ValueError("body size")
                    body = strict_json(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("body type")
                    info = proxy.policy.consume_context(self.headers.get(CONTEXT_HEADER, ""), body, route_name)
                    ticket = self.headers.get(TICKET_HEADER, "")
                    if proxy.bridge:
                        ticket = proxy.bridge.ticket(self.headers.get("X-Owner-Mode-Run", ""), body, route_name)
                    outbound, headers, effective = proxy.prepare(route_name, body, ticket, info=info)
                    # Deliberately never forward caller-supplied owner authentication.
                    authorization = self.headers.get("Authorization")
                    if authorization:
                        headers["Authorization"] = authorization
                    if proxy.routes[route_name].get('api_key_file'):
                        headers['Authorization'] = 'Bearer ' + read_api_key(proxy.routes[route_name]['api_key_file'])
                    # Raw bytes are relayed without Content-Encoding, so ask the engine not to compress.
                    headers["Accept-Encoding"] = "identity"
                    upstream = proxy.routes[route_name]["upstream"].rstrip("/") + "/chat/completions"
                    timeout = httpx.Timeout(proxy.routes[route_name].get("timeout", 120), connect=10)
                    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                        def send(content, hdrs):
                            return client.send(client.build_request("POST", upstream, content=content, headers=hdrs),
                                               stream=True)
                        response = send(outbound, headers)
                        buffered = None
                        if response.status_code == 400:
                            # Small error body: read (bounded) so it can be classified, then pass on or drop it.
                            buffered = b""
                            for chunk in response.iter_raw():
                                buffered += chunk
                                if len(buffered) > MAX_ERROR_BODY:
                                    buffered = buffered[:MAX_ERROR_BODY]
                                    break
                            response.close()
                            case = owner_400_case(buffered)
                            if (effective != "off" and case == "no_profiles"
                                    and info.get("activation_mode") != "always_on"):
                                # Owner mode unavailable on the engine: resend the same request ONCE as stock
                                # (owner fields removed, unsigned). This 400 never reaches the client.
                                logger.warning("OWNER_MODE_UNAVAILABLE route=%s: engine started without owner-mode "
                                               "profiles answered HTTP 400; resent once with owner_mode off", route_name)
                                info = dict(info, decision="owner_mode_unavailable")
                                effective = "off"
                                headers = {k: v for k, v in headers.items() if k != "X-Owner-Mode-Auth"}
                                response, buffered = send(canonical(proxy.stock(body, route_name)), headers), None
                            elif case != "other":
                                # Never retried: a bad scale or profile must not be silently demoted to stock.
                                logger.error("OWNER_MODE_REJECTED route=%s case=%s owner_mode_sent=%s: engine answered "
                                             "HTTP 400; surfaced to the caller, not retried", route_name, case, effective)
                                info = dict(info, decision="owner_mode_rejected_" + case)
                        try:
                            status = response.status_code
                            actual = response.headers.get("X-Owner-Mode-Effective", "none")
                            if (info.get("activation_mode") == "always_on" and 200 <= status < 300
                                    and actual != effective):
                                info["decision"] = "always_on_unconfirmed"
                                status = 502
                                raise ValueError("Always On profile was not confirmed")
                            self.send_response(response.status_code)
                            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                            # This is the engine's decision, never an optimistic client decision.
                            # "none" in logs = the engine sent no header (it does not implement owner mode).
                            self.send_header("X-Owner-Mode-Effective", "off" if actual == "none" else actual)
                            if buffered is not None:
                                self.send_header("Content-Length", str(len(buffered)))
                            self.end_headers()
                            if buffered is not None:
                                self.wfile.write(buffered)
                            else:
                                for chunk in response.iter_raw():
                                    self.wfile.write(chunk)
                                    self.wfile.flush()
                        finally:
                            response.close()
                except (BrokenPipeError, ConnectionResetError):
                    status = status or 499
                except Exception:
                    # Fixed error text; exceptions can contain bodies or credentials.
                    status = status or 502
                    try:
                        self.send_error(502, "provider wrapper unavailable")
                    except Exception:
                        pass
                finally:
                    proxy.record(route_name, effective, actual, info=info, status=status,
                                 ms=int((time.monotonic() - started) * 1000))

        self.server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="owner-mode-provider")

    @property
    def port(self):
        return self.server.server_port

    def base_url(self, route):
        return f"http://127.0.0.1:{self.port}/{route}/v1"

    def start(self):
        self.thread.start()

    def close(self):
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        self.server.server_close()

    def record(self, route, requested, effective, *, info=None, status=0, ms=0):
        if self.audit:
            # No prompt, caller/session identifier, key, token, header value, or
            # caller-controlled reason. Identity is used for routing, not logging.
            row = {"route": route, "platform": "", "owner_verified": False,
                   "activation_mode": "verified",
                   "decision": "no_middleware_context", "identity_ms": None}
            row.update({k: v for k, v in (info or {}).items() if k in row})
            row.update({"owner_mode_sent": requested,
                        "owner_mode_scale_sent": self.scales.get(route) if requested != "off" else None,
                        "effective": effective, "status": status, "ms": ms})
            try:
                self.audit(row)
            except Exception:
                pass

    def stock(self, body, route=None):
        # Absent means off; preserve all remaining request fields.
        result = {k: v for k, v in body.items() if k not in OWNER_FIELDS}
        return compatible_request(result, self.routes[route].get('request_compatibility')) if route else result

    def prepare(self, route, body, ticket="", *, info=None):
        mode = "off"
        headers = {"Content-Type": "application/json"}
        # Read the local operator policy on every request, including requests
        # without harness middleware. No identity proof is manufactured here.
        state = self.approvals.read(require_existing=True) if self.approvals is not None else None
        enabled = state is None or state["enabled"]
        always_on = state is not None and state["mode"] == "always_on"
        if info is not None:
            info["activation_mode"] = "off" if not enabled else "always_on" if always_on else "verified"
            if not enabled:
                info["decision"] = "deployment_disabled"
            elif always_on:
                info["decision"] = "always_on"
        if not enabled:
            return canonical(self.stock(body, route)), headers, "off"
        try:
            if always_on:
                candidate = self.routes[route].get("profile")
            else:
                candidate = self.policy.consume(ticket, body, route) if ticket else "off"
            if (candidate == self.routes[route].get("profile") and candidate != "off"
                    and isinstance(candidate, str) and candidate
                    and route not in self.disabled):
                key = read_key(self.key_file)
                mode = candidate
        except Exception:
            if always_on:
                raise ValueError("Always On requires a usable profile and signing key") from None
            mode = "off"
        if mode == "off":
            if always_on:
                raise ValueError("Always On requires an enabled profile for this route")
            return canonical(self.stock(body, route)), headers, "off"
        final = self.stock(body, route)
        final["owner_mode"] = mode
        # Gain comes ONLY from this route's configuration; any client-supplied value was dropped above.
        if self.scales.get(route) is not None:
            final["owner_mode_scale"] = self.scales[route]
        try:
            raw = canonical(final)
            headers["X-Owner-Mode-Auth"] = sign(key, raw)
        except Exception:
            if always_on:
                raise ValueError("Always On request could not be signed") from None
            return canonical(self.stock(body, route)), {"Content-Type": "application/json"}, "off"
        return raw, headers, mode
