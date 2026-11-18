# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Loopback provider wrapper. Only this boundary can sign a final wire body."""
from http.cookiejar import CookieJar, DefaultCookiePolicy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
import time
from urllib.parse import urlsplit

import httpx

from .auth import read_api_key, read_key, sign
from .policy import TICKET_HEADER, CONTEXT_HEADER, canonical, strict_json, safe_identity_reason
from .model_metadata import compatible_request, checked_request_compatibility
from .profile_sets import (applied, digest_check, label, route_profiles, scales_sent, selected,
                           verified_label, wire)


logger = logging.getLogger("owner_mode.proxy")
MAX_BODY = 64 * 1024 * 1024
OWNER_FIELDS = ("owner_mode", "owner_mode_scale")
# The engine's owner-related 400s share type/param/code; only `message` tells them apart. Exactly one of them means
# "owner mode is unavailable on this server" (started without a usable key/profiles): that one is retried as stock.
# An unknown profile follows the profile's fail rule: optional profiles are dropped, a required one is surfaced.
# Any other 400 (bad scale, bad type) is a configuration error and is surfaced, never demoted.
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
MAX_DISCOVERY_BODY = 1024 * 1024


def owner_400_case(raw):
    """Map an engine 400 body to a fixed label (never echoes engine text)."""
    try:
        error = json.loads(raw)["error"]
        if error.get("code") != "invalid_owner_mode":
            return "other"
        return OWNER_400_CASES.get(error.get("message"), "other_owner_mode")
    except Exception:
        return "other"


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


class _NoCookies(DefaultCookiePolicy):
    """Connection reuse must not carry one caller's cookies to another."""
    def set_ok(self, cookie, request):
        return False


class ProviderProxy:
    def __init__(self, policy, routes, key_file, *, port=0, audit=None, bridge=None, approvals=None):
        self.policy, self.routes, self.key_file, self.audit = policy, dict(routes), key_file, audit
        self.bridge = bridge
        self.approvals = approvals if approvals is not None else getattr(bridge, "store", None)
        self.disabled = set()
        for name, route in self.routes.items():
            if not name or not name.replace("-", "").isalnum():
                raise ValueError("invalid route")
            url = urlsplit(route["upstream"])
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("invalid upstream")
            checked_request_compatibility(route.get('request_compatibility'))
            try:
                route_profiles(route)
            except ValueError:
                # Keep serving stock traffic; this route can never sign.
                self.disabled.add(name)
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
                # Read-only discovery and status share the configured model route.
                path = self.path.split("/")
                allowed = {('v1', 'models'), ('v1', 'owner-mode', 'profiles'),
                           ('v1', 'flashalign', 'status'), ('v1', 'flashalign', 'capabilities')}
                if len(path) < 4 or tuple(path[2:]) not in allowed or path[1] not in proxy.routes:
                    self.send_error(404)
                    return
                try:
                    headers = {}
                    if self.headers.get("Authorization"):
                        headers["Authorization"] = self.headers["Authorization"]
                    if proxy.routes[path[1]].get('api_key_file'):
                        headers['Authorization'] = 'Bearer ' + read_api_key(proxy.routes[path[1]]['api_key_file'])
                    upstream = proxy.routes[path[1]]["upstream"].rstrip("/") + '/' + '/'.join(path[3:])
                    with proxy.client.stream('GET', upstream, headers=headers, timeout=httpx.Timeout(10, connect=5)) as response:
                        chunks, size = [], 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 1024 * 1024:
                                raise ValueError('Oversized discovery response')
                            chunks.append(chunk)
                        content = b''.join(chunks)
                        status = response.status_code
                        content_type = response.headers.get('Content-Type', 'application/json')
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
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
                info, effective, actual, status, identity = {}, "off", "none", 0, {}
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
                        ticket = proxy.bridge.ticket(self.headers.get("X-Owner-Mode-Run", ""), body, route_name, info=info)
                    outbound, headers, effective = proxy.prepare(route_name, body, ticket, info=info)
                    declared = proxy.declared(route_name)
                    always_on = info.get("activation_mode") == "always_on"
                    sent = selected(declared, effective)
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
                    def send(content, hdrs):
                        return proxy.client.send(proxy.client.build_request("POST", upstream, content=content,
                                                                             headers=hdrs, timeout=timeout),
                                                 stream=True)
                    response = send(outbound, headers)
                    try:
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
                            must = always_on or any(rule.fail == "required" for rule in sent)
                            reduced = False
                            if case == "unknown_profile" and len(sent) > 1 and not always_on:
                                # One profile of the set is not loaded. Keep the profiles the engine lists, and
                                # every required one, and resend. A required profile the engine lacks is
                                # rejected again and surfaced below. If the engine rejects a set of optional
                                # profiles again (a profile left between the two calls, or its list is wrong),
                                # the stock resend below still serves the request: at most two resends.
                                loaded = proxy.loaded_profiles(route_name, headers)
                                kept = tuple(rule for rule in sent if rule.fail == "required"
                                             or (loaded is not None and rule.name in loaded))
                                if kept != sent:
                                    logger.error("OWNER_MODE_PROFILE_DROPPED route=%s dropped=%s kept=%s: the engine "
                                                 "does not have every profile of the set; resent once without the "
                                                 "optional profiles it lacks", route_name,
                                                 label(tuple(rule for rule in sent if rule not in kept)), label(kept))
                                    info = dict(info, decision="owner_mode_profile_dropped")
                                    outbound, headers, effective = proxy.signed(route_name, body, kept, headers)
                                    sent, reduced = kept, True
                                    response = send(outbound, headers)
                                    buffered, case = None, "other"
                                    if response.status_code == 400:
                                        buffered = b""
                                        for chunk in response.iter_raw():
                                            buffered += chunk
                                            if len(buffered) > MAX_ERROR_BODY:
                                                buffered = buffered[:MAX_ERROR_BODY]
                                                break
                                        response.close()
                                        case = owner_400_case(buffered)
                            if (effective != "off" and not must
                                    and case in ("no_profiles", "unknown_profile")):
                                # The engine takes none of the optional profiles sent: resend the same request
                                # as stock (owner fields removed, unsigned). This 400 never reaches the client.
                                if case == "no_profiles":
                                    logger.warning("OWNER_MODE_UNAVAILABLE route=%s: engine started without owner-mode "
                                                   "profiles answered HTTP 400; resent once with owner_mode off",
                                                   route_name)
                                    info = dict(info, decision="owner_mode_unavailable")
                                else:
                                    logger.error("OWNER_MODE_PROFILE_DROPPED route=%s dropped=%s kept=off: the engine "
                                                 "%s; resent once with owner_mode off", route_name, effective,
                                                 "rejected the profiles it listed" if reduced
                                                 else "does not have the optional profile")
                                    info = dict(info, decision="owner_mode_profile_dropped")
                                effective, sent = "off", ()
                                headers = {k: v for k, v in headers.items() if k != "X-Owner-Mode-Auth"}
                                response, buffered = send(canonical(proxy.stock(body, route_name)), headers), None
                            elif case != "other":
                                # Never retried: a bad scale, a bad type or a missing required profile must not
                                # be silently demoted to stock.
                                logger.error("OWNER_MODE_REJECTED route=%s case=%s owner_mode_sent=%s: engine answered "
                                             "HTTP 400; surfaced to the caller, not retried%s", route_name, case,
                                             effective, " (this engine accepts one profile per request)"
                                             if case == "bad_owner_mode_type" and len(sent) > 1 else "")
                                info = dict(info, decision="owner_mode_rejected_" + case)
                        status = response.status_code
                        actual = response.headers.get("X-Owner-Mode-Effective", "none")
                        if 200 <= status < 300 and actual != effective:
                            confirmed = applied(sent, actual)
                            missing = tuple(rule for rule in sent if rule not in confirmed)
                            if always_on:
                                info["decision"] = "always_on_unconfirmed"
                                status = 502
                                raise ValueError("Always On profile was not confirmed")
                            if any(rule.fail == "required" for rule in missing):
                                info["decision"] = "required_profile_unconfirmed"
                                status = 502
                                raise ValueError("Required profile was not confirmed")
                            if missing:
                                logger.warning("OWNER_MODE_UNCONFIRMED route=%s sent=%s effective=%s: the engine did "
                                               "not confirm every optional profile", route_name, effective,
                                               "off" if actual == "none" else label(confirmed))
                        from .client import response_metadata
                        identity = response_metadata(response, actual)
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        # This is the engine's decision, never an optimistic client decision.
                        # "none" in logs = the engine sent no header (it does not implement owner mode).
                        self.send_header("X-Owner-Mode-Effective", "off" if actual == "none" else actual)
                        for name, field in (('X-FlashAlign-Profile-Digest', 'profile_digest'),
                                            ('X-FlashAlign-Model-Digest', 'model_digest'),
                                            ('X-FlashAlign-Generation', 'generation')):
                            if field in identity:
                                self.send_header(name, identity[field])
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
                except Exception as error:
                    # Fixed error text; exceptions can contain bodies or credentials.
                    status = status or 502
                    extension = getattr(error, "extension", None)
                    if extension:
                        # An installed adapter failed while this request needed its registry. Only the
                        # names the loader recorded are logged, never the failure's own text.
                        logger.warning("OWNER_MODE_EXTENSION route=%s: %s could not be loaded or registered; "
                                       "answered HTTP 502", route_name, extension)
                    try:
                        self.send_error(502, "provider wrapper unavailable")
                    except Exception:
                        pass
                finally:
                    proxy.record(route_name, effective, actual, info=info, status=status,
                                 ms=int((time.monotonic() - started) * 1000),
                                 digest=identity.get("profile_digest"))

        self.server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
        self.server.daemon_threads = True
        self._client_guard = threading.Lock()
        self._client = self._client_context = None
        self._client_closed = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="owner-mode-provider")

    @property
    def client(self):
        with self._client_guard:
            if self._client_closed:
                raise RuntimeError("provider wrapper is closed")
            if self._client is None:
                context = httpx.Client(follow_redirects=False, cookies=CookieJar(policy=_NoCookies()),
                                       limits=httpx.Limits(max_connections=None, max_keepalive_connections=20))
                try:
                    self._client = context.__enter__()
                except BaseException:
                    context.close()
                    raise
                self._client_context = context
            return self._client

    @property
    def port(self):
        return self.server.server_port

    def base_url(self, route):
        return f"http://127.0.0.1:{self.port}/{route}/v1"

    def start(self):
        self.thread.start()

    def close(self):
        try:
            if self.thread.is_alive():
                self.server.shutdown()
                self.thread.join(timeout=2)
            self.server.server_close()
        finally:
            with self._client_guard:
                self._client_closed = True
                if self._client_context is not None:
                    self._client_context.__exit__(None, None, None)
                    self._client_context = None

    def record(self, route, requested, effective, *, info=None, status=0, ms=0, digest=None):
        if self.audit:
            # No prompt, caller/session identifier, key, token, header value, or
            # caller-controlled reason. Identity is used for routing, not logging.
            row = {"route": route, "platform": "", "owner_verified": False,
                   "activation_mode": "verified",
                   "decision": "no_middleware_context", "identity_ms": None,
                   "identity_reason": "no_context"}
            row.update({k: v for k, v in (info or {}).items() if k in row})
            row["identity_reason"] = safe_identity_reason(row["identity_reason"])
            row.update({"owner_mode_sent": requested,
                        "owner_mode_scale_sent": (scales_sent(selected(self.declared(route), requested))
                                                  if requested != "off" else None),
                        "effective": effective, "status": status, "ms": ms,
                        "profile_digest": None, "digest_drift": None, "digest_unchecked": None})
            if requested != "off":
                # The engine's own digest for each applied profile, checked against the route's pins.
                drifted, unchecked = digest_check(selected(self.declared(route), requested), effective, digest)
                row.update(profile_digest=digest, digest_drift=",".join(drifted) or None,
                           digest_unchecked=",".join(unchecked) or None)
                if drifted:
                    logger.error("OWNER_MODE_DIGEST_DRIFT route=%s profiles=%s: the engine applied profile content "
                                 "that differs from the digest this route pins", route, row["digest_drift"])
            try:
                self.audit(row)
            except Exception:
                pass

    def stock(self, body, route=None):
        # Absent means off; preserve all remaining request fields.
        result = {k: v for k, v in body.items() if k not in OWNER_FIELDS}
        return compatible_request(result, self.routes[route].get('request_compatibility')) if route else result

    def declared(self, route):
        """The route's current profile rules. A disabled or invalid route declares none."""
        if route in self.disabled:
            return ()
        try:
            return route_profiles(self.routes[route])
        except (KeyError, ValueError):
            return ()

    def loaded_profiles(self, route, headers):
        """Names the engine reports as loaded, or None when it cannot be asked."""
        try:
            auth = {"Authorization": headers["Authorization"]} if headers.get("Authorization") else {}
            with self.client.stream("GET", self.routes[route]["upstream"].rstrip("/") + "/owner-mode/profiles",
                                    headers=auth, timeout=httpx.Timeout(10, connect=5)) as response:
                if response.status_code != 200:
                    return None
                raw = b""
                for chunk in response.iter_bytes():
                    raw += chunk
                    if len(raw) > MAX_DISCOVERY_BODY:
                        return None
            rows = json.loads(raw)["data"]
            return frozenset(row["name"] for row in rows if isinstance(row.get("name"), str)
                             and row.get("loaded", True) is True)
        except Exception:
            return None

    def signed(self, route, body, rules, headers):
        """Final wire bytes for an already selected subset of the route's profiles; stock when it is empty."""
        headers = {k: v for k, v in headers.items() if k != "X-Owner-Mode-Auth"}
        if not rules:
            return canonical(self.stock(body, route)), headers, "off"
        final = self.stock(body, route)
        final.update(wire(rules))
        raw = canonical(final)
        headers["X-Owner-Mode-Auth"] = sign(read_key(self.key_file), raw)
        return raw, headers, label(rules)

    def prepare(self, route, body, ticket="", *, info=None):
        rules = ()
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
        declared = self.declared(route)
        if always_on:
            rules = declared
        else:
            # A ticket grants exactly this route's verified profiles. Profiles with the
            # "always" rule need no ticket and no identity. A ticket is spent even when
            # the route has nothing to grant.
            wanted = verified_label(declared)
            try:
                granted = self.policy.consume(ticket, body, route, info=info) if ticket else "off"
            except Exception:
                granted = "off"
            verified = wanted != "off" and granted == wanted
            rules = tuple(rule for rule in declared if rule.access == "always" or verified)
        required = always_on or any(rule.fail == "required" for rule in rules)
        try:
            if rules:
                key = read_key(self.key_file)
        except Exception:
            if always_on:
                raise ValueError("Always On requires a usable profile and signing key") from None
            if required:
                raise ValueError("A required profile needs a usable signing key") from None
            rules = ()
        if not rules:
            if always_on:
                raise ValueError("Always On requires an enabled profile for this route")
            return canonical(self.stock(body, route)), headers, "off"
        final = self.stock(body, route)
        # Profiles and strengths come ONLY from this route's configuration; any client-supplied value was dropped above.
        final.update(wire(rules))
        try:
            raw = canonical(final)
            headers["X-Owner-Mode-Auth"] = sign(key, raw)
        except Exception:
            if always_on:
                raise ValueError("Always On request could not be signed") from None
            if required:
                raise ValueError("A required profile could not be signed") from None
            return canonical(self.stock(body, route)), {"Content-Type": "application/json"}, "off"
        return raw, headers, label(rules)
