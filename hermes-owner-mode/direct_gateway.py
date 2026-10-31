# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Managed loopback gateway for engine-only Always On deployments."""
from __future__ import annotations

# Also executable from an immutable, private runtime copy.
if not __package__:
    import importlib.util
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('owner_mode', root/'__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    __package__ = 'owner_mode'

import argparse
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import httpx

from .auth import read_key, read_api_key
from .contacts import ApprovalStore
from . import engine_setup as lifecycle
from .policy import Policy
from .proxy import ProviderProxy
from .setup_runtime import SetupError
from .model_metadata import checked_request_compatibility


class _NotReady(SetupError):
    pass


@contextmanager
def _locked(state):
    # Imported only on the managed Linux path. Ordinary remote SDK imports do
    # not require platform-specific process supervision modules.
    import fcntl
    fd = os.open(Path(state)/'gateway.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise SetupError('Gateway lock must be an owned private regular file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SetupError('Another gateway lifecycle operation is in progress') from None
        yield
    finally:
        os.close(fd)


def read_private(path):
    """Configuration and receipts must be owned regular private files."""
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size > 4*1024*1024):
            raise SetupError('Gateway state must be an owned private regular file')
        with os.fdopen(fd) as source:
            fd = None
            value = json.load(source)
        if not isinstance(value, dict):
            raise SetupError('Gateway state must be a JSON object')
        return value
    finally:
        if fd is not None:
            os.close(fd)


def prepare_gateway(home, *, base_url, model, profile, key_file, api_key_file=None, api_key=None,
                    request_compatibility=None):
    """Stage an immutable runtime and private policy. Never starts a process."""
    lifecycle._require_pidfds()
    root = Path(home)/'.local/share/flashalign/gateways'
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _locked(root):
        return _prepare_gateway(home, base_url=base_url, model=model, profile=profile, key_file=key_file,
                                api_key_file=api_key_file, api_key=api_key,
                                request_compatibility=request_compatibility)


def _prepare_gateway(home, *, base_url, model, profile, key_file, api_key_file=None, api_key=None,
                     request_compatibility=None):
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise SetupError('Gateway upstream must be an HTTP(S) model URL without credentials')
    if not isinstance(model, str) or not model or not isinstance(profile, str) or not profile or profile == 'off':
        raise SetupError('Always On gateway requires a model and enabled profile')
    key_file = str(Path(key_file).expanduser().resolve())
    read_key(key_file)
    route = {'upstream': base_url.rstrip('/'), 'profile': profile, 'owner_mode_scale': 1.}
    if api_key_file is not None:
        api_key_file = str(Path(api_key_file).expanduser().resolve())
        read_api_key(api_key_file)
        route['api_key_file'] = api_key_file
    elif api_key is not None and (not isinstance(api_key, str) or not api_key or '\n' in api_key or '\r' in api_key):
        raise SetupError('Invalid existing upstream API credential')
    if request_compatibility is not None:
        if isinstance(request_compatibility, dict):
            if 'model' in request_compatibility:
                rule = checked_request_compatibility(request_compatibility)
                if rule['model'] != model:
                    raise SetupError('Gateway compatibility rule names a different model')
            else:
                declaration = checked_request_compatibility(request_compatibility, require_model=False)
                rule = checked_request_compatibility({**declaration, 'model':model})
        else:
            rule = checked_request_compatibility(request_compatibility)
        route['request_compatibility'] = rule
    source = Path(__file__).resolve().parent
    files = {p.name: p.read_bytes() for p in source.glob('*.py')}
    digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())}
    material = json.dumps({'route':route, 'model':model, 'key_file':key_file, 'runtime':digests,
                           'api_key_digest':hashlib.sha256(api_key.encode()).hexdigest()
                                            if api_key is not None and api_key_file is None else None}, sort_keys=True)
    state = Path(home)/'.local/share/flashalign/gateways'/hashlib.sha256(material.encode()).hexdigest()[:24]
    descriptor = state/'deployment.json'
    if descriptor.exists():
        previous = read_private(descriptor)
        config = read_private(previous['config'])
        if previous.get('kind') != 'direct_gateway' or config.get('runtime_hashes') != digests:
            raise SetupError('Existing gateway state does not match the prepared runtime')
        for name, digest in digests.items():
            if lifecycle.digest(Path(previous['runtime'])/name) != digest:
                raise SetupError('Prepared gateway runtime changed; preserve its state before replacing it')
        # Preparation never asserts liveness, even when a prior activation left
        # a descriptor behind. Only the identity and model checks can do that.
        return {**previous, 'status':'prepared'}
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime = state/'runtime'
    runtime.mkdir(mode=0o700)
    for name, raw in files.items():
        (runtime/name).write_bytes(raw)
        (runtime/name).chmod(0o600)
    if api_key_file is None and api_key is not None:
        token_path = state/'upstream.key'
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as target:
            target.write(api_key)
        route['api_key_file'] = str(token_path)
    approvals = state/'approvals.json'
    ApprovalStore(approvals).set_mode('always_on')
    port = lifecycle.unused_port()
    config_path = state/'gateway.json'
    config = {'version':1, 'port':port, 'model':model, 'profile':profile,
              'key_file':key_file, 'approvals_file':str(approvals), 'routes':{'head':route},
              'instance':secrets.token_hex(24), 'runtime_hashes':digests}
    lifecycle.private_json(config_path, config)
    result = {'version':1, 'kind':'direct_gateway', 'status':'prepared', 'state':str(state),
              'config':str(config_path), 'runtime':str(runtime), 'python':sys.executable,
              'deployment_file':str(descriptor),
              'base_url':f'http://127.0.0.1:{port}/head/v1', 'model':model, 'profile':profile,
              'approval_store':str(approvals)}
    result['stop_command'] = [sys.executable, str(runtime/'direct_gateway.py'), '--stop', str(descriptor)]
    result['activate_command'] = [sys.executable, str(runtime/'direct_gateway.py'), '--activate', str(descriptor)]
    lifecycle.private_json(descriptor, result)
    return result


def _ready_and_profile(deployment, config, supervisor, deadline):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise SetupError('Gateway startup timed out')
    command = (lifecycle._supervisor_state(Path(deployment['state']), supervisor) or {}).get('command')
    if command is None or lifecycle.process_identity(command.get('pid')) != command:
        raise _NotReady('Gateway supervisor has not established its command identity')
    base = deployment['base_url'].removesuffix('/head/v1')
    try:
        with httpx.Client(timeout=min(2.,remaining), follow_redirects=False, trust_env=False) as client:
            try:
                response = client.get(base+'/_flashalign/ready', headers={'X-FlashAlign-Instance':config['instance']})
            except httpx.ConnectError:
                raise _NotReady('Gateway is not listening yet') from None
            if response.status_code != 200 or response.json() != {'instance':config['instance'], 'process':command}:
                raise SetupError('Gateway readiness identity did not match its owned process')
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise SetupError('Gateway startup timed out')
            # This is deliberately an unsigned, ordinary OpenAI request. The
            # final gateway, not a special SDK, must enforce Always On.
            response = client.post(deployment['base_url']+'/chat/completions',
                json={'model':deployment['model'], 'messages':[{'role':'user','content':'Reply OK.'}],
                      'max_tokens':1, 'stream':False}, timeout=remaining)
            if response.status_code != 200 or response.headers.get('X-Owner-Mode-Effective') != deployment['profile']:
                raise SetupError('Gateway did not confirm the configured profile for an unsigned request')
    except (httpx.HTTPError, ValueError) as error:
        if isinstance(error, SetupError):
            raise
        raise SetupError('Gateway readiness request failed') from None
    if lifecycle.process_identity(command['pid']) != command:
        raise SetupError('Gateway exited during its activation check')
    return {'status':'active', 'profile':deployment['profile'], 'unsigned_request_confirmed':True}


def activate_gateway(deployment, *, timeout=30.):
    """Start/reuse one owned gateway; raw upstream service is never stopped."""
    lifecycle._require_pidfds()
    with _locked(deployment['state']):
        return _activate_gateway(deployment, timeout=timeout)


def _activate_gateway(deployment, *, timeout):
    deadline = time.monotonic()+timeout
    state = Path(deployment['state']); marker = state/'running.json'
    config = read_private(deployment['config'])
    # Changing policy is activation, never a side effect of preparation.
    store = ApprovalStore(config['approvals_file'])
    policy = store.read(require_existing=True)
    if not policy['enabled']:
        raise SetupError('Gateway deployment is disabled; enable it explicitly before activation')
    if policy['mode'] != 'always_on':
        store.set_mode('always_on')
    for name, digest in config['runtime_hashes'].items():
        if lifecycle.digest(Path(deployment['runtime'])/name) != digest:
            raise SetupError('Prepared gateway runtime changed')
    if marker.exists():
        previous = read_private(marker)
        supervisors = previous.get('supervisors', [])
        if len(supervisors) == 1 and lifecycle._supervisors_running(state, supervisors):
            check = _ready_and_profile(deployment, config, supervisors[0], deadline)
            return {**deployment, 'status':'running', 'gateway_check':check}
        stopped = lifecycle.stop_engine(deployment)
        if stopped['status'] != 'stopped':
            raise SetupError('Previous gateway ownership or cleanup is incomplete; preserve running.json')
    lifecycle._check_available_port(urlsplit(deployment['base_url']).port)
    receipt = 'supervisor-0.json'
    command = [deployment['python'], str(Path(deployment['runtime'])/'engine_supervisor.py'), '--state', str(state/receipt), '--',
               deployment['python'], str(Path(deployment['runtime'])/'direct_gateway.py'), '--serve', deployment['config']]
    log_fd = os.open(state/'gateway.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    process = None
    try:
        with os.fdopen(log_fd,'ab') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       cwd=state, start_new_session=True)
        lifecycle._children[process.pid] = process
        identity = lifecycle.process_identity(process.pid, include_zombies=True)
        if identity is None:
            # Keep a durable record even when /proc cannot supply a birth ID.
            # The common stop path refuses to signal an unverified PID and
            # preserves this marker for recovery if ownership stays unknown.
            fallback = {'pid':process.pid, 'start':None, 'uid':os.getuid()}
            lifecycle.private_json(marker, {'processes':[fallback], 'owned_members':[],
                'supervisors':[{'process':fallback,'receipt':receipt}], 'startup_identity_unverified':True})
            raise SetupError('Gateway supervisor exited before ownership was established')
        supervisor = {'process':identity, 'receipt':receipt}
        ownership = {'processes':[identity], 'owned_members':[identity], 'supervisors':[supervisor]}
        lifecycle.private_json(marker, ownership)
        last = None
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise SetupError('Gateway exited during startup; inspect its private gateway.log')
            try:
                check = _ready_and_profile(deployment, config, supervisor, deadline)
                result = {**deployment, 'status':'running', 'gateway_check':check}
                lifecycle.private_json(state/'deployment.json', result)
                return result
            except _NotReady as error:
                last = error
                time.sleep(min(.05,max(0.,deadline-time.monotonic())))
        raise SetupError('Gateway startup timed out before the configured profile was confirmed') from last
    except BaseException as error:
        if process is not None and marker.exists():
            stopped = lifecycle.stop_engine(deployment)
            if stopped['status'] != 'stopped':
                raise SetupError('Gateway startup failed and owned cleanup is incomplete; preserve running.json') from error
        raise


def stop_gateway(deployment):
    if deployment.get('kind') != 'direct_gateway':
        raise SetupError('Not a managed direct gateway descriptor')
    lifecycle._require_pidfds()
    with _locked(deployment['state']):
        return lifecycle.stop_engine(deployment)


def prior_client_gateway(home):
    """Resolve only this home's private, matching advertised gateway pointer."""
    path = Path(home)/'.config/flashalign/client.json'
    if not path.exists():
        return None
    client = read_private(path)
    if client.get('gateway') is not True:
        return None
    pointer = client.get('gateway_deployment')
    if not isinstance(pointer, str) or not pointer:
        raise SetupError('Existing gateway client has no owned deployment pointer')
    descriptor = Path(pointer).resolve()
    root = (Path(home)/'.local/share/flashalign/gateways').resolve()
    if not descriptor.is_relative_to(root) or descriptor.name != 'deployment.json':
        raise SetupError('Existing gateway pointer is outside this deployment home')
    deployment = read_private(descriptor)
    state = descriptor.parent
    if (deployment.get('kind') != 'direct_gateway' or Path(deployment.get('state','')).resolve() != state
            or Path(deployment.get('config','')).resolve() != state/'gateway.json'
            or Path(deployment.get('runtime','')).resolve() != state/'runtime'
            or deployment.get('base_url') != client.get('base_url')
            or deployment.get('model') != client.get('model') or deployment.get('profile') != client.get('profile')):
        raise SetupError('Existing gateway pointer does not match its advertised deployment')
    return deployment


def retire_prior_gateway(previous, replacement=None):
    """After replacement proof, retire only the previously advertised gateway."""
    if previous is None or replacement is not None and previous['state'] == replacement['state']:
        return None
    stopped = stop_gateway(previous)
    if stopped.get('status') != 'stopped':
        cleanup = stop_gateway(replacement) if replacement is not None else None
        suffix = ' Replacement gateway cleanup also needs recovery.' if cleanup and cleanup.get('status') != 'stopped' else ''
        raise SetupError('Previous gateway cleanup is incomplete; client configuration was preserved.'+suffix)
    return stopped


def serve(config_path):
    config = read_private(config_path)
    store = ApprovalStore(config['approvals_file'])
    store.read(require_existing=True)
    class ConfiguredProxy(ProviderProxy):
        def prepare(self, route, body, ticket='', *, info=None):
            if body.get('model') != config['model']:
                raise ValueError('Gateway request does not name its configured model')
            return super().prepare(route, body, ticket, info=info)
    proxy = ConfiguredProxy(Policy(), config['routes'], config['key_file'], port=config['port'], approvals=store)
    original = proxy.server.RequestHandlerClass
    class ReadyHandler(original):
        def do_GET(self):
            if self.path == '/head/v1/models':
                raw = json.dumps({'object':'list','data':[{'id':config['model'],'object':'model'}]}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
                return
            if self.path != '/_flashalign/ready':
                return super().do_GET()
            if not hmac.compare_digest(self.headers.get('X-FlashAlign-Instance',''),config['instance']):
                self.send_error(404)
                return
            raw = json.dumps({'instance':config['instance'], 'process':lifecycle.process_identity(os.getpid())}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    proxy.server.RequestHandlerClass = ReadyHandler
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig,lambda *_:done.set())
    proxy.start()
    try:
        done.wait()
    finally:
        proxy.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--serve',type=Path)
    action.add_argument('--activate',type=Path)
    action.add_argument('--stop',type=Path)
    args = parser.parse_args(argv)
    if args.serve:
        serve(args.serve)
    elif args.activate:
        print(json.dumps(activate_gateway(read_private(args.activate))))
    else:
        print(json.dumps(stop_gateway(read_private(args.stop))))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('FlashAlign gateway failed; inspect its private state.',file=sys.stderr)
        raise SystemExit(1)
