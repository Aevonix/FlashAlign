# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Generation ownership for the native llama.cpp process and local admin socket."""
import asyncio
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import stat
import threading

from flashalign_admin_server import LocalAdminServer, dispatch_operation
from flashalign_lifecycle import (Artifact, Generation, LifecycleError,
                                 LifecycleManager, RecoveryRequired, WorkerAck)
from profile_gguf import convert, read_profile, read_snapshot


def _identity(directory):
    hashes = {name: hashlib.sha256(read_snapshot(Path(directory)/name, limit)).hexdigest()
              for name, limit in (('profile.json', 4*1024*1024), ('profile.safetensors', 256*1024*1024))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stage_projection(source, destination):
    source, destination = Path(source), Path(destination)
    before, _, _ = read_profile(source)
    digest = _identity(source)
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise LifecycleError('Profile staging requires an empty private directory')
    for name in ('profile.json', 'profile.safetensors', 'sha256.json', 'LICENSE', 'LICENSE.txt',
                 'LICENSE.source', 'NOTICE.md', 'RESPONSIBLE_USE.md'):
        original = source/name
        if original.is_file():
            (destination/name).write_bytes(read_snapshot(original, 256*1024*1024 if name == 'profile.safetensors' else 4*1024*1024))
            (destination/name).chmod(0o600)
    checked, _, _ = read_profile(destination)
    if checked != before or _identity(destination) != digest:
        raise LifecycleError('Profile changed during staging')
    return Artifact(checked['name'], digest, checked['model']['id'], checked['model']['revision'],
                    checked.get('transform', 'projection'), destination, tuple(checked.get('requires', [])))


class NativeControl:
    """Private launcher-to-worker transport; never exposed through the public listener."""
    def __init__(self, port, token):
        self.port, self.token = port, token

    def call(self, action, **fields):
        body = json.dumps({'action': action, **fields}, separators=(',', ':'), allow_nan=False).encode()
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=180)
        try:
            connection.request('POST', '/flashalign/internal/profile', body=body,
                               headers={'Content-Type': 'application/json', 'X-FlashAlign-Internal-Key': self.token})
            response = connection.getresponse()
            data = response.read(1024*1024+1)
            if response.status != 200 or len(data) > 1024*1024:
                raise LifecycleError('Native lifecycle operation was rejected')
            value = json.loads(data)
            if not isinstance(value, dict):
                raise LifecycleError('Invalid native lifecycle response')
            return value
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise LifecycleError('Native lifecycle transport failed') from error
        finally:
            connection.close()


class LlamaCppBackend:
    identity = 'llamacpp-projection-v1'
    worker_ids = ('native',)

    def __init__(self, control, *, model, profile_model, gguf_python):
        self.control = control
        self.model, self.profile_model, self.gguf_python = Path(model), dict(profile_model), Path(gguf_python)
        self.loaded = {}
        self.absent_retirements = set()

    def validate(self, artifact):
        metadata, _, _ = read_profile(artifact.directory)
        if (artifact.model_id != self.profile_model['id'] or artifact.model_revision != self.profile_model['revision']
                or metadata['model'] != self.profile_model or artifact.operator != 'projection'
                or artifact.requirements or _identity(artifact.directory) != artifact.digest):
            raise LifecycleError('Profile does not match this native model and operator')

    def estimate(self, artifact):
        return {'profile_bytes': 2*(artifact.directory/'profile.safetensors').stat().st_size + 1024*1024}

    def reserve(self, transaction, generation):
        self.validate(generation.artifact)

    def _ack(self, transaction, phase, generation=None):
        return [WorkerAck('native', transaction['id'], generation.id if generation else None,
                          generation.artifact.digest if generation else None, phase)]

    def _prepare(self, generation):
        self.validate(generation.artifact)
        native = generation.artifact.directory/'runtime.gguf'
        candidate = generation.artifact.directory/('.runtime-'+secrets.token_hex(16)+'.gguf')
        try:
            receipt = convert(generation.artifact.directory, candidate, self.gguf_python, self.model)
            candidate.chmod(0o600)
            if native.exists() or native.is_symlink():
                if hashlib.sha256(read_snapshot(native, 512*1024*1024)).hexdigest() != receipt['adapter_sha256']:
                    raise LifecycleError('Stored native profile differs from its validated conversion')
            else:
                os.replace(candidate, native)
        finally:
            if candidate.exists(): candidate.unlink()
        value = self.control.call('prepare', generation=generation.id, digest=generation.artifact.digest,
                                  path=str(native.resolve()), native_digest=receipt['profile_digest'],
                                  file_sha256=receipt['adapter_sha256'])
        if type(value.get('index')) is not int or value['index'] < 0:
            raise LifecycleError('Native worker returned an invalid generation index')
        self.loaded[generation.id] = (value['index'], generation.artifact.digest)

    def prepare(self, transaction, generation):
        if generation.id not in self.loaded:
            self._prepare(generation)
        elif self.loaded[generation.id][1] != generation.artifact.digest:
            raise LifecycleError('Generation digest changed')
        return self._ack(transaction, 'prepared', generation)

    def fence(self, transaction):
        self.control.call('fence')
        return self._ack(transaction, 'fenced')

    def commit(self, transaction, generation):
        self.control.call('commit', generation=generation.id, digest=generation.artifact.digest)
        return self._ack(transaction, 'committed', generation)

    def resume(self, transaction):
        self.control.call('resume')
        return self._ack(transaction, 'resumed')

    def abort(self, transaction, generation):
        if not transaction['was_allocated']:
            # A worker may have finished preparation even when its response
            # was lost. Discover ownership before declaring the abort complete.
            rows = self.control.call('inventory').get('generations')
            if not isinstance(rows, list): raise RecoveryRequired('Native worker inventory is unavailable')
            found = [row for row in rows if row.get('generation') == generation.id]
            if found:
                if len(found) != 1 or found[0].get('digest') != generation.artifact.digest:
                    raise RecoveryRequired('Native prepared generation differs from its journal')
                self.control.call('retire', generation=generation.id, digest=generation.artifact.digest)
            self.loaded.pop(generation.id, None)
        return self._ack(transaction, 'aborted', generation)

    def references(self, generation):
        if generation.id in self.absent_retirements: return 0
        value = self.control.call('references', generation=generation.id, digest=generation.artifact.digest)
        count = value.get('references')
        if type(count) is not int or count < 0:
            raise RecoveryRequired('Native worker returned invalid references')
        return count

    def retire(self, transaction, generation):
        if generation.id not in self.absent_retirements:
            self.control.call('retire', generation=generation.id, digest=generation.artifact.digest)
        self.loaded.pop(generation.id, None)
        self.absent_retirements.discard(generation.id)
        return self._ack(transaction, 'retired', generation)

    def reconcile(self, transaction, snapshot):
        self.control.call('fence')
        native = self.control.call('inventory').get('generations')
        if not isinstance(native, list):
            raise RecoveryRequired('Native worker inventory is unavailable')
        expected = {gid: row for gid, row in snapshot['generations'].items()
                    if row['allocated'] and row['state'] != 'retired'}
        self.loaded = {}
        self.absent_retirements = set()
        for row in native:
            gid = row.get('generation')
            if (gid not in expected or row.get('digest') != expected[gid]['artifact']['digest']
                    or type(row.get('index')) is not int):
                raise RecoveryRequired('Native worker inventory differs from its journal')
            self.loaded[gid] = (row['index'], row['digest'])
        for gid, row in expected.items():
            if gid not in self.loaded:
                if transaction.get('operation') == 'retire' and transaction.get('generation') == gid:
                    # The previous process may have released this generation
                    # before its acknowledgment or journal update was lost.
                    self.absent_retirements.add(gid)
                    continue
                fields = {**row['artifact'], 'directory': Path(snapshot['artifact_root'])/gid}
                fields['requirements'] = tuple(fields['requirements'])
                self._prepare(Generation(gid, Artifact(**fields)))
        return self._ack(transaction, 'reconciled')

    def evict_cache(self):
        return self.control.call('evict')


def _admin_key(path):
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_size != 32):
            raise LifecycleError('Invalid private administration key')
        return stream.read(32)


class NativeLifecycle:
    def __init__(self, state, *, control, model, profiles, gguf_python, resource_limit=1024*1024*1024, admin_socket=None):
        self.instance_id = secrets.token_hex(16)
        self._active_requests = 0
        self._requests_lock = threading.Lock()
        self.state = Path(state)/'lifecycle'
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.admin_socket = Path(admin_socket) if admin_socket is not None else self.state/'admin.sock'
        metadata = [read_profile(path)[0] for path in profiles]
        if not metadata or any(item['model'] != metadata[0]['model'] for item in metadata):
            raise LifecycleError('Initial profiles must identify the same model')
        key = _admin_key(self.state/'admin.key')
        self.backend = LlamaCppBackend(control, model=model, profile_model=metadata[0]['model'], gguf_python=gguf_python)
        self.manager = LifecycleManager(self.state, backend=self.backend, stager=stage_projection,
                                        admin_key=key, resource_limits={'profile_bytes': resource_limit})
        try:
            self._start(profiles, metadata, key)
        except BaseException:
            try:
                if getattr(self, 'loop', None) is not None:
                    try:
                        asyncio.run_coroutine_threadsafe(self.server.close(), self.loop).result(timeout=10)
                    finally:
                        self.loop.call_soon_threadsafe(self.loop.stop)
                        self.thread.join(timeout=10)
                        self.loop.close()
            finally:
                self.manager.close()
            raise

    def _start(self, profiles, metadata, key):
        self.permit = self.manager.admin(key)
        if self.manager.journal_was_present:
            self.manager.recover(self.permit)
        for path, item in zip(profiles, metadata):
            self.manager.activate_configured(self.permit, path, name=item['name'])
        self.server = LocalAdminServer(self.admin_socket, status=self.manager.status,
                                      operation=self.operation, conflict_errors=(LifecycleError,))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name='flashalign-local-admin', daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.server.start(), self.loop).result(timeout=10)

    def operation(self, request):
        return dispatch_operation(self.manager, self.permit, request, evict_cache=self.backend.evict_cache)

    def names(self):
        return {name for name, row in self.manager.status()['aliases'].items() if row['generation'] is not None}

    def profiles(self):
        status = self.manager.status()
        generations = {row['id']: row for row in status['generations']}
        return [{'name': name, 'model': {'id': generations[row['generation']]['model_id'],
                                       'revision': generations[row['generation']]['model_revision']},
                 'loaded': True, 'generation': row['generation'], 'revision': row['revision']}
                for name, row in status['aliases'].items() if row['generation'] is not None]

    def safe_status(self):
        status = self.manager.status()
        with self._requests_lock:
            active = self._active_requests
        value = {'schema': 'flashalign.runtime-status.v1', 'instance_id': self.instance_id,
                 'engine': 'llamacpp', 'version': json.loads((Path(__file__).parent/'backend.json').read_text())['revision'],
                 'model': self.backend.profile_model['id'], 'active_requests': active,
                 'output_tokens_total': None, 'ttft_seconds_sum': None, 'ttft_seconds_count': None}
        aliases = [row['generation'] for row in status['aliases'].values() if row['generation'] is not None]
        if len(aliases) == 1:
            value['generation'] = aliases[0]
            value['profile_digest'] = next(row['digest'] for row in status['generations'] if row['id'] == aliases[0])
        return value

    @contextmanager
    def request_scope(self):
        with self._requests_lock:
            self._active_requests += 1
        try:
            yield
        finally:
            with self._requests_lock:
                self._active_requests -= 1

    @contextmanager
    def pin(self, alias):
        with self.manager.pin(alias) as lease:
            yield self.backend.loaded[lease.generation.id][0], lease.generation.id, lease.generation.artifact.digest

    def close(self):
        asyncio.run_coroutine_threadsafe(self.server.close(), self.loop).result(timeout=10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()
        self.manager.close()
