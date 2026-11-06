# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""API-side lifecycle ownership, independent of generation signing keys."""
import asyncio
from collections.abc import Mapping
import os
from pathlib import Path
import secrets
import threading
from types import SimpleNamespace

from owner_mode.profile_admin_server import LocalAdminServer
from owner_mode.profile_lifecycle import GenerationBusy, LifecycleError, LifecycleManager
from owner_mode.profile_staging import stage_validated_profile
from .native_profile import NativeProfile as Profile
from .lifecycle import VllmBackend


class AliasProfiles(Mapping):
    """Authentication sees public aliases; workers only see generation IDs."""
    def __init__(self, service):
        self.service = service

    def __iter__(self):
        return iter(k for k, v in self.service.manager.status()['aliases'].items() if v['generation'] is not None)

    def __len__(self):
        return sum(v['generation'] is not None for v in self.service.manager.status()['aliases'].values())

    def __getitem__(self, alias):
        status = self.service.manager.status()
        selected = status['aliases'].get(alias)
        if selected is None or selected['generation'] is None:
            raise KeyError(alias)
        record = next(row for row in status['generations'] if row['id'] == selected['generation'])
        return SimpleNamespace(name=alias, digest=record['digest'],
            is_native_wo=record['operator']=='native_matrix_override_v1',
            meta={'model':{'id':record['model_id'],'revision':record['model_revision']}})

    def __contains__(self, alias):
        row = self.service.manager.status()['aliases'].get(alias)
        return row is not None and row['generation'] is not None


class LifecycleService:
    def __init__(self, engine_client, loop, paths, state_dir, socket_path, *, loader=Profile):
        self.loader = loader
        self.instance_id = secrets.token_hex(16)
        self._admission = threading.RLock()
        self._operations = threading.RLock()
        initial = [loader(path) for path in paths]
        identities = {(p.meta['model']['id'], p.meta['model']['revision']) for p in initial}
        if len(identities) != 1:
            raise LifecycleError('A lifecycle service requires one pinned model')
        self.identity = next(iter(identities))
        self.profiles = {}
        backend = VllmBackend(engine_client, loop, validate=self.validate, estimate=self.estimate)
        if any((r['model_id'], r['model_revision']) != self.identity for r in backend.inventories):
            raise LifecycleError('API and worker model identities differ')
        free = min(r['available_device_bytes'] for r in backend.inventories)
        configured = int(os.environ.get('FLASHALIGN_PROFILE_DEVICE_BUDGET', free // 2))
        if configured <= 0 or configured > free:
            raise LifecycleError('Invalid profile device overlap budget')
        admin_key = secrets.token_bytes(32)
        self.manager = LifecycleManager(state_dir, backend=backend, stager=stage_validated_profile,
            admin_key=admin_key, resource_limits={
                'device_bytes': configured, 'host_bytes': int(os.environ.get('FLASHALIGN_PROFILE_HOST_BUDGET', 4 << 30))})
        # The permit never leaves this process. LocalAdminServer separately
        # authenticates the operating-system peer, not an inference header.
        self.permit = self.manager.admin(admin_key)
        self.backend = backend
        self.socket_path = socket_path
        self.server = None
        self.aliases = AliasProfiles(self)
        try:
            if self.manager.journal_was_present:
                self.manager.recover(self.permit)
            else:
                for path in paths:
                    generation = self.manager.stage(self.permit, path)
                    self.manager.activate(self.permit, generation.artifact.name, generation.id, expected_revision=0)
        except BaseException:
            self.manager.close()
            raise

    def validate(self, artifact):
        if (artifact.model_id, artifact.model_revision) != self.identity:
            raise LifecycleError('Profile targets another checkpoint')
        profile = self.loader(artifact.directory)
        if artifact.operator != 'native_matrix_override_v1':
            raise LifecycleError('Unsupported runtime operator')
        return profile

    def estimate(self, artifact):
        profile = self.validate(artifact)
        size = (profile.directory / profile.spec.payload_file).stat().st_size
        size = 4 * size // profile.spec.tp_world_size
        return {'device_bytes': size, 'host_bytes': size}

    def profile(self, generation):
        if generation.id not in self.profiles:
            profile = self.loader(generation.artifact.directory)
            profile.runtime_digest, profile.digest = profile.digest, generation.artifact.digest
            self.profiles[generation.id] = profile
        return self.profiles[generation.id]

    def pin(self, alias):
        # This method runs on the API event loop. Never wait for a control
        # thread whose collective RPC requires that same loop to progress.
        if not self._admission.acquire(blocking=False):
            raise GenerationBusy('Profile admission is temporarily fenced')
        try:
            return self.manager.pin(alias)
        finally:
            self._admission.release()

    def evict_cache(self):
        with self._admission:
            if any(any(row['references'].values()) for row in self.manager.status()['generations']):
                raise GenerationBusy('Requests must drain before cache eviction')
            return self.backend.evict_cache()

    def public_status(self):
        snapshot = self.manager.status()
        rows = self.backend._call('status')
        if {row.get('worker_id') for row in rows} != set(self.backend.worker_ids):
            raise LifecycleError('Worker status inventory changed')
        result = {'schema':'flashalign.runtime-status.v1', 'engine':'vllm',
                  'model':self.identity[0], 'instance_id':self.instance_id,
                  'numerical_attestation':False}
        active = {row['generation'] for row in snapshot['aliases'].values() if row['generation'] is not None}
        inventory = {row['id']:row for row in snapshot['generations']}
        if len(active) == 1:
            gid = next(iter(active))
            result.update(generation=gid, profile_digest=inventory[gid]['digest'])
        elif not active:
            result['generation'] = 'off'
        for name in ('graph_mode', 'speculative_method'):
            values = {row.get(name) for row in rows}
            if len(values) == 1:
                result[name] = next(iter(values))
        if 'speculative_method' in result:
            result['speculation_enabled'] = result['speculative_method'] is not None
        for name in ('expected_sites','installed_sites','exercised_sites','active_requests'):
            values = [row.get(name) for row in rows]
            if all(type(value) is int and value >= 0 for value in values):
                result[name] = max(values)
        memory = [row.get('gpu_memory_bytes') for row in rows]
        if all(type(value) is int and value >= 0 for value in memory):
            result['gpu_memory_bytes'] = sum(memory)
        result['active_profile_generations'] = [
            {'generation':gid,'profile_digest':inventory[gid]['digest']} for gid in sorted(active)]
        return result

    def capabilities(self):
        return {'schema':'flashalign.capabilities.v1','hot_loading':True,
                'execution_attestation':False,'status':True}

    def operation(self, payload):
        from owner_mode.profile_admin_server import dispatch_operation
        with self._operations:
            result = dispatch_operation(self.manager, self.permit, payload, evict_cache=self.evict_cache)
            live = {row['id'] for row in self.manager.status()['generations']}
            for gid in set(self.profiles) - live:
                del self.profiles[gid]
            return result

    async def start(self):
        self.server = LocalAdminServer(self.socket_path, status=self.manager.status,
            operation=self.operation, conflict_errors=(LifecycleError,))
        await self.server.start()

    async def close(self):
        if self.server is not None:
            await self.server.close()
        self.manager.close()


async def initialize(scope, paths, *, loader=Profile):
    root = Path(os.environ['FLASHALIGN_PROFILE_STATE'])
    client = getattr(scope.get('app').state, 'engine_client', None) if scope.get('app') is not None else None
    if client is None:
        raise LifecycleError('vLLM engine client is unavailable during lifecycle startup')
    loop = asyncio.get_running_loop()
    service = await asyncio.to_thread(LifecycleService, client, loop, paths, root,
        Path(os.environ['FLASHALIGN_PROFILE_ADMIN_SOCKET']), loader=loader)
    try:
        await service.start()
    except BaseException:
        service.manager.close()
        raise
    return service
