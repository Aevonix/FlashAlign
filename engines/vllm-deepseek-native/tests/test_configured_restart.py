# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Configured profiles respect journal aliases on restart with CPU fake transports."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_bootstrap import setup_service, close


class AdminTransport:
    def __init__(self, *args, **kwargs):
        pass

    async def start(self):
        return self

    async def close(self):
        pass


def offline_admin(module, monkeypatch):
    monkeypatch.setattr(module, 'LocalAdminServer', AdminTransport)
    if hasattr(module, 'NativeLifecycle'):
        # Event-loop creation itself opens a socketpair. Keep administration
        # fully in process while exercising the real startup reconciliation.
        def event_loop():
            loop = Mock()
            loop.is_closed.return_value = False
            loop.close.side_effect = lambda: setattr(loop.is_closed, 'return_value', True)
            loop.call_soon_threadsafe.side_effect = lambda callback: callback()
            return loop
        def schedule(coroutine, loop):
            coroutine.close()
            return SimpleNamespace(result=lambda **kwargs: None)
        monkeypatch.setattr(module, 'asyncio', SimpleNamespace(
            new_event_loop=event_loop, run_coroutine_threadsafe=schedule))
        monkeypatch.setattr(module, 'threading', SimpleNamespace(
            Lock=module.threading.Lock, Thread=Mock))


def test_new_configured_name_is_added_and_existing_alias_is_preserved(setup_service, monkeypatch):
    module, source, _, _, create = setup_service
    offline_admin(module, monkeypatch)
    if hasattr(module, 'read_profile'):
        monkeypatch.setattr(module, 'read_profile', lambda path: (
            json.loads((path/'profile.json').read_text()), None, None))
    service = create()
    original = service.manager.status()['aliases']['owner']
    close(service)
    profile = json.loads((source/'profile.json').read_text())
    profile['name'] = 'additional'
    (source/'profile.json').write_text(json.dumps(profile))
    reopened = create()
    try:
        aliases = reopened.manager.status()['aliases']
        assert aliases['owner'] == original
        assert aliases['additional']['generation'] is not None
        assert aliases['additional']['revision'] == 1
    finally:
        close(reopened)
    again = create()
    try:
        assert again.manager.status()['aliases'] == aliases
        assert 'prepared' not in again.backend.events
    finally:
        close(again)


@pytest.mark.parametrize('retire', [False, True], ids=['kept', 'retired'])
def test_configured_deactivated_profile_stays_deactivated(setup_service, monkeypatch, retire):
    module, _, _, _, create = setup_service
    offline_admin(module, monkeypatch)
    service = create()
    old = service.manager.status()['aliases']['owner']['generation']
    if hasattr(module, 'NativeLifecycle'):
        service.backend.evict_cache = Mock()
    service.operation({'action': 'deactivate', 'alias': 'owner', 'expected_revision': 1})
    if retire:
        service.operation({'action': 'retire', 'generation': old})
    generations = service.manager.status()['generations']
    close(service)
    stager = 'stage_profile' if hasattr(module, 'Controller') else (
        'stage_projection' if hasattr(module, 'NativeLifecycle') else 'stage_validated_profile')
    stage = Mock(wraps=getattr(module, stager))
    monkeypatch.setattr(module, stager, stage)
    reopened = create()
    try:
        alias = reopened.manager.status()['aliases']['owner']
        assert alias == {'generation': None, 'revision': 2}
        assert reopened.manager.status()['generations'] == generations
        stage.assert_not_called()
        assert 'prepared' not in reopened.backend.events
    finally:
        close(reopened)
