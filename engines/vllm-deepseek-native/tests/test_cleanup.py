# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Owned Docker cleanup with mocked commands; no Docker daemon or GPU is used."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest import mock

import pytest

import cleanup
import engine_config
import launch
import supervisor


def deployment(tmp_path):
    state = tmp_path/'state'; state.mkdir(mode=0o700)
    supervisor.save(state/'deployment.json', {'container_runtime': 'docker', 'image': engine_config.IMAGE,
                                            'image_id': 'sha256:'+'1'*64})
    runtime = supervisor.private_directory(state/'native-runtime')
    supervisor.private_directory(runtime/'runs')
    run = supervisor.private_directory(runtime/'runs'/('a'*24))
    intent = {'schema': 'flashalign.container-intent.v1', 'container_name': 'flashalign-native-'+run.name,
              'image_id': 'sha256:'+'1'*64, 'daemon_id': 'fixture-daemon',
              'labels': {'io.aevonix.flashalign.deployment': hashlib.sha256(str(runtime).encode()).hexdigest(),
                         'io.aevonix.flashalign.run': run.name},
              'parent_pid': 123, 'parent_start': 'fixture', 'started_unix': 1., 'profile_digest': '2'*64}
    record = {key: value for key, value in intent.items() if key not in ('schema', 'container_name')}
    record['container_id'] = 'c'*64
    supervisor.save(run/'container-intent.json', intent); supervisor.save(run/'container.json', record)
    return state, run, record


def fake_docker(monkeypatch, record, *, daemon='fixture-daemon', inspect_error=False,
                list_error=False, listed='', stop_error=False, stop_effect=True, actual=None):
    state = {'running': True, 'calls': []}
    def run(command, **kwargs):
        state['calls'].append(command)
        if command[1] == 'info':
            if isinstance(daemon, Exception): raise daemon
            value = daemon() if callable(daemon) else daemon
            return SimpleNamespace(returncode=0, stdout=value+'\n')
        if command[1] == 'inspect':
            if inspect_error:
                return SimpleNamespace(returncode=1, stdout='')
            value = actual or {'id': record['container_id'], 'image': record['image_id'],
                               'labels': record['labels'], 'running': state['running']}
            return SimpleNamespace(returncode=0, stdout=json.dumps(value))
        if command[1:3] == ['container', 'ls']:
            return SimpleNamespace(returncode=1 if list_error else 0, stdout=listed)
        if command[1] == 'stop':
            assert command[-1] == record['container_id']
            if stop_effect: state['running'] = False
            return SimpleNamespace(returncode=1 if stop_error else 0, stdout='')
        raise AssertionError('Unexpected fake Docker command')
    monkeypatch.setattr(supervisor.subprocess, 'run', run)
    return state


def test_bound_running_container_is_stopped_and_verified(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    fake = fake_docker(monkeypatch, record)
    result = cleanup.cleanup(state, time.monotonic()+35)
    assert result['external_resources_stopped'] is True and result['resource_count'] == 1
    assert fake['running'] is False
    assert [row for row in fake['calls'] if row[1] == 'stop'] == [
        ['docker', 'stop', '--time', '10', record['container_id']]]


@pytest.mark.parametrize('daemon', ['another-daemon', OSError('synthetic daemon unavailable')])
def test_changed_or_unavailable_daemon_never_proves_absence(tmp_path, monkeypatch, daemon):
    state, _, record = deployment(tmp_path)
    fake = fake_docker(monkeypatch, record, daemon=daemon)
    with pytest.raises((OSError, ValueError, RuntimeError)):
        cleanup.cleanup(state, time.monotonic()+35)
    assert not any(row[1] == 'stop' for row in fake['calls'])


@pytest.mark.parametrize('list_error,listed', [(True, ''), (False, 'c'*64)])
def test_inspect_failure_with_unknown_or_present_container_is_incomplete(tmp_path, monkeypatch, list_error, listed):
    state, _, record = deployment(tmp_path)
    fake = fake_docker(monkeypatch, record, inspect_error=True, list_error=list_error, listed=listed)
    with pytest.raises(RuntimeError): cleanup.cleanup(state, time.monotonic()+35)
    assert not any(row[1] == 'stop' for row in fake['calls'])


def test_confirmed_absence_requires_successful_query_on_same_daemon(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    fake = fake_docker(monkeypatch, record, inspect_error=True)
    assert cleanup.cleanup(state, time.monotonic()+35)['external_resources_stopped'] is True
    assert sum(row[1] == 'info' for row in fake['calls']) == 2
    assert not any(row[1] == 'stop' for row in fake['calls'])


def test_daemon_change_during_absence_query_is_incomplete(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    identities = iter(['fixture-daemon', 'another-daemon'])
    fake_docker(monkeypatch, record, daemon=lambda: next(identities), inspect_error=True)
    with pytest.raises(ValueError): cleanup.cleanup(state, time.monotonic()+35)


@pytest.mark.parametrize('change', [{'labels': {'foreign': 'yes'}}, {'image': 'sha256:'+'f'*64}])
def test_foreign_container_identity_is_never_stopped(tmp_path, monkeypatch, change):
    state, _, record = deployment(tmp_path)
    actual = {'id': record['container_id'], 'image': record['image_id'],
              'labels': record['labels'], 'running': True, **change}
    fake = fake_docker(monkeypatch, record, actual=actual)
    with pytest.raises(ValueError): cleanup.cleanup(state, time.monotonic()+35)
    assert not any(row[1] == 'stop' for row in fake['calls'])


@pytest.mark.parametrize('missing', ['container.json', 'container-intent.json'])
def test_partial_launch_without_bound_record_or_intent_is_unresolved(tmp_path, monkeypatch, missing):
    state, run, record = deployment(tmp_path)
    (run/missing).unlink()
    fake = fake_docker(monkeypatch, record)
    with pytest.raises(OSError): cleanup.cleanup(state, time.monotonic()+35)
    assert fake['calls'] == []


def test_record_cannot_borrow_another_creation_intent(tmp_path, monkeypatch):
    state, run, record = deployment(tmp_path)
    supervisor.save(run/'container.json', {**record, 'daemon_id': 'another-daemon'})
    fake = fake_docker(monkeypatch, record)
    with pytest.raises(ValueError): cleanup.cleanup(state, time.monotonic()+35)
    assert fake['calls'] == []


def test_stop_failure_requires_verified_final_stopped_state(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    fake_docker(monkeypatch, record, stop_error=True, stop_effect=False)
    with pytest.raises(RuntimeError): cleanup.cleanup(state, time.monotonic()+35)


def test_concurrent_stop_cli_error_does_not_hide_verified_stopped_state(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    fake_docker(monkeypatch, record, stop_error=True, stop_effect=True)
    assert cleanup.cleanup(state, time.monotonic()+35)['external_resources_stopped'] is True


def test_expired_deadline_performs_no_docker_command(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    fake = fake_docker(monkeypatch, record)
    with pytest.raises(ValueError): cleanup.cleanup(state, time.monotonic()-1)
    assert fake['calls'] == []


def test_unsafe_run_symlink_is_not_followed(tmp_path, monkeypatch):
    state, _, record = deployment(tmp_path)
    (state/'native-runtime/runs'/('d'*24)).symlink_to(tmp_path, target_is_directory=True)
    fake_docker(monkeypatch, record)
    with pytest.raises(ValueError): cleanup.cleanup(state, time.monotonic()+35)


def test_prepared_deployment_with_no_creation_attempt_has_no_external_resources(tmp_path, monkeypatch):
    state = tmp_path/'state'; state.mkdir(mode=0o700)
    supervisor.save(state/'deployment.json', {'container_runtime': 'docker', 'image': engine_config.IMAGE})
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *a, **kw: pytest.fail('no Docker query without an attempt'))
    assert cleanup.cleanup(state, time.monotonic()+35)['resource_count'] == 0


def fake_launch(tmp_path, monkeypatch):
    model = tmp_path/'model'; model.mkdir()
    key = tmp_path/'key'; key.write_text('synthetic fixture, never used for authentication')
    profile = tmp_path/'profile'; profile.mkdir()
    monkeypatch.setattr(launch, 'NativeProfile', lambda path: SimpleNamespace(
        name='fixture', digest='1'*64, directory=profile, validate_payload=lambda: None))
    monkeypatch.setattr(launch, 'model_mount_plan', lambda path: {})
    monkeypatch.setattr(launch, 'verify', lambda *a: None)
    monkeypatch.setattr(launch, 'load_key', lambda *a: None)
    monkeypatch.setattr(launch, 'docker_image', lambda: {'id': 'sha256:'+'1'*64})
    monkeypatch.setattr(launch, 'resources', lambda *a, **kw: {})
    monkeypatch.setattr(launch, 'docker_daemon_identity', lambda: 'fixture-daemon')
    monkeypatch.setattr(launch, 'container_command', lambda *a, **kw: ['docker', 'create', 'fixture'])
    monkeypatch.setattr(launch.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(launch.socket, 'socket', lambda: mock.MagicMock())
    return ['--model', str(model), '--profile', str(profile), '--key-file', str(key),
            '--replay-file', str(tmp_path/'replay/nonces'), '--state-dir', str(tmp_path/'runtime'),
            '--model-receipt', str(tmp_path/'synthetic-model-receipt.json')]


def test_creation_intent_is_private_and_durable_before_docker_create(tmp_path, monkeypatch):
    args = fake_launch(tmp_path, monkeypatch)
    def create(command, **kwargs):
        assert command[:2] == ['docker', 'create']
        intents = list((tmp_path/'runtime/runs').glob('*/container-intent.json'))
        assert len(intents) == 1
        intent = json.loads(intents[0].read_text())
        assert intent['daemon_id'] == 'fixture-daemon'
        assert intent['schema'] == 'flashalign.container-intent.v1'
        assert intents[0].stat().st_mode & 0o777 == 0o600
        assert (tmp_path/'runtime/runs').stat().st_mode & 0o777 == 0o700
        assert not intents[0].with_name('container.json').exists()
        raise RuntimeError('Synthetic create acknowledgement loss')
    monkeypatch.setattr(launch, 'run', create)
    with pytest.raises(RuntimeError, match='Synthetic create'): launch.main(args)


def test_dry_run_does_not_leave_an_unresolved_container_intent(tmp_path, monkeypatch):
    args = fake_launch(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, 'run', lambda *a, **kw: pytest.fail('dry run must not create a container'))
    assert launch.main([*args, '--dry-run']) == 0
    assert not (tmp_path/'runtime/runs').exists()
