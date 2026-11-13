# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launch diagnostics retain the failing step without starting a container."""
import hashlib
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import launch
import supervisor
from test_cleanup import fake_launch


@pytest.mark.parametrize('function,step', [
    ('healthy', 'engine health and profile discovery'),
    ('native_coverage', 'native worker coverage'),
    ('check_live_worker_pids', 'native worker liveness'),
])
def test_startup_timeout_names_last_readiness_failure(tmp_path, monkeypatch, function, step):
    args = fake_launch(tmp_path, monkeypatch)
    monkeypatch.setattr(launch, 'run', lambda *args, **kwargs: 'c' * 64)
    guard = Mock()
    guard.poll.return_value = None
    def start_guard(argv, **kwargs):
        record = Path(argv[argv.index('--record') + 1])
        supervisor.save(record.with_name('guard-ready.json'), {
            'record_sha256': hashlib.sha256(record.read_bytes()).hexdigest()})
        return guard
    monkeypatch.setattr(launch.subprocess, 'Popen', start_guard)
    monkeypatch.setattr(launch, 'owned_container', lambda record: {'running': True})
    monkeypatch.setattr(launch, 'available_memory', lambda: 64 * launch.GIB)
    stopped = Mock()
    monkeypatch.setattr(launch, 'stop_owned', stopped)
    monkeypatch.setattr(launch, 'healthy', lambda *args, **kwargs: 4096)
    monkeypatch.setattr(launch, 'native_coverage', lambda *args, **kwargs: {})
    monkeypatch.setattr(launch, 'check_live_worker_pids', lambda *args: None)
    def failure(*args, **kwargs):
        raise ValueError('synthetic readiness cause')
    monkeypatch.setattr(launch, function, failure)
    ticks = count(0, 100)
    monkeypatch.setattr(launch.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)
    with pytest.raises(TimeoutError, match=step + ': ValueError: synthetic readiness cause'):
        launch.main([*args, '--startup-timeout', '60'])
    stopped.assert_called_once()
    guard.terminate.assert_called_once()


def test_failed_helper_names_operation_exit_and_stderr(monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=17, stdout='', stderr='synthetic create rejected\n'))
    with pytest.raises(RuntimeError, match='docker create failed \\(exit 17\\): synthetic create rejected'):
        supervisor.run(['docker', 'create', 'fixture'])
