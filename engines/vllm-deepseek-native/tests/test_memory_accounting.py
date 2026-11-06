# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""The native launcher must not replace an unknown cgroup limit with host RAM."""
import pytest

import supervisor


GIB = 1024 ** 3


@pytest.fixture
def accounting(tmp_path):
    proc, cgroup = tmp_path / 'proc', tmp_path / 'cgroup'
    (proc / 'self').mkdir(parents=True)
    cgroup.mkdir()
    (proc / 'meminfo').write_text(f'MemAvailable: {500 * 1024 ** 2} kB\n')
    (proc / 'self/cgroup').write_text('0::/\n')
    (cgroup / 'cgroup.controllers').write_text('cpu memory pids\n')
    return proc, cgroup


def finite(path, limit, used, *, file=0, inactive=0, shmem=0):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'memory.max').write_text(str(limit))
    (path / 'memory.current').write_text(str(used))
    (path / 'memory.stat').write_text(
        f'file {file}\ninactive_file {inactive}\nshmem {shmem}\n')


def test_v1_limit_cannot_be_silently_replaced_with_host_ram(accounting):
    proc, cgroup = accounting
    (proc / 'self/cgroup').write_text('3:memory:/limited\n')
    limited = cgroup / 'memory/limited'
    limited.mkdir(parents=True)
    (limited / 'memory.limit_in_bytes').write_text(str(64 * GIB))
    (limited / 'memory.usage_in_bytes').write_text(str(60 * GIB))
    with pytest.raises(ValueError, match='unified cgroup v2'):
        supervisor.available_memory(proc, cgroup)


def test_missing_membership_file_is_rejected(accounting):
    proc, cgroup = accounting
    (proc / 'self/cgroup').unlink()
    with pytest.raises(ValueError, match='cgroup v2 membership is unreadable'):
        supervisor.available_memory(proc, cgroup)


def test_finite_parent_limit_is_preserved(accounting):
    proc, cgroup = accounting
    (proc / 'self/cgroup').write_text('0::/parent/child\n')
    finite(cgroup / 'parent', 64 * GIB, 60 * GIB)
    finite(cgroup / 'parent/child', 'max', 0)
    assert supervisor.available_memory(proc, cgroup) == 4 * GIB


@pytest.mark.parametrize('with_unlimited_limit', [False, True])
def test_valid_unbounded_v2_root_uses_host_memory(accounting, with_unlimited_limit):
    proc, cgroup = accounting
    if with_unlimited_limit:
        finite(cgroup, 'max', 0)
    assert supervisor.available_memory(proc, cgroup) == 500 * GIB


@pytest.mark.parametrize('membership', ['', '0::/absent\n', '0::relative\n',
                                        '0::/../../outside\n', '0::/\n0::/\n'])
def test_missing_or_invalid_membership_is_rejected(accounting, membership):
    proc, cgroup = accounting
    (proc / 'self/cgroup').write_text(membership)
    with pytest.raises(ValueError, match='cgroup v2'):
        supervisor.available_memory(proc, cgroup)


def test_symlink_escape_is_rejected(accounting, tmp_path):
    proc, cgroup = accounting
    outside = tmp_path / 'outside'
    outside.mkdir()
    (cgroup / 'escaped').symlink_to(outside, target_is_directory=True)
    (proc / 'self/cgroup').write_text('0::/escaped\n')
    with pytest.raises(ValueError, match='cgroup v2'):
        supervisor.available_memory(proc, cgroup)


def test_unresolvable_symlink_loop_is_rejected(accounting):
    proc, cgroup = accounting
    (cgroup / 'loop').symlink_to('loop', target_is_directory=True)
    (proc / 'self/cgroup').write_text('0::/loop\n')
    with pytest.raises(ValueError, match='cgroup v2 membership path is unavailable'):
        supervisor.available_memory(proc, cgroup)


def test_missing_v2_hierarchy_marker_is_rejected(accounting):
    proc, cgroup = accounting
    (cgroup / 'cgroup.controllers').unlink()
    with pytest.raises(ValueError, match='cgroup v2'):
        supervisor.available_memory(proc, cgroup)


@pytest.mark.parametrize('missing', ['memory.current', 'memory.stat'])
def test_incomplete_finite_accounting_is_rejected(accounting, missing):
    proc, cgroup = accounting
    finite(cgroup, 64 * GIB, 60 * GIB)
    (cgroup / missing).unlink()
    with pytest.raises(ValueError, match='accounting is incomplete'):
        supervisor.available_memory(proc, cgroup)


def test_ordinary_inactive_cache_credit_is_preserved(accounting):
    proc, cgroup = accounting
    finite(cgroup, 64 * GIB, 60 * GIB, file=12 * GIB,
           inactive=8 * GIB, shmem=6 * GIB)
    assert supervisor.available_memory(proc, cgroup) == 10 * GIB
