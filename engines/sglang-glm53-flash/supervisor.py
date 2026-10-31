# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Supervise owned GLM workers and enforce the host/cgroup memory floor."""
import argparse

import ctypes

import json

import os

from pathlib import Path

import signal

import subprocess

import sys

import time

def memory_headroom(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    fields = dict(line.split(':', 1) for line in (proc/'meminfo').read_text().splitlines())
    host = int(fields['MemAvailable'].split()[0])*1024
    values = [host]
    relative = next((line.split('::', 1)[1] for line in (proc/'self/cgroup').read_text().splitlines()
                     if line.startswith('0::')), None)
    if relative is None:
        raise ValueError('The GLM Flash memory guard requires unified cgroup v2 accounting')
    limits, raw_limits, counters = [], [], []
    if relative is not None:
        # A namespaced cgroup root need not expose the absolute /proc path.
        current = (cgroup/relative.lstrip('/')).resolve()
        if not current.is_relative_to(cgroup.resolve()) or not current.exists():
            current = cgroup.resolve()
        while current.is_relative_to(cgroup.resolve()):
            if (current/'memory.max').exists() and (current/'memory.current').exists():
                maximum = (current/'memory.max').read_text().strip()
                used = int((current/'memory.current').read_text())
                stat_file = current/'memory.stat'
                counts = {key: int(value) for key, value in
                          (line.split() for line in stat_file.read_text().splitlines())} if stat_file.exists() else {}
                # Account conservatively for ordinary checkpoint page cache.
                # Anonymous pinned pages and shmem never earn this credit.
                ordinary = max(0, counts.get('file', 0)-counts.get('shmem', 0)
                               -counts.get('unevictable', 0)-counts.get('file_dirty', 0)
                               -counts.get('file_writeback', 0))
                used = max(used, int((current/'memory.current').read_text()))
                credit = min(used, max(0, counts.get('inactive_file', 0)), ordinary)
                raw_available = available = None
                if maximum != 'max':
                    raw_available = max(0, int(maximum)-used)
                    available = min(int(maximum), raw_available+credit)
                    values.append(available)
                    limits.append(available)
                    raw_limits.append(raw_available)
                counters.append({'path': str(current.relative_to(cgroup.resolve())),
                                 'max_bytes': int(maximum) if maximum != 'max' else None, 'current_bytes': used,
                                 'raw_available_bytes': raw_available,
                                 'file_bytes': counts.get('file', 0), 'shmem_bytes': counts.get('shmem', 0),
                                 'inactive_file_bytes': counts.get('inactive_file', 0),
                                 'unevictable_bytes': counts.get('unevictable', 0),
                                 'file_dirty_bytes': counts.get('file_dirty', 0),
                                 'file_writeback_bytes': counts.get('file_writeback', 0),
                                 'ordinary_inactive_file_credit_bytes': credit,
                                 'working_available_bytes': available})
            if current == cgroup.resolve():
                break
            current = current.parent
    return {'host_available_bytes': host, 'cgroup_available_bytes': min(limits) if limits else None,
            'cgroup_raw_available_bytes': min(raw_limits) if raw_limits else None,
            'cgroups': counters, 'effective_available_bytes': min(values)}

def _process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None

def _process_tree(pid, root_start):
    """Snapshot identities while the same root process still owns the tree."""
    if root_start is None or _process_start(pid) != root_start:
        return None
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            processes[int(path.parent.name)] = {
                'pid': int(path.parent.name), 'parent': int(fields[1]),
                'start': fields[19], 'uid': path.parent.stat().st_uid,
                'state': fields[0]}
        except (OSError, ValueError, IndexError):
            continue
    if processes.get(pid, {}).get('start') != root_start or _process_start(pid) != root_start:
        return None
    selected = {pid}
    while True:
        enlarged = selected | {child for child, record in processes.items()
                               if record['parent'] in selected}
        if enlarged == selected:
            break
        selected = enlarged
    return [processes[child] for child in selected]

def _signal_identity(record, signum):
    """Use a pidfd so validation cannot turn into a signal to a reused PID."""
    if record['uid'] != os.getuid() or record['state'] == 'Z':
        return False
    try:
        fd = os.pidfd_open(record['pid'])
        try:
            if _process_start(record['pid']) != record['start']:
                return False
            signal.pidfd_send_signal(fd, signum)
            return True
        finally:
            os.close(fd)
    except OSError:
        return False

def _signal_tree(child, signum, root_start, *, exclude_root=False):
    records = _process_tree(child.pid, root_start)
    if records is None:
        return False
    for record in records:
        if not exclude_root or record['pid'] != child.pid:
            _signal_identity(record, signum)
    return True

def _set_subreaper(enabled):
    """Adopt orphaned native workers until this standalone supervisor exits."""
    if not callable(getattr(os, 'pidfd_open', None)) or not callable(getattr(signal, 'pidfd_send_signal', None)):
        raise RuntimeError('GLM Flash supervision requires Linux pidfd support')
    fd = os.pidfd_open(os.getpid())
    os.close(fd)
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    # Linux PR_GET_CHILD_SUBREAPER / PR_SET_CHILD_SUBREAPER affect this process
    # only. They keep workers observable when an intermediate parent crashes.
    if (libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0
            or libc.prctl(36, int(enabled), 0, 0, 0) != 0):
        raise OSError(ctypes.get_errno(), 'Cannot supervise orphaned engine workers')
    return bool(previous.value)

def _finish_children(root_start, child, grace=5):
    """Stop and reap only descendants of this standalone supervisor."""
    root = os.getpid()
    for signum in (signal.SIGTERM, signal.SIGKILL):
        deadline = time.monotonic()+grace
        while True:
            records = _process_tree(root, root_start)
            if records is None:
                raise RuntimeError('Supervisor process identity changed during cleanup')
            descendants = [record for record in records if record['pid'] != root]
            # Preserve Popen's real return code before reaping adopted children.
            child.poll()
            if child.returncode is not None:
                while True:
                    try:
                        reaped, _ = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        # Kernel confirmation, not a mixed-time /proc snapshot:
                        # all descendants have exited and been reaped.
                        return
                    if reaped == 0:
                        break
            live = [record for record in descendants if record['state'] != 'Z']
            for record in live:
                _signal_identity(record, signum)
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
    raise RuntimeError('GLM Flash worker cleanup is incomplete')

def supervise(command, floor_bytes):
    # Inherit the wizard's managed process group so forced rollback also reaches
    # native workers if this supervisor is killed before graceful shutdown ends.
    root_start = _process_start(os.getpid())
    if root_start is None:
        raise RuntimeError('The supervisor process identity could not be recorded')
    existing = _process_tree(os.getpid(), root_start)
    if existing is None or len(existing) != 1:
        raise RuntimeError('The GLM Flash supervisor must run in a dedicated process without existing children')
    previous_subreaper = _set_subreaper(True)
    child = None
    stopping = False
    def stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, stop)
        child = subprocess.Popen(command)
        while child.poll() is None:
            memory = memory_headroom()
            if memory['effective_available_bytes'] < floor_bytes:
                print('FlashAlign stopped GLM Flash because the host/cgroup safety floor was reached.', flush=True)
                stop()
            if stopping:
                return 1
            time.sleep(2)
        return child.returncode
    finally:
        try:
            if child is not None:
                _finish_children(root_start, child)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            _set_subreaper(previous_subreaper)
