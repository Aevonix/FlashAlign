# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Real standalone process-family cleanup; no model, GPU or external network."""
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from test_deepseek_setup import launcher
from owner_mode import engine_supervisor


class SharedSupervisorSnapshots(unittest.TestCase):
    def scan(self, first, second):
        root={'pid':101,'start':'100','uid':os.getuid()}
        child={'pid':102,'start':'101','uid':os.getuid()}
        identities={101:root,102:child}
        with (mock.patch.object(engine_supervisor,'identity',side_effect=lambda pid,*_:identities.get(pid)),
              mock.patch.object(engine_supervisor,'_child_ids',side_effect=lambda row:[102] if row['pid']==101 else []),
              mock.patch.object(engine_supervisor,'_process',side_effect=[first,second])):
            return engine_supervisor.descendants([root])

    def test_scheduler_state_change_preserves_owned_child(self):
        child={'pid':102,'start':'101','uid':os.getuid()}
        for before,after in [('R','S'),('S','D'),('D','R')]:
            with self.subTest(before=before,after=after):
                first={'identity':child,'parent':101,'state':before}
                self.assertIn(child,self.scan(first,{**first,'state':after}))

    def test_reused_child_pid_is_not_accepted(self):
        first={'identity':{'pid':102,'start':'101','uid':os.getuid()},'parent':101,'state':'R'}
        second={**first,'identity':{**first['identity'],'start':'102'}}
        self.assertEqual([101],[row['pid'] for row in self.scan(first,second)])

    def test_reparented_child_is_not_accepted_from_stale_parent(self):
        first={'identity':{'pid':102,'start':'101','uid':os.getuid()},'parent':101,'state':'R'}
        self.assertEqual([101],[row['pid'] for row in self.scan(first,{**first,'parent':999})])


class StandaloneSupervisor(unittest.TestCase):
    def run_family(self, *, interrupt=False, parent_status=0, separate_group=False):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary)/'workers.json'
            worker = (
                "import json,os,signal,socket,time;from pathlib import Path;"
                "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
                "s.bind(('127.0.0.1',0));s.listen(1);"
                f"Path({str(ready)!r}).write_text(json.dumps({{'pid':os.getpid(),"
                "'start':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],"
                "'port':s.getsockname()[1]}));time.sleep(120)"
            )
            parent = (
                "import subprocess,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session={separate_group!r});"
                f"time.sleep({120 if interrupt else .4});sys.exit({parent_status})"
            )
            program = (
                "import importlib.util,sys;"
                f"s=importlib.util.spec_from_file_location('supervisor',{launcher.__file__!r});"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                f"raise SystemExit(m.supervise([sys.executable,'-c',{parent!r}],0))"
            )
            process = subprocess.Popen([sys.executable, '-c', program], start_new_session=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            info = None
            try:
                deadline = time.monotonic()+8
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists(), 'worker did not bind its local test port')
                info = json.loads(ready.read_text())
                with socket.create_connection(('127.0.0.1', info['port']), timeout=1):
                    pass
                if interrupt:
                    process.send_signal(signal.SIGTERM)
                process.wait(timeout=15)
                self.assertEqual(process.returncode, 1 if interrupt else parent_status)
                self.assertFalse((Path('/proc')/str(info['pid'])).exists(),
                                 'worker survived or was not reaped')
                with socket.socket() as probe:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind(('127.0.0.1', info['port']))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                if info is not None:
                    path = Path('/proc')/str(info['pid'])/'stat'
                    if path.exists():
                        fd = os.pidfd_open(info['pid'])
                        try:
                            if path.read_text().rsplit(')', 1)[1].split()[19] == info['start']:
                                signal.pidfd_send_signal(fd, signal.SIGKILL)
                        finally:
                            os.close(fd)

    def test_successful_parent_exit_stops_and_reaps_orphaned_listener(self):
        self.run_family()

    def test_failed_parent_preserves_exit_code_and_stops_workers(self):
        self.run_family(parent_status=7)

    def test_sigterm_stops_workers_in_another_process_group(self):
        self.run_family(interrupt=True, separate_group=True)

    def test_missing_pidfd_support_fails_before_start(self):
        with (mock.patch.object(launcher.os, 'pidfd_open', None),
              mock.patch.object(launcher.subprocess, 'Popen') as start):
            with self.assertRaisesRegex(RuntimeError, 'pidfd'):
                launcher.supervise([sys.executable, '-c', 'pass'], 0)
        start.assert_not_called()

    def test_changed_identity_never_receives_a_signal(self):
        record = {'pid': os.getpid(), 'start': 'stale', 'uid': os.getuid(), 'state': 'S'}
        with mock.patch.object(launcher.signal, 'pidfd_send_signal') as send:
            self.assertFalse(launcher._signal_identity(record, signal.SIGTERM))
        send.assert_not_called()

    def test_existing_unrelated_child_is_preserved(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(120)'])
        try:
            with self.assertRaisesRegex(RuntimeError, 'dedicated process'):
                launcher.supervise([sys.executable, '-c', 'pass'], 0)
            self.assertIsNone(child.poll())
        finally:
            child.terminate();child.wait(timeout=5)

    def test_subreaper_setting_is_restored(self):
        libc = ctypes.CDLL(None, use_errno=True)
        before, after = ctypes.c_int(), ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(before), 0, 0, 0), 0)
        with mock.patch.object(launcher, 'memory_headroom', return_value={'effective_available_bytes': 10}):
            self.assertEqual(launcher.supervise([sys.executable, '-c', 'pass'], 0), 0)
        self.assertEqual(libc.prctl(37, ctypes.byref(after), 0, 0, 0), 0)
        self.assertEqual(before.value, after.value)

    def test_reparented_zombie_is_reaped_despite_a_mixed_time_snapshot(self):
        def state(pid):
            try:
                return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                return None
        def wait_for(predicate):
            deadline = time.monotonic()+5
            while not predicate():
                if time.monotonic() > deadline:
                    self.fail('owned process did not reach the expected state')
                time.sleep(.01)
        previous = launcher._set_subreaper(True)
        parent = None
        grandchild = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                ready = Path(temporary)/'ready.json'
                release = Path(temporary)/'release'
                program = (
                    'import json,os,subprocess,sys,time;from pathlib import Path;'
                    "child=subprocess.Popen([sys.executable,'-c','pass']);"
                    "p=Path('/proc')/str(child.pid)/'stat';"
                    "\nwhile p.read_text().rsplit(')',1)[1].split()[0]!='Z':time.sleep(.005)\n"
                    f"Path({str(ready)!r}).write_text(json.dumps({{'grandchild':child.pid}}));"
                    f"\nwhile not Path({str(release)!r}).exists():time.sleep(.005)\n"
                    'os._exit(7)'
                )
                parent = subprocess.Popen([sys.executable, '-c', program])
                wait_for(ready.exists)
                grandchild = json.loads(ready.read_text())['grandchild']
                root, start = os.getpid(), launcher._process_start(os.getpid())
                original = launcher._process_tree
                before = original(root, start)
                old = next(row for row in before if row['pid'] == grandchild)
                self.assertEqual(old['parent'], parent.pid)
                release.touch()
                wait_for(lambda: state(parent.pid) == 'Z')
                after = original(root, start)
                self.assertEqual(next(row for row in after if row['pid'] == grandchild)['parent'], root)
                # Valid interleaving: child was read before reparenting, parent after.
                mixed = [old if row['pid'] == grandchild else row for row in after]
                with mock.patch.object(launcher, '_process_tree', return_value=mixed):
                    launcher._finish_children(start, parent, grace=.2)
                self.assertEqual(parent.returncode, 7)
                self.assertIsNone(state(grandchild))
                with self.assertRaises(ChildProcessError):
                    os.waitpid(grandchild, os.WNOHANG)
        finally:
            if parent is not None:
                if parent.poll() is None:
                    parent.kill()
                parent.wait(timeout=5)
            if grandchild is not None:
                try:
                    os.waitpid(grandchild, os.WNOHANG)
                except ChildProcessError:
                    pass
            launcher._set_subreaper(previous)


if __name__ == '__main__':
    unittest.main()
