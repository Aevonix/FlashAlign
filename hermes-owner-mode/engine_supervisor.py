# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Dedicated Linux subreaper for one managed engine command; no model logic."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid,include_zombies=False):
    try:
        path=Path('/proc')/str(pid);parts=(path/'stat').read_text().rsplit(')',1)[1].split()
        if parts[0]=='Z' and not include_zombies:return None
        return {'pid':pid,'start':parts[19],'uid':path.stat().st_uid}
    except (OSError,ValueError,IndexError):return None


def _scan_descendants(roots):
    """Compatibility fallback when the kernel does not expose task children."""
    snapshot={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            parts=path.read_text().rsplit(')',1)[1].split();pid=int(path.parent.name)
            snapshot[pid]={'identity':{'pid':pid,'start':parts[19],'uid':path.parent.stat().st_uid},'parent':int(parts[1]),'state':parts[0]}
        except (OSError,ValueError,IndexError):continue
    selected=set()
    for root in roots:
        if (isinstance(root,dict) and snapshot.get(root.get('pid'),{}).get('identity')==root
                and root.get('uid')==os.getuid() and identity(root['pid'],True)==root):selected.add(root['pid'])
    while True:
        added={pid for pid,row in snapshot.items() if row['parent'] in selected
               and row['identity']['uid']==os.getuid()
               and int(row['identity']['start'])>=int(snapshot[row['parent']]['identity']['start'])}
        enlarged=selected|added
        if enlarged==selected:break
        selected=enlarged
    return [snapshot[pid]['identity'] for pid in sorted(selected)
            if snapshot[pid]['state']!='Z' and identity(pid)==snapshot[pid]['identity']]


def _process(pid):
    try:
        path=Path('/proc')/str(pid)
        parts=(path/'stat').read_text().rsplit(')',1)[1].split()
        return {'identity':{'pid':pid,'start':parts[19],'uid':path.stat().st_uid},
                'parent':int(parts[1]),'state':parts[0]}
    except (OSError,ValueError,IndexError):return None


def _child_ids(record):
    """Read every thread's children. None requests the compatibility fallback."""
    if identity(record['pid'],True)!=record:return []
    task=Path('/proc')/str(record['pid'])/'task'
    try:
        threads=list(task.iterdir())
    except OSError:
        return None if identity(record['pid'],True)==record else []
    children=set()
    for thread in threads:
        if not thread.name.isdigit():continue
        try:
            children.update(int(value) for value in (thread/'children').read_text().split())
        except FileNotFoundError:
            # A thread may exit during enumeration. A live task without the
            # children file instead means this kernel interface is unavailable.
            if thread.exists() and identity(record['pid'],True)==record:return None
        except (OSError,ValueError):return None
    return children


def descendants(roots):
    """Follow verified task children without scanning unrelated host processes.

    Birth identities and parent links are rechecked around traversal. Previously
    captured roots remain valid after reparenting; PGID/SID changes are irrelevant.
    """
    roots=list(roots)
    selected={}
    pending=[]
    for record in roots:
        if (isinstance(record,dict) and type(record.get('pid')) is int
                and record.get('uid')==os.getuid() and identity(record['pid'],True)==record):
            selected[record['pid']]=record
            pending.append(record)
    visited=set()
    while pending:
        parent=pending.pop()
        birth=tuple(parent[key] for key in ('pid','start','uid'))
        if birth in visited:continue
        visited.add(birth)
        if identity(parent['pid'],True)!=parent:continue
        children=_child_ids(parent)
        if children is None:
            return _scan_descendants([*roots,*selected.values()])
        if identity(parent['pid'],True)!=parent:continue
        for pid in children:
            child=_process(pid)
            if (child is None or child['parent']!=parent['pid']
                    or child['identity']['uid']!=parent['uid']
                    or int(child['identity']['start'])<int(parent['start'])):continue
            # Reject a reused PID or a changed parent relation before retaining
            # the child as an owned root for the next traversal step.
            again=_process(pid)
            if (identity(parent['pid'],True)!=parent or again is None
                    or again['identity']!=child['identity'] or again['parent']!=parent['pid']):continue
            record=child['identity']
            selected[pid]=record
            pending.append(record)
    return [record for pid,record in sorted(selected.items()) if identity(pid)==record]


def signal_identity(record,signum):
    if record.get('uid')!=os.getuid() or identity(record.get('pid'))!=record:return False
    try:
        fd=os.pidfd_open(record['pid'])
        try:
            if identity(record['pid'])!=record:return False
            signal.pidfd_send_signal(fd,signum);return True
        finally:os.close(fd)
    except OSError:return False


def set_subreaper():
    fd=os.pidfd_open(os.getpid());os.close(fd)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot supervise managed engine orphans')


def save(path,value):
    temporary=path.with_name(path.name+'.new-'+str(os.getpid()))
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as out:json.dump(value,out,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
    os.replace(temporary,path)


def reap(child):
    if child is not None and child.poll() is None:return False
    while True:
        try:
            pid,_=os.waitpid(-1,os.WNOHANG)
            if pid==0:return False
        except ChildProcessError:return True


def finish(root,child,path,base):
    """Retain the subreaper until the kernel confirms no children remain."""
    while True:
        for signum in (signal.SIGTERM,signal.SIGKILL):
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if reap(child):return
                rows=[row for row in descendants([root]) if row!=root]
                try:save(path,{**base,'status':'stopping','owned_members':rows,'all_children_reaped':False})
                except OSError:pass  # Receipt I/O must never abandon living children.
                for row in rows:signal_identity(row,signum)
                time.sleep(.05)
        # An uninterruptible child may take longer. Keep the anchor alive and
        # let the external bounded stop report incomplete instead of orphaning it.
        try:save(path,{**base,'status':'cleanup_incomplete','all_children_reaped':False})
        except OSError:pass
        time.sleep(.2)


def supervise(command,path):
    root=identity(os.getpid(),True)
    if root is None:raise RuntimeError('Supervisor identity unavailable')
    set_subreaper();stopping=False;child=None;code=1
    def stop(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    base={'version':1,'supervisor':root,'command':None}
    save(path,{**base,'status':'starting','all_children_reaped':False})
    try:
        if not stopping:
            child=subprocess.Popen(command)
            base['command']=identity(child.pid,True)
            if base['command'] is None:raise RuntimeError('Managed command identity unavailable')
            save(path,{**base,'status':'running','all_children_reaped':False})
            while child.poll() is None and not stopping:time.sleep(.1)
            code=child.returncode if child.returncode is not None else 1
    finally:
        finish(root,child,path,base)
        save(path,{**base,'status':'stopped','all_children_reaped':True,'exit_code':code})
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state',type=Path,required=True);parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('A managed command is required')
    return supervise(command,args.state)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        print('Managed engine supervisor failed; inspect its private state receipt.',file=sys.stderr);raise SystemExit(1)
