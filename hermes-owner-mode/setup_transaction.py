# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Stage managed configuration changes and recover interrupted local commits."""
from contextvars import ContextVar
from functools import wraps
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile

from .setup_runtime import SetupError

_CURRENT = ContextVar('flashalign_setup_transaction', default=None)
FILES = {'.hermes/config.yaml', '.openclaw/openclaw.json',
         '.config/owner-mode/approvals.json', '.config/owner-mode/upstream-api.key',
         '.config/owner-mode/openclaw-bridge.json', '.config/owner-mode/setup-result.json'}
TREES = {'.hermes/plugins/owner-mode', '.openclaw/extensions/owner-mode'}


def _json(path, value):
    temporary = path.with_name(path.name+'.new-'+secrets.token_hex(6))
    created = False
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync(path.parent)
    finally:
        if created:temporary.unlink(missing_ok=True)


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _fingerprint(path):
    if not os.path.lexists(path):return None
    if path.is_symlink():raise SetupError('Managed setup targets cannot be symlinks')
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else [path, *sorted(path.rglob('*'))]
    for item in paths:
        relative = str(item.relative_to(path)) if item != path else '.'
        info = item.lstat()
        if info.st_uid != os.getuid():
            raise SetupError('Managed setup files must be owned by the current user')
        digest.update(relative.encode()+b'\0'+str(stat.S_IMODE(info.st_mode)).encode()+b'\0')
        if item.is_symlink():digest.update(b'L'+os.readlink(item).encode())
        elif item.is_file():digest.update(b'F'+item.read_bytes())
        elif item.is_dir():digest.update(b'D')
        else:raise SetupError('Unsupported file in managed setup path')
    return digest.hexdigest()


def _copy(source, destination, kind):
    if kind == 'tree':shutil.copytree(source, destination, symlinks=True)
    else:shutil.copy2(source, destination)


def _install(source, target, kind, identifier):
    """Replace only the managed target; retain any displaced tree as a backup."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'tree':
        temporary = Path(tempfile.mkdtemp(prefix='.'+target.name+'.flashalign-new-',dir=target.parent))
    else:
        fd, name = tempfile.mkstemp(prefix='.'+target.name+'.flashalign-new-',dir=target.parent)
        os.close(fd);temporary = Path(name)
    displaced = target.with_name('.'+target.name+'.flashalign-before-'+identifier+'-'+secrets.token_hex(6))
    try:
        if kind == 'tree':shutil.copytree(source,temporary,symlinks=True,dirs_exist_ok=True)
        else:
            shutil.copy2(source,temporary)
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
        if kind == 'tree' and target.exists():
            if displaced.exists():raise SetupError('Managed setup backup path already exists')
            os.replace(target, displaced)
        os.replace(temporary, target)
        _sync(target.parent)
    finally:
        if temporary.exists():
            if temporary.is_dir():shutil.rmtree(temporary)
            else:temporary.unlink()


def _paths(home):
    state = home/'.config/owner-mode'
    return state, state/'setup-pending.json'


def _target(home, relative):
    target = home/relative
    if not target.parent.resolve().is_relative_to(home):
        raise SetupError('Setup recovery target parent escapes the selected home')
    if target.is_symlink():raise SetupError('Setup recovery target cannot be a symlink')
    return target


def pending_setup(home):
    return os.path.lexists(_paths(Path(home).resolve())[1])


def _lock(state):
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(state/'setup-transaction.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        os.close(fd);raise SetupError('Invalid setup transaction lock')
    try:fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise SetupError('Another FlashAlign setup is already running') from None
    return fd


def _load_pending(home, pending):
    info = pending.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1 or info.st_size > 1024*1024):
        raise SetupError('Invalid pending setup receipt')
    value = json.loads(pending.read_text())
    if not isinstance(value,dict):raise SetupError('Invalid pending setup receipt')
    identifier = value.get('id','')
    if (value.get('version') != 1 or value.get('home') != str(home)
            or len(identifier) != 24 or any(c not in '0123456789abcdef' for c in identifier)):
        raise SetupError('Pending setup receipt does not match this home')
    if not isinstance(value.get('entries'),list):raise SetupError('Invalid pending setup entries')
    for index,entry in enumerate(value['entries']):
        if not isinstance(entry,dict) or entry.get('kind') not in ('file','tree'):
            raise SetupError('Invalid pending setup entry')
        relative = entry.get('path')
        allowed = TREES if entry.get('kind') == 'tree' else FILES
        if (relative not in allowed or entry.get('backup')!=f'backup-{index}'
                or entry.get('staged')!=f'staged-{index}'):
            raise SetupError('Pending setup receipt contains an unsupported target')
    return value


def _rollback_unlocked(home, directory, journal):
    # Validate every target before changing any, preserving edits made after an interruption.
    for entry in journal['entries']:
        actual = _fingerprint(_target(home,entry['path']))
        allowed = (entry['before'], entry.get('after'))
        if entry['kind']=='tree':allowed += (None,)
        if actual not in allowed:
            raise SetupError('Setup recovery stopped because a managed path was edited: '+entry['path'])
        if entry['before'] is not None and _fingerprint(directory/entry['backup']) != entry['before']:
            raise SetupError('Setup before-image failed validation: '+entry['path'])
    for entry in reversed(journal['entries']):
        target = _target(home,entry['path']);actual = _fingerprint(target)
        if actual == entry['before']:continue
        if entry['before'] is None:
            if target.exists():
                if entry['kind'] == 'tree':shutil.rmtree(target)
                else:target.unlink()
        else:
            _install(directory/entry['backup'], target, entry['kind'], journal['id']+'-rollback')


def _approval_lock(path):
    fd=os.open(str(path)+'.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    info=os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid()
            or stat.S_IMODE(info.st_mode)!=0o600 or info.st_nlink!=1):
        os.close(fd);raise SetupError('Invalid approval-store lock')
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise SetupError('Approval settings are being updated; retry setup') from None
    return fd


def _rollback(home, directory, journal):
    fd=None
    if _CURRENT.get() is None and any(e['path']=='.config/owner-mode/approvals.json' for e in journal['entries']):
        fd=_approval_lock(home/'.config/owner-mode/approvals.json')
    try:_rollback_unlocked(home,directory,journal)
    finally:
        if fd is not None:os.close(fd)


def _recover_locked(home, state, pending):
    if not os.path.lexists(pending):return False
    journal = _load_pending(home, pending)
    directory = state/'setup-transactions'/journal['id']
    if directory.is_symlink() or not directory.is_dir() or not directory.resolve().is_relative_to(state.resolve()):
        raise SetupError('Pending setup transaction directory is invalid')
    if journal['status'] not in ('staging','committed','rolled_back','abandoned_staging'):
        _rollback(home, directory, journal)
        journal['status'] = 'rolled_back'
    elif journal['status'] == 'staging':journal['status'] = 'abandoned_staging'
    _json(directory/'transaction.json', journal)
    pending.unlink();_sync(state)
    return True


def recover_setup(home):
    home = Path(home).resolve();state,pending = _paths(home)
    if not os.path.lexists(pending):return False
    if not state.resolve().is_relative_to(home):raise SetupError('Setup state escapes the selected home')
    fd = _lock(state)
    try:return _recover_locked(home,state,pending)
    finally:os.close(fd)


class SetupTransaction:
    def __init__(self, home):
        self.home = Path(home).resolve();self.state,self.pending = _paths(self.home)
    def __enter__(self):
        if not self.state.resolve().is_relative_to(self.home):raise SetupError('Setup state escapes the selected home')
        self.lock = _lock(self.state)
        try:
            _recover_locked(self.home,self.state,self.pending)
            identifier = secrets.token_hex(12)
            self.directory = self.state/'setup-transactions'/identifier
            self.directory.mkdir(parents=True, mode=0o700)
            self.journal = {'version':1,'id':identifier,'home':str(self.home),'status':'staging','entries':[]}
            self.resource_locks=[]
            self._save();self.token = _CURRENT.set(self)
            return self
        except BaseException:
            os.close(self.lock);raise
    def _save(self):
        _json(self.directory/'transaction.json',self.journal)
        _json(self.pending,self.journal)
    def _stage(self, path, kind):
        path = Path(path).absolute()
        try:relative = path.relative_to(self.home).as_posix()
        except ValueError:raise SetupError('Setup target is outside the selected home') from None
        if relative not in (TREES if kind=='tree' else FILES):raise SetupError('Unsupported setup transaction target')
        if not path.parent.resolve().is_relative_to(self.home):raise SetupError('Setup target parent escapes the selected home')
        for entry in self.journal['entries']:
            if entry['path']==relative:return self.directory/entry['staged']
        if relative=='.config/owner-mode/approvals.json':
            self.resource_locks.append(_approval_lock(path))
        index=len(self.journal['entries']);before=_fingerprint(path)
        entry={'path':relative,'kind':kind,'before':before,'backup':f'backup-{index}','staged':f'staged-{index}'}
        if before is not None:
            _copy(path,self.directory/entry['backup'],kind)
            if _fingerprint(self.directory/entry['backup'])!=before or _fingerprint(path)!=before:
                raise SetupError('Setup target changed while its before-image was saved')
        staged=self.directory/entry['staged']
        if kind=='tree':staged.mkdir()
        elif before is not None:shutil.copy2(path,staged)
        self.journal['entries'].append(entry);self._save()
        return staged
    def stage_file(self,path):return self._stage(path,'file')
    def stage_tree(self,path):return self._stage(path,'tree')
    def __exit__(self, kind, value, trace):
        try:
            if kind is None:
                for entry in self.journal['entries']:
                    entry['after']=_fingerprint(self.directory/entry['staged'])
                    if entry['after'] is None:raise SetupError('Staged setup output is missing')
                    if _fingerprint(_target(self.home,entry['path']))!=entry['before']:
                        raise SetupError('Setup target changed before commit: '+entry['path'])
                self.journal['status']='committing';self._save()
                for entry in self.journal['entries']:
                    if _fingerprint(_target(self.home,entry['path']))!=entry['before']:
                        raise SetupError('Setup target changed during commit: '+entry['path'])
                    _install(self.directory/entry['staged'],_target(self.home,entry['path']),entry['kind'],self.journal['id'])
                self.journal['status']='committed';self._save()
            else:self.journal['status']='abandoned_staging';self._save()
        except BaseException:
            if self.journal['status']=='committing':
                try:
                    _rollback(self.home,self.directory,self.journal)
                    self.journal['status']='rolled_back';self._save()
                except BaseException:
                    self.journal['status']='rollback_required';self._save();raise
            raise
        finally:
            if self.journal['status'] in ('committed','rolled_back','abandoned_staging'):
                self.pending.unlink(missing_ok=True);_sync(self.state)
            _CURRENT.reset(self.token)
            for fd in self.resource_locks:os.close(fd)
            os.close(self.lock)
        return False


def current_transaction():return _CURRENT.get()


def transactional_setup(function):
    @wraps(function)
    def wrapped(home,*args,**kwargs):
        if current_transaction() is not None:return function(home,*args,**kwargs)
        with SetupTransaction(home):return function(home,*args,**kwargs)
    return wrapped
