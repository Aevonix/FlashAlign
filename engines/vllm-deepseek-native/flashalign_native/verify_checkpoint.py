# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Verify pinned stock files, keeping resumable receipts outside model weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat


def fingerprint(path):
    path=Path(path).absolute()
    entry=path.lstat()
    if not (stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode)):
        raise ValueError('Checkpoint entry must be a regular file or resolved file symlink')
    resolved=path.resolve(strict=True)
    fd=os.open(resolved,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode):raise ValueError('Checkpoint entry must be a regular file')
        return {**{name:getattr(s,name) for name in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')},
                'resolved_path':str(resolved),
                'entry':{name:getattr(entry,name) for name in ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')},
                'link_text':os.readlink(path) if stat.S_ISLNK(entry.st_mode) else None}
    finally:os.close(fd)


def file_hash(path,expected=None):
    expected=fingerprint(path) if expected is None else expected
    digest=hashlib.sha256()
    descriptor=os.open(expected['resolved_path'],os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'rb') as stream:
        before=os.fstat(stream.fileno())
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        if any(getattr(before,name)!=expected[name] for name in fields):
            raise ValueError('Checkpoint target changed before hashing')
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        after=os.fstat(stream.fileno())
        if any(getattr(after,name)!=expected[name] for name in fields):
            raise ValueError('Checkpoint target changed during hashing')
    return digest.hexdigest()


def private_receipt(path):
    info=path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
        raise ValueError('Receipt parent must be an owned private directory')
    if path.exists() or path.is_symlink():
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
                raise ValueError('Receipt must be an owned private regular file')
            if info.st_size>1024*1024:raise ValueError('Oversized checkpoint receipt')
        finally:os.close(fd)


def save_receipt(path,value):
    private_receipt(path)
    temporary=path.with_name(path.name+'.'+secrets.token_hex(8)+'.tmp')
    fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def verify(directory,receipt_path,pins,*,fill=False,force=False,progress=None):
    root=Path(directory);receipt=Path(receipt_path).expanduser().absolute()
    if fill:receipt.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    private_receipt(receipt)
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    identity={'id':pins['model'],'revision':pins['revision']}
    if previous and previous.get('model')!=identity:raise ValueError('Receipt belongs to a different model revision')
    rows=previous.get('files',{})
    state={'model':identity,'complete':False,'files':dict(rows)}
    rehashed=0
    for name,expected in pins['metadata_sha256'].items():
        path=root/name;before=fingerprint(path)
        if file_hash(path,before)!=expected or fingerprint(path)!=before:
            raise ValueError('Checkpoint metadata differs from pinned model: '+name)
    for index,(name,expected) in enumerate(pins['shards'].items()):
        path=root/name;before=fingerprint(path)
        if before['st_size']!=expected['bytes']:raise ValueError('Checkpoint shard size differs: '+name)
        cached=rows.get(name,{})
        if force or cached.get('fingerprint')!=before or cached.get('sha256')!=expected['sha256']:
            if not fill:raise ValueError('Missing or stale full-hash receipt for shard: '+name)
            actual=file_hash(path,before)
            if actual!=expected['sha256'] or fingerprint(path)!=before:
                raise ValueError('Checkpoint shard failed pinned SHA-256: '+name)
            state['files'][name]={'fingerprint':before,'sha256':actual}
            rehashed+=1
            save_receipt(receipt,state)
        if progress is not None:progress(index+1,len(pins['shards']),rehashed)
    if set(state['files'])!=set(pins['shards']):raise ValueError('Unexpected checkpoint receipt entries')
    state['complete']=True
    if fill:save_receipt(receipt,state)
    elif not previous.get('complete'):raise ValueError('Checkpoint receipt is incomplete')
    return {'verified':True,'model':identity,'shards':len(pins['shards']),'rehashed_shards':rehashed}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--verify',action='store_true',help='Hash missing/stale shards and save resumable private receipts')
    p.add_argument('--full',action='store_true',help='Rehash every shard; implies --verify')
    args=p.parse_args()
    pins=json.loads((Path(__file__).parent/'model-pins.json').read_text())
    progress=(lambda count,total,rehashed:print(json.dumps({'checked_shards':count,'total_shards':total,'rehashed':rehashed}),flush=True)) if args.verify or args.full else None
    result=verify(args.model,args.receipt,pins,fill=args.verify or args.full,force=args.full,progress=progress)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
