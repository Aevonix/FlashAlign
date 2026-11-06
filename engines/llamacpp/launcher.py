# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Launcher for the pinned native FlashAlign llama.cpp integration."""
import argparse
from contextlib import nullcontext
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

from auth_contract import authenticate_header, read_signing_key
from profile_gguf import convert, read_profile
from flashalign_replay import ReplayStore, deployment_replay_store
from lifecycle import NativeControl, NativeLifecycle
from flashalign_lifecycle import LifecycleError


class FieldError(ValueError):
    pass


def load_key(path):
    return read_signing_key(path)


class Gate:
    def __init__(self,key,names,clock=time.time,capacity=100000,replay_path=None,*,replay_store=None):
        self.key=key;self.names=names if callable(names) else set(names);self.clock=clock;self.capacity=capacity
        self.used={};self.lock=threading.Lock()
        self.store=replay_store if replay_store is not None else (ReplayStore(replay_path,key,capacity=capacity) if replay_path is not None else None)
    def select(self,data,body,headers):
        mode=body.get('owner_mode','off')
        if mode is None: mode='off'
        scale=body.get('owner_mode_scale',1.)
        if scale is None: scale=1.
        if not isinstance(mode,str): raise FieldError('owner_mode must be "off" or an owner-mode profile name')
        if type(scale) not in (float,int) or not math.isfinite(scale) or not 0<scale<=2:
            raise FieldError('owner_mode_scale must be a number in (0, 2]')
        if mode=='off': return 'off',1.
        names = self.names() if callable(self.names) else self.names
        if not names: raise FieldError('this server was started without owner-mode profiles: owner_mode must be "off"')
        now=self.clock()
        auth=authenticate_header(self.key,data,headers,now=now)
        if auth is None: return 'off',1.
        nonce,ts=auth.nonce,auth.timestamp
        if self.store is not None:
            if not self.store.consume(nonce,ts+61,now=now):return 'off',1.
        else:
            with self.lock:
                try:
                    self.used={n:until for n,until in self.used.items() if until>=now}
                    if nonce in self.used or len(self.used)>=self.capacity: return 'off',1.
                    self.used[nonce]=ts+61
                except MemoryError: return 'off',1.
        if mode not in names: raise FieldError('owner_mode names no profile this server loaded')
        return mode,float(scale)


def handler(gate,profiles,backend_port,runtime=None):
    ids={p['name']:i for i,p in enumerate(profiles)}
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def respond(self,status,value,effective=None):
            raw=json.dumps(value).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
            if effective is not None:self.send_header('X-Owner-Mode-Effective',effective)
            self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            path=self.path.split('?',1)[0].rstrip('/')
            if path in ('/owner-mode/profiles','/v1/owner-mode/profiles'):
                return self.respond(200,{'data':runtime.profiles() if runtime else [
                    {'name':p['name'],'model':p['model'],'loaded':True} for p in profiles]})
            if runtime and path == '/v1/flashalign/status':
                return self.respond(200,runtime.safe_status())
            if runtime and path == '/v1/flashalign/capabilities':
                return self.respond(200,{'schema':'flashalign.capabilities.v1','hot_loading':True,
                                         'execution_attestation':False,'status':True})
            if path not in ('/health','/v1/health','/v1/models','/metrics'):
                return self.respond(404,{'error':'not found'})
            self.forward('GET',self.path,None,None)
        def do_POST(self):
            if self.path.split('?',1)[0].rstrip('/') not in ('/v1/chat/completions','/v1/completions'):
                return self.respond(404,{'error':'not found'})
            try:
                if self.headers.get_all('Transfer-Encoding'): raise ValueError('Chunked request bodies are unsupported')
                lengths=self.headers.get_all('Content-Length',[])
                if len(lengths)!=1: raise ValueError('Exactly one Content-Length is required')
                size=int(lengths[0])
                if not 0<size<=32*1024*1024: raise ValueError('Invalid request size')
                data=self.rfile.read(size)
                if len(data)!=size: raise ValueError('Incomplete body')
                body=json.loads(data)
                if not isinstance(body,dict): raise ValueError('Request must be a JSON object')
                selected,scale=gate.select(data,body,self.headers.get_all('X-Owner-Mode-Auth',[]))
                body.pop('owner_mode',None);body.pop('owner_mode_scale',None)
                body.pop('lora',None)
                # Always override raw adapter selection. The internal listener is loopback-only.
                with runtime.request_scope() if runtime else nullcontext():
                    if runtime and selected != 'off':
                        with runtime.pin(selected) as (index, generation, profile_digest):
                            body['lora']=[{'id':index,'scale':scale}]
                            self.forward('POST',self.path,json.dumps(body,separators=(',',':')).encode(),selected,
                                         generation=generation,profile_digest=profile_digest)
                    else:
                        if runtime and runtime.manager.status()['admission_fenced']:
                            raise LifecycleError('Admission is fenced')
                        body['lora']=[{'id':ids[selected] if selected!='off' else 0,'scale':scale if selected!='off' else 0.}]
                        self.forward('POST',self.path,json.dumps(body,separators=(',',':')).encode(),selected)
            except FieldError as error:
                self.respond(400,{'error':{'code':'invalid_owner_mode','message':str(error),'param':'owner_mode'}})
            except (ValueError,UnicodeError,json.JSONDecodeError) as error:
                self.respond(400,{'error':{'code':'invalid_request','message':str(error)}})
            except LifecycleError:
                self.respond(503,{'error':{'code':'profile_unavailable','message':'Profile admission is temporarily unavailable'}})
        def forward(self,method,path,data,effective,*,generation=None,profile_digest=None):
            connection=http.client.HTTPConnection('127.0.0.1',backend_port,timeout=1800)
            started=False
            try:
                connection.request(method,path,body=data,headers={'Content-Type':'application/json'} if data is not None else {})
                response=connection.getresponse()
                self.send_response(response.status)
                for name,value in response.getheaders():
                    if name.lower() not in ('connection','transfer-encoding','x-owner-mode-effective','content-length'):
                        self.send_header(name,value)
                if effective is not None:self.send_header('X-Owner-Mode-Effective',effective)
                if generation is not None:self.send_header('X-FlashAlign-Generation',generation)
                if profile_digest is not None:self.send_header('X-FlashAlign-Profile-Digest',profile_digest)
                self.send_header('Connection','close');self.end_headers();self.close_connection=True;started=True
                while chunk:=response.read1(65536):self.wfile.write(chunk);self.wfile.flush()
            except (ConnectionError,OSError,http.client.HTTPException):
                if not started:self.respond(502,{'error':{'message':'Native inference service unavailable'}})
                self.close_connection=True
            finally:connection.close()
    return Handler


def speculative_arguments(extra, draft_model, architecture, metadata):
    """Preserve explicit speculation and require a qualified method for this graph."""
    legacy={'--draft','--draft-n','--draft-max','--draft-min','--draft-p-min',
            '--spec-ngram-size-n','--spec-ngram-size-m','--spec-ngram-min-hits'}
    if any(value.split('=',1)[0] in legacy for value in extra):
        raise ValueError('Use current method-specific speculative options; legacy draft/ngram options are not supported by this engine')
    if '--no-drafts' in extra:
        raise ValueError('Speculative decoding cannot be disabled by this launcher')
    spec=[]
    for i,value in enumerate(extra):
        if value=='--spec-type':
            if i+1>=len(extra) or extra[i+1].startswith('-'):
                raise ValueError('Missing speculative decoding method')
            spec.extend(extra[i+1].split(','))
        elif value.startswith('--spec-type='):
            spec.extend(value.split('=',1)[1].split(','))
    by_architecture=metadata.get('speculative_types_by_model_architecture')
    allowed=(by_architecture.get(architecture,[]) if by_architecture is not None
             else metadata.get('speculative_types_qualified',[]))
    draftless=metadata.get('draftless_speculative_types',[])
    result=list(extra)
    if not spec:
        method=metadata.get('default_speculative_type_by_model_architecture',{}).get(architecture)
        if draft_model and (method is None or method in draftless):
            method=next((value for value in allowed if value not in draftless),None)
        if method is not None:
            result.extend(['--spec-type',method]);spec=[method]
        elif draft_model:
            raise ValueError('No qualified speculative method uses the supplied draft model')
    if any(value not in allowed for value in spec):
        raise ValueError('The speculative method is not qualified for this model; existing speculation will not be replaced or disabled')
    if any(value not in draftless for value in spec) and not draft_model:
        raise ValueError('The selected speculative method requires an explicit draft model')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--gguf-python',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--profile',type=Path,action='append',required=True)
    p.add_argument('--key-file',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--admin-socket',type=Path)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8030)
    p.add_argument('--draft-model',type=Path);p.add_argument('--engine-args',type=Path)
    p.add_argument('--profile-memory-mib',type=int,default=1024)
    a=p.parse_args()
    key=load_key(a.key_file);a.state.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not 1 <= a.profile_memory_mib <= 65536:raise ValueError('profile-memory-mib must be in 1..65536')
    profiles=[]
    for directory in a.profile:
        meta,_,digest=read_profile(directory)
        if meta['name'] in [x['name'] for x in profiles]:raise ValueError('Duplicate profiles')
        if profiles and meta['model']!=profiles[0]['model']:
            raise ValueError('All profiles must target the same pinned model')
        profiles.append(meta)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));internal=sock.getsockname()[1]
    extra=json.loads(a.engine_args.read_text()) if a.engine_args else []
    if not isinstance(extra,list) or any(not isinstance(x,str) for x in extra):raise ValueError('engine-args must contain an argument array')
    reserved={'--host','--port','--model','-m','--lora','--lora-scaled','--lora-init-without-apply','--slot-save-path',
              '--model-draft','--spec-draft-model','-md','--sleep-idle-seconds'}
    if any(x.split('=',1)[0] in reserved for x in extra):raise ValueError('engine-args cannot override managed model/listener/adapter isolation')
    sys.path.insert(0,str(a.gguf_python))
    import gguf
    architecture=gguf.GGUFReader(a.model,'r').get_field('general.architecture').contents()
    metadata=json.loads((Path(__file__).resolve().parent/'backend.json').read_text())
    extra=speculative_arguments(extra,a.draft_model,architecture,metadata)
    argv=[str(a.binary.resolve()),'--model',str(a.model.resolve()),'--host','127.0.0.1','--port',str(internal),'--lora-init-without-apply','--alias',profiles[0]['model']['id'],'--jinja',*extra]
    if a.draft_model:argv.extend(['--model-draft',str(a.draft_model.resolve())])
    log_path=a.state/'engine.log'
    fd=os.open(log_path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    child=None;runtime=None
    internal_token=secrets.token_hex(32)
    def stop(signum, frame): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        with os.fdopen(fd,'wb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                                   env={**os.environ,'FLASHALIGN_INTERNAL_ADMIN_TOKEN':internal_token})
            deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError('Native engine failed to start; see private engine.log')
                try:
                    c=http.client.HTTPConnection('127.0.0.1',internal,timeout=2);c.request('GET','/health');r=c.getresponse();ready=r.status==200;r.read();c.close()
                    if ready:break
                except OSError:pass
                time.sleep(.5)
            else:raise TimeoutError('Native engine startup timed out')
            runtime=NativeLifecycle(a.state,control=NativeControl(internal,internal_token),model=a.model,
                                    profiles=a.profile,gguf_python=a.gguf_python,
                                    resource_limit=a.profile_memory_mib*1024*1024, admin_socket=a.admin_socket)
            gate=Gate(key,runtime.names,
                      replay_store=deployment_replay_store(key,a.state/'replay.sqlite3'))
            server=ThreadingHTTPServer((a.host,a.port),handler(gate,profiles,internal,runtime))
            print(json.dumps({'status':'ready','profiles':[x['name'] for x in profiles],'backend':'llamacpp','host':a.host,'port':a.port}),flush=True)
            try:server.serve_forever(poll_interval=.5)
            finally:server.server_close()
    finally:
        if runtime is not None:runtime.close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
