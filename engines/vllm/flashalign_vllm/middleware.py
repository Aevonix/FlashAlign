# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""ASGI boundary for exact-body authentication and vLLM request metadata."""
import json
import os

from .core import FieldError, Gate, cache_key, load_key, load_profiles
from .flashalign_replay import deployment_replay_store


class OwnerMiddleware:
    def __init__(self, app):
        from .plugin import REGISTERED
        if not REGISTERED:
            raise RuntimeError('FlashAlign worker plugin must be enabled before the API starts')
        self.app = app
        self.profiles = load_profiles(json.loads(os.environ['FLASHALIGN_PROFILES']))
        key = load_key(os.environ['FLASHALIGN_KEY_FILE'])
        self.gate = Gate(self.profiles, key, replay_store=deployment_replay_store(key))
        self.model = os.environ['FLASHALIGN_MODEL_ID']

    async def reply(self, send, status, body, effective='off'):
        encoded = json.dumps(body,separators=(',',':')).encode()
        await send({'type':'http.response.start','status':status,'headers':[
            (b'content-type',b'application/json'),(b'content-length',str(len(encoded)).encode()),
            (b'x-owner-mode-effective',effective.encode())]})
        await send({'type':'http.response.body','body':encoded})

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return
        if scope['type'] != 'http':
            return await self.app(scope,receive,send)
        path = scope.get('path','')
        if scope.get('method') == 'GET' and path == '/v1/owner-mode/profiles':
            return await self.reply(send,200,{'data':[{'name':p.name,'model':p.meta['model'],'loaded':True}
                                                       for p in self.profiles.values()]})
        if scope.get('method') != 'POST':
            return await self.app(scope,receive,send)
        routes=('/v1/chat/completions','/v1/completions')
        if path not in routes:
            return await self.reply(send,400,{'error':{'code':'invalid_owner_mode',
                'message':'This FlashAlign server exposes Chat Completions and Completions only'}})
        chunks=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            block=message.get('body',b'');chunks.append(block);size+=len(block)
            if size>96*1024*1024:
                return await self.reply(send,413,{'error':{'message':'Request body exceeds limit'}})
            if not message.get('more_body'):break
        raw=b''.join(chunks)
        try:body=json.loads(raw)
        except (UnicodeError,ValueError):
            return await self.reply(send,400,{'error':{'message':'Invalid JSON request'}})
        if not isinstance(body,dict):
            return await self.reply(send,400,{'error':{'message':'Request must be an object'}})
        headers=[v.decode('ascii','replace') for k,v in scope.get('headers',[]) if k.lower()==b'x-owner-mode-auth']
        try:
            selection=self.gate.authorize(raw,body,headers) if path in routes else None
            if selection is not None and ('prompt_embeds' in body or 'prompt_embeddings' in body):
                raise FieldError('Owner mode does not accept precomputed input embeddings')
        except FieldError as error:
            return await self.reply(send,400,{'error':{'code':'invalid_owner_mode','type':'invalid_request_error',
                                                     'param':'owner_mode','message':str(error)}})
        extra=body.get('vllm_xargs') or {}
        if not isinstance(extra,dict):
            return await self.reply(send,400,{'error':{'message':'vllm_xargs must be an object'}})
        # Clients cannot forge worker metadata through any public endpoint.
        extra={k:v for k,v in extra.items() if not k.startswith('_flashalign_')}
        body.pop('owner_mode',None);body.pop('owner_mode_scale',None)
        # Even caller-supplied salts are wrapped in the effective namespace.
        body['cache_salt']=cache_key(selection,self.profiles,self.model,body.get('cache_salt'))
        if path in routes:
            if selection is not None:
                extra.update(_flashalign_profile=selection.name,_flashalign_digest=selection.digest,
                             _flashalign_scale=selection.scale)
        if extra:body['vllm_xargs']=extra
        else:body.pop('vllm_xargs',None)
        data=json.dumps(body,separators=(',',':')).encode()
        consumed=False
        async def updated_receive():
            nonlocal consumed
            if not consumed:
                consumed=True;return {'type':'http.request','body':data,'more_body':False}
            return await receive()
        effective='off' if selection is None else selection.name
        async def updated_send(message):
            if message['type']=='http.response.start':
                message=dict(message)
                message['headers']=[(k,v) for k,v in message.get('headers',[]) if k.lower()!=b'x-owner-mode-effective']
                message['headers'].append((b'x-owner-mode-effective',effective.encode()))
            await send(message)
        scope=dict(scope)
        scope['headers']=[(k,v) for k,v in scope.get('headers',[]) if k.lower() not in (b'content-length',b'x-owner-mode-auth')]
        scope['headers'].append((b'content-length',str(len(data)).encode()))
        await self.app(scope,updated_receive,updated_send)
