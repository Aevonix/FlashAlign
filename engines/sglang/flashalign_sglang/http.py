# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""ASGI authentication around SGLang's existing generation endpoints."""
import json

from .gate import AUTH, EFFECTIVE, FieldError
from .runtime import registry

GENERATION = {'/generate','/v1/chat/completions','/v1/completions'}
DISCOVERY = {'/owner-mode/profiles','/v1/owner-mode/profiles'}
UTILITIES = {'/flush_cache','/freeze_gc'}


class Middleware:
    def __init__(self,app):
        self.app=app

    async def __call__(self,scope,receive,send):
        if scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        profiles,gate=registry()
        path=scope.get('path','').rstrip('/')
        async def reply(status,body):
            data=json.dumps(body,separators=(',',':')).encode()
            await send({'type':'http.response.start','status':status,'headers':[
                (b'content-type',b'application/json'),(b'content-length',str(len(data)).encode()),
                (EFFECTIVE,b'off')]})
            await send({'type':'http.response.body','body':data})
        if path in DISCOVERY and scope['method']=='GET':
            return await reply(200,{'data':[{'name':p.name,'model':p.model,'loaded':True} for p in profiles.values()]})
        if profiles and scope['method']=='POST' and any(x in path for x in ('update_weights','load_weights','load_lora','update_lora')):
            return await reply(409,{'error':{'message':'Restart the FlashAlign engine to replace model weights or profiles','type':'invalid_request_error'}})
        if profiles and scope['method']=='POST' and path not in GENERATION and path not in UTILITIES:
            return await reply(405,{'error':{'message':'This FlashAlign deployment exposes only its authenticated generation routes','type':'invalid_request_error'}})
        is_json=any(k.lower()==b'content-type' and b'application/json' in v.lower() for k,v in scope['headers'])
        if scope['method']!='POST' or (path not in GENERATION and not is_json):
            return await self.app(scope,receive,send)
        raw=bytearray()
        while True:
            event=await receive()
            if event['type']=='http.disconnect':return
            raw.extend(event.get('body',b''))
            if len(raw)>96*1024*1024:
                return await reply(413,{'error':{'message':'Request body exceeds limit','type':'invalid_request_error'}})
            if not event.get('more_body',False):break
        try:
            body=json.loads(raw)
            if not isinstance(body,dict):raise ValueError()
            headers=[value.decode('ascii','replace') for name,value in scope['headers'] if name.lower()==AUTH.encode()]
            selected=gate.select(bytes(raw),body,headers) if path in GENERATION else None
            body.pop('owner_mode',None);body.pop('owner_mode_scale',None)
            # Session reuse is a separate cache path, not qualified by cache_salt.
            if body.get('session_params') or body.get('session_id'):
                raise FieldError('FlashAlign SGLang does not support session-cache reuse')
            salt=body.get('cache_salt')
            if isinstance(salt,list):body['cache_salt']=[gate.cache_salt(selected,x) for x in salt]
            else:body['cache_salt']=gate.cache_salt(selected,salt)
            # Prevent native per-request logging from printing contact content.
            if path=='/generate':body['no_logs']=True
            cooked=json.dumps(body,separators=(',',':')).encode()
        except FieldError as error:
            return await reply(400,{'error':{'message':str(error),'type':'invalid_request_error','param':'owner_mode','code':'invalid_owner_mode'}})
        except (ValueError,UnicodeError):
            return await reply(400,{'error':{'message':'Invalid request body','type':'invalid_request_error'}})
        new_scope={**scope,'headers':[(k,v) for k,v in scope['headers'] if k.lower() not in (b'content-length',AUTH.encode())]+[
            (b'content-length',str(len(cooked)).encode())]}
        used=False
        async def next_body():
            nonlocal used
            if not used:
                used=True;return {'type':'http.request','body':cooked,'more_body':False}
            return await receive()
        async def tagged(event):
            if event['type']=='http.response.start':
                event={**event,'headers':[(k,v) for k,v in event.get('headers',[]) if k.lower()!=EFFECTIVE]+[
                    (EFFECTIVE,(selected.name if selected else 'off').encode())]}
            await send(event)
        return await self.app(new_scope,next_body,tagged)
