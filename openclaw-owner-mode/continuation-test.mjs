// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import {createInterface} from 'node:readline';
import {createAdapter} from './index.mjs';

setTimeout(()=>{throw new Error('Continuation contract test exceeded20seconds');},20000).unref();
const [url,keyFile]=process.argv.slice(2);
const hooks=new Map();let provider,lifecycle;
const api={config:{session:{dmScope:'per-account-channel-peer'},agents:{defaults:{models:{
  'owner-mode/test':{params:{temperature:0,maxTokens:4096}},
}}}},agent:{events:{registerAgentEventSubscription(value){lifecycle=value.handle;}}},
  on(name,fn){hooks.set(name,fn);},registerProvider(value){provider=value;}};
createAdapter(api,{bridgeUrl:url,controlKeyFile:keyFile,model:{id:'test',name:'test',
  contextWindow:65536,maxTokens:4096,input:['text']}});
const parent={runId:'parent',sessionKey:'parent-session',channelId:'whatsapp',channel:'whatsapp',
  senderId:'+15550000011',accountId:'default',messageId:'message',trigger:'user',modelProviderId:'owner-mode'};
const child={runId:'child',sessionKey:'child-session',modelProviderId:'owner-mode'};
const endContext={runId:child.runId,childSessionKey:child.sessionKey,requesterSessionKey:parent.sessionKey};
const endEvent={runId:child.runId,targetSessionKey:child.sessionKey,targetKind:'subagent',reason:'completed',outcome:'ok'};
async function resolve(ctx){return (await hooks.get('before_model_resolve')({},ctx)).modelOverride;}
async function send(id){
  const selected=typeof id==='string'?provider.resolveDynamicModel({modelId:id}):id;assert.ok(selected);
  return provider.wrapStreamFn({streamFn:async(model,ctx,options)=>{
    const body={model:'test',messages:[{role:'user',content:'Native test content.'}]};
    const final=await options?.onPayload?.(body,model)??body;
    const response=await fetch(model.baseUrl+'/chat/completions',{method:'POST',
      headers:{'Content-Type':'application/json',...model.headers},body:JSON.stringify(final)});
    assert.equal(response.status,200);return response.headers.get('X-Owner-Mode-Effective');
  }})(selected,{},{});
}
await hooks.get('message_received')({content:'Native test content.',senderId:parent.senderId,
  sessionKey:parent.sessionKey,messageId:parent.messageId},parent);
await hooks.get('before_dispatch')({content:'Native test content.',channel:parent.channel,isGroup:false,
  senderId:parent.senderId,sessionKey:parent.sessionKey,messageId:parent.messageId},parent);
const parentAlias=await resolve(parent);assert.equal(await send(parentAlias),'owner-test');
const heldParent=provider.resolveDynamicModel({modelId:parentAlias});
await hooks.get('agent_end')({success:true},parent);
assert.equal(await send(await resolve(parent)),'owner-test');
await lifecycle({runId:parent.runId,sessionKey:'other-session',stream:'lifecycle',data:{phase:'end'}});
await lifecycle({runId:'other-run',sessionKey:parent.sessionKey,stream:'lifecycle',data:{phase:'end'}});
assert.equal(await send(await resolve(parent)),'owner-test');
async function spawn(ctx){
  await hooks.get('before_tool_call')({toolName:'sessions_spawn'},parent);
  await hooks.get('subagent_spawned')({runId:ctx.runId,childSessionKey:ctx.sessionKey,mode:'run'},
    {runId:ctx.runId,childSessionKey:ctx.sessionKey,requesterSessionKey:parent.sessionKey});
  await hooks.get('after_tool_call')({toolName:'sessions_spawn'},parent);
}
await spawn(child);
const first=await resolve(child);assert.equal(await send(first),'owner-test');
const heldChild=provider.resolveDynamicModel({modelId:first});
await hooks.get('agent_end')({success:true},child);
const second=await resolve(child);assert.notEqual(second,'test');assert.equal(await send(second),'owner-test');
assert.equal(provider.prepareExtraParams({config:api.config,modelId:second,extraParams:{}}).temperature,0);
assert.equal((await resolve({...child,runId:'unapproved-new-run'})),'test');
assert.equal((await resolve({...child,sessionKey:'different-session'})),'test');
await hooks.get('subagent_ended')({...endEvent,runId:'other-run'},endContext);
await hooks.get('subagent_ended')({...endEvent,targetSessionKey:'other-session'},endContext);
await hooks.get('subagent_ended')(endEvent,{...endContext,requesterSessionKey:'other-parent'});
await hooks.get('subagent_ended')({...endEvent,targetKind:'acp'},endContext);
assert.equal(await send(await resolve(child)),'owner-test');
await hooks.get('subagent_ended')(endEvent,endContext);
assert.equal(await resolve(child),'test');
assert.equal(await send('test'),'off');
assert.equal(provider.resolveDynamicModel({modelId:first}),undefined);
assert.equal(await send(heldChild),'off'); // Previously resolved model has lost its live capability.
const revoked={...child,runId:'revoked-child',sessionKey:'revoked-child-session'};
await spawn(revoked);const revokedAlias=await resolve(revoked);assert.equal(await send(revokedAlias),'owner-test');
await hooks.get('agent_end')({success:true},revoked);
await lifecycle({runId:parent.runId,sessionKey:parent.sessionKey,stream:'lifecycle',data:{phase:'end'}});
assert.equal(await send(heldParent),'off');
assert.equal(await resolve(parent),'test');
assert.equal(await send(await resolve(revoked)),'owner-test');
const lines=createInterface({input:process.stdin});const next=lines[Symbol.asyncIterator]();
console.log('REVOKE_CONTACT');assert.equal((await next.next()).value,'revoked');lines.close();
assert.equal(await send(await resolve(revoked)),'off');
assert.equal(await resolve({...parent,runId:'completed-new-run',trigger:'system',senderId:undefined}),'test');
console.log('OPENCLAW_CONTINUATION_CONTRACT_OK');
