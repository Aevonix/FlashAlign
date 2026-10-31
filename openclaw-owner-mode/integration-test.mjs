// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import { createAdapter } from './index.mjs';

const [url, keyFile] = process.argv.slice(2);
const hooks = new Map(); let provider;
const api = {config:{session:{dmScope:'per-account-channel-peer'},agents:{defaults:{models:{
  'owner-mode/test':{params:{temperature:0,topP:.9,maxTokens:256}},
}}}},on(name, fn) {hooks.set(name, fn);}, registerProvider(value) {provider = value;}};
const adapter = createAdapter(api, {bridgeUrl:url, controlKeyFile:keyFile,
  model:{id:'test',name:'test',contextWindow:32768,maxTokens:1024,
    input:['text'],cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}});
const own='Please answer my question.';
const parent={runId:'oc-parent',sessionKey:'oc-session',channelId:'whatsapp',senderId:'+15550000011',
  channel:'whatsapp',trigger:'user',accountId:'default',modelProviderId:'owner-mode',messageId:'parent-message'};

async function nativeIngress(fields, context) {
  const ingressContext={sessionKey:context.sessionKey,channelId:context.channel,
    accountId:context.accountId,senderId:context.senderId,messageId:context.messageId};
  await hooks.get('message_received')({content:fields.content || '',from:context.senderId,
    senderId:context.senderId,messageId:context.messageId,sessionKey:context.sessionKey,
    media:fields.media,originalMedia:fields.originalMedia,location:fields.location,
    metadata:fields.metadata,providerUpdate:fields.providerUpdate},ingressContext);
  await hooks.get('before_dispatch')({...fields,channel:context.channel,senderId:context.senderId,
    messageId:context.messageId,sessionKey:context.sessionKey},ingressContext);
}
await nativeIngress({content:own,isGroup:false},parent);
const resolved=await hooks.get('before_model_resolve')({prompt:own},parent);
assert.notEqual(resolved.modelOverride,'test');
const configBefore=JSON.stringify(api.config);
const canonicalParams=provider.prepareExtraParams({config:api.config,modelId:resolved.modelOverride,extraParams:{}});
assert.equal(canonicalParams.temperature,0);assert.equal(canonicalParams.topP,.9);
assert.equal(canonicalParams.maxTokens,undefined);
assert.equal(JSON.stringify(api.config),configBefore);
const model=provider.resolveDynamicModel({modelId:resolved.modelOverride});
async function directSend(model, body) {
  const response=await fetch(model.baseUrl+'/chat/completions',{method:'POST',
    headers:{'Content-Type':'application/json',...model.headers},
    body:JSON.stringify(body)});
  assert.equal(response.status,200);return response.headers.get('x-owner-mode-effective');
}
async function send(model, content) {
  return provider.wrapStreamFn({streamFn:async (selected, context, options) => {
    const body={model:'test',messages:[{role:'user',content}]};
    const final=await options?.onPayload?.(body,selected) ?? body;
    return directSend(selected,final);
  }})(model,{},{});
}
assert.equal(await directSend(model,{model:'test',messages:[{role:'user',content:own}]}),'off');
assert.equal(await send(model,own),'owner-test');
assert.equal(await adapter.toolEnabled('oc-parent','sessions_spawn'),true);
await hooks.get('before_tool_call')({toolName:'sessions_spawn'},parent);
const child={runId:'oc-child',sessionKey:'oc-child-session',modelProviderId:'owner-mode',
  inputProvenance:{kind:'inter_session',sourceSessionKey:'oc-session',sourceTool:'sessions_spawn'}};
await hooks.get('subagent_spawned')({runId:child.runId,childSessionKey:child.sessionKey,mode:'run'},
  {runId:child.runId,childSessionKey:child.sessionKey,requesterSessionKey:parent.sessionKey});
// The real child model hook has no inputProvenance and starts after the tool returns.
const nativeChild={...child};delete nativeChild.inputProvenance;
await hooks.get('after_tool_call')({toolName:'sessions_spawn'},parent);
const childModel=await hooks.get('before_model_resolve')({prompt:'A child task'},nativeChild);
assert.equal(await send(provider.resolveDynamicModel({modelId:childModel.modelOverride}),'A child task'),'owner-test');
await hooks.get('subagent_spawned')({runId:'late-child',childSessionKey:'late-session',mode:'run'},
  {runId:'late-child',childSessionKey:'late-session',requesterSessionKey:parent.sessionKey});
assert.equal((await hooks.get('before_model_resolve')({},
  {runId:'late-child',sessionKey:'late-session',modelProviderId:'owner-mode'})).modelOverride,'test');
await hooks.get('before_tool_call')({toolName:'sessions_spawn'},parent);
const simultaneous={...parent,runId:'other-parent',messageId:'other-message'};
await nativeIngress({content:own,isGroup:false},simultaneous);
const simultaneousModel=await hooks.get('before_model_resolve')({},simultaneous);
assert.equal(await send(provider.resolveDynamicModel({modelId:simultaneousModel.modelOverride}),own),'owner-test');
await hooks.get('before_tool_call')({toolName:'sessions_spawn'},simultaneous);
await hooks.get('subagent_spawned')({runId:'ambiguous-child',childSessionKey:'ambiguous-session',mode:'run'},
  {runId:'ambiguous-child',childSessionKey:'ambiguous-session',requesterSessionKey:parent.sessionKey});
assert.equal((await hooks.get('before_model_resolve')({},
  {runId:'ambiguous-child',sessionKey:'ambiguous-session',modelProviderId:'owner-mode'})).modelOverride,'test');
await hooks.get('agent_end')({},simultaneous);
const mixedOrigin={...parent,runId:'native-not-approved',sessionKey:'native-not-approved',
  messageId:'native-not-approved-message',senderId:'+15550000012',inputProvenance:child.inputProvenance};
await nativeIngress({isGroup:false,content:own},mixedOrigin);
const refused=await hooks.get('before_model_resolve')({prompt:own},mixedOrigin);
assert.equal(refused.modelOverride,'test');
await hooks.get('after_tool_call')({toolName:'sessions_spawn'},parent);
const forged={runId:'unverified',sessionKey:'unverified',modelProviderId:'owner-mode',
  channelId:'whatsapp',senderId:parent.senderId};
const stock=await hooks.get('before_model_resolve')({prompt:own},forged);
assert.equal(stock.modelOverride,'test');
assert.equal(await send(provider.resolveDynamicModel({modelId:'test'}),own),'off');
assert.equal(provider.resolveDynamicModel({modelId:'owner-run-not-issued'}),undefined);

async function nativeCase(id, events, content, expected='owner-test') {
  const context={...parent,runId:id,sessionKey:id,messageId:id+'-0'};
  for (let i=0;i<events.length;i++) {
    const {senderId=parent.senderId,...fields}=events[i];
    if (id==='wrong-text') await hooks.get('inbound_claim')({channel:'whatsapp',isGroup:false,senderId,...fields},
      {...context,channelId:'whatsapp',senderId,messageId:id+'-'+i});
    else await nativeIngress({isGroup:false,...fields},{...context,senderId,messageId:id+'-'+i});
  }
  const selected=await hooks.get('before_model_resolve')({prompt:typeof content==='string'?content:''},context);
  const resolved=provider.resolveDynamicModel({modelId:selected.modelOverride});
  assert.equal(await send(resolved,content),expected,id);
  await hooks.get('agent_end')({},context);
}
await nativeCase('voice',[{content:'',media:[{path:'/native/audio.ogg',kind:'audio'}]}],'A host-produced transcript.');
await nativeCase('image',[{content:'',media:[{path:'/native/photo.png',kind:'image'}]}],
  [{type:'text',text:'Describe the image.'},{type:'image_url',image_url:{url:'data:image/png;base64,AAAA'}}]);
await nativeCase('forwarded-image',[{content:'',media:[{path:'/native/photo.png',kind:'image'}],metadata:{forwarded:true}}],
  [{type:'image_url',image_url:{url:'data:image/png;base64,AAAA'}}]);
await nativeCase('quoted-image',[{content:'',media:[{path:'/native/photo.png',kind:'image'}],replyToBody:'Quoted context'}],
  [{type:'text',text:'Quoted context'},{type:'image_url',image_url:{url:'data:image/png;base64,AAAA'}}]);
await nativeCase('forward',[{content:own,metadata:{forwarded:true}}],own);
await nativeCase('quote',[{content:own,replyToBody:'Quoted text'}],'Quoted text\n\n'+own);
await nativeCase('group',[{content:'First message.'},{content:'Second message.'}],'First message.\n\nSecond message.');
await nativeCase('mixed',[{content:'First message.'},{content:'Second message.',senderId:'+15550000012'}],
  'First message.\n\nSecond message.','off');
await nativeCase('wrong-text',[{content:own}],'Different text.','off');

// A new native binding does not turn a same-session CLI call into native ingress.
const bound={...parent,runId:'native-after-cli',sessionKey:'same-session-cli',messageId:'cli-native-message'};
await nativeIngress({content:own,isGroup:false},bound);
const cli={...bound,runId:'cli-at-native-session'};delete cli.senderId;
assert.equal((await hooks.get('before_model_resolve')({},cli)).modelOverride,'test');
const nativeAfterCli=await hooks.get('before_model_resolve')({},bound);
// An unbound capability cannot authorize any body before the provider callback binds it.
assert.equal(await directSend(provider.resolveDynamicModel({modelId:nativeAfterCli.modelOverride}),
 {model:'test',messages:[{role:'user',content:'Unbound different content.'}]}),'off');
assert.equal(await send(provider.resolveDynamicModel({modelId:nativeAfterCli.modelOverride}),own),'owner-test');
await hooks.get('agent_end')({},bound);
const collision={...parent,runId:'before-group-collision',sessionKey:'group-collision',messageId:'before-group-collision'};
await nativeIngress({content:own,isGroup:false},collision);
const collisionModel=provider.resolveDynamicModel({modelId:(await hooks.get('before_model_resolve')({},collision)).modelOverride});
assert.equal(await send(collisionModel,own),'owner-test');
await nativeIngress({content:own,isGroup:true},{...collision,messageId:'group-at-bound-session'});
assert.equal(await send(collisionModel,own),'off');
api.config.session.dmScope='main';
assert.equal(await send(model,own),'off');
await nativeCase('shared-session',[{content:own}],own,'off');
api.config.session.dmScope='per-account-channel-peer';
await nativeCase('real-group',[{content:own,isGroup:true}],own,'off');

await hooks.get('agent_end')({},parent);
assert.equal(await send(model,own),'off');
const stockParams={temperature:.3};
assert.equal(provider.prepareExtraParams({config:api.config,modelId:resolved.modelOverride,extraParams:stockParams}),stockParams);
assert.equal(provider.prepareExtraParams({config:api.config,modelId:'owner-run-not-issued',extraParams:stockParams}),stockParams);
console.log('OPENCLAW_NATIVE_CONTRACT_OK');
