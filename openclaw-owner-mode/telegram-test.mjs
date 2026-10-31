// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import {createAdapter} from './index.mjs';

const [url, keyFile] = process.argv.slice(2);
const hooks = new Map(); let provider;
const api = {config:{session:{dmScope:'per-account-channel-peer'}},
  on(name, fn) {hooks.set(name, fn);}, registerProvider(value) {provider=value;}};
createAdapter(api, {bridgeUrl:url, controlKeyFile:keyFile,
  model:{id:'test', name:'test', contextWindow:32768, maxTokens:64, input:['text','image']}});

async function run(id, changes={}) {
  const context={runId:id, sessionKey:`telegram-${id}`, channelId:'telegram', channel:'telegram',
    accountId:'bot-a', senderId:'123456789', messageId:`message-${id}`, trigger:'user',
    modelProviderId:'owner-mode', ...changes};
  const event={content:'Describe this.', isGroup:false, channel:context.channel,
    senderId:context.senderId, messageId:context.messageId, sessionKey:context.sessionKey,
    media:[{url:'data:image/png;base64,AAAA'}]};
  await hooks.get('message_received')(event, context);
  await hooks.get('before_dispatch')(event, context);
  const selected=await hooks.get('before_model_resolve')({}, context);
  const model=provider.resolveDynamicModel({modelId:selected.modelOverride});
  const content=[{type:'text',text:'Describe this.'},
    {type:'image_url',image_url:{url:'data:image/png;base64,AAAA'}}];
  return provider.wrapStreamFn({streamFn:async (chosen, ctx, options) => {
    const body={model:'test', messages:[{role:'user',content}]};
    const final=await options?.onPayload?.(body, chosen) ?? body;
    const response=await fetch(chosen.baseUrl+'/chat/completions', {method:'POST',
      headers:{'Content-Type':'application/json',...chosen.headers},body:JSON.stringify(final)});
    assert.equal(response.status,200);
    return response.headers.get('x-owner-mode-effective');
  }})(model, {}, {});
}

assert.equal(await run('approved'), 'owner-test');
assert.equal(await run('wrong-account',{accountId:'bot-b'}), 'off');
assert.equal(await run('display-name',{senderId:'@alice'}), 'off');
assert.equal(await run('wrong-sender',{senderId:'987654321'}), 'off');
// A terminal-looking label is neither native terminal attestation nor a chat adapter.
assert.equal(await run('forged-terminal',{channel:'local-terminal',channelId:'local-terminal',senderId:'uid:1000'}), 'off');
