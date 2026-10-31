// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createAdapter } from './index.mjs';
import { wrapTerminalBackend, terminalReceipt, terminalCleanup } from './terminal.mjs';

const root=mkdtempSync(join(tmpdir(),'flashalign-terminal-test-'));
try {
  const key=join(root,'key');writeFileSync(key,Buffer.alloc(32),{mode:0o600});
  const hooks=new Map();const calls=[];let provider;
  const api={on:(name,fn)=>hooks.set(name,fn),registerProvider:value=>{provider=value}};
  createAdapter(api,{bridgeUrl:'http://127.0.0.1:1',controlKeyFile:key,model:{id:'test'}},
    async (url,options)=>{calls.push({action:new URL(url).pathname,body:JSON.parse(options.body)});
      return {ok:true,json:async()=>({capability:'cap',mode:'owner'})};});
  const context={sessionKey:'session',runId:'native-turn',modelProviderId:'owner-mode'};
  assert.equal((await hooks.get('before_model_resolve')({},context)).modelOverride,'test');
  assert.equal(calls.length,0);
  const backend={async sendChat(options){
    assert.equal(terminalReceipt(options.runId,'different-session'),undefined);
    const result=await hooks.get('before_model_resolve')({prompt:'actual typed message'}, {...context,runId:options.runId});
    assert.notEqual(result.modelOverride,'test');
    const model=provider.resolveDynamicModel({modelId:result.modelOverride});
    await provider.wrapStreamFn({streamFn:async(selected,context,options)=>{
      await options.onPayload({messages:[{role:'user',content:'actual typed message'}]},selected);
    }})(model,{},{});
    return {runId:options.runId};
  }};
  const release=wrapTerminalBackend(backend,'terminal-session');
  await backend.sendChat({sessionKey:'session',runId:'native-turn',message:'actual typed message'});
  assert.deepEqual(calls[0],{action:'/_owner/terminal_claim',body:{session:'session',run:'native-turn',
    terminal_session:'terminal-session',text:'actual typed message',rendered:'actual typed message'}});
  assert.equal(terminalReceipt('native-turn','session'),undefined);
  release();
  assert.equal(terminalReceipt('other','session'),undefined);
  const media={async sendChat(options){return {runId:options.runId}}};
  const releaseMedia=wrapTerminalBackend(media,'session');
  await media.sendChat({sessionKey:'session',runId:'media',message:'picture',images:['image']});
  assert.equal(terminalReceipt('media','session'),undefined);
  releaseMedia();
  const stopped=[];
  const cleanup=terminalCleanup(async()=>{stopped.push('backend/mcp');},async()=>{stopped.push('bridge');});
  await Promise.all([cleanup(),cleanup()]);
  assert.deepEqual(stopped,['backend/mcp','bridge']);
  console.log('Native terminal receipt, no-label bypass, one-use and media exclusion passed');
} finally {rmSync(root,{recursive:true,force:true});}
