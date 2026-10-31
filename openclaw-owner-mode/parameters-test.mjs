// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import { canonicalModelParams, canonicalTokenDefault, checkedRequestPolicy, requestModel } from './model-params.mjs';

const id='organization/model', key='owner-mode/'+id;
const cfg={agents:{defaults:{params:{temperature:.8,maxTokens:32},models:{[key]:{params:{
  temperature:0,topP:.9,max_tokens:64,cacheRetention:'short',seed:3,headers:{secret:'not-a-credential'},
  baseUrl:'https://invalid.example',owner_mode:'untrusted',messages:[],tools:[],model:'other',
}}}},entries:{one:{models:{[key]:{params:{topP:.7,max_completion_tokens:128}}},params:{seed:9}},
  two:{models:{[key]:{params:{temperature:.4}}},params:{maxTokens:256}}}}};
const before=JSON.stringify(cfg);
const existing={temperature:.8,maxTokens:32,frequencyPenalty:.2};
const first=canonicalModelParams(cfg,id,'one',existing);
assert.equal(first.temperature,0);assert.equal(first.topP,.7);assert.equal(first.seed,9);
assert.equal(first.frequencyPenalty,.2);assert.equal(first.cacheRetention,'short');
assert.equal(first.maxTokens,undefined);assert.equal(first.max_tokens,undefined);
assert.equal(first.max_completion_tokens,undefined);
for(const field of ['headers','baseUrl','owner_mode','messages','tools','model'])assert.equal(first[field],undefined);
assert.equal(canonicalTokenDefault(cfg,id,'one'),128);
assert.equal(canonicalTokenDefault(cfg,id,'two'),256);
assert.equal(canonicalModelParams(cfg,id,'two',existing).temperature,.4);
assert.equal(canonicalModelParams(cfg,id,'one',existing).temperature,0);
assert.equal(JSON.stringify(cfg),before);assert.equal(existing.maxTokens,32);
const legacy=structuredClone(cfg);delete legacy.agents.entries;
legacy.agents.list=[{id:'one',models:{[key]:{params:{temperature:.3,maxTokens:512}}}}];
assert.equal(canonicalModelParams(legacy,id,'one',{}).temperature,.3);
assert.equal(canonicalTokenDefault(legacy,id,'one'),512);
const policy=checkedRequestPolicy({output_budget:'remaining_context',supports_thinking_off:false,implicit_model_limit:true});
const catalog={id,maxTokens:8192,contextWindow:65536};
assert.equal(requestModel(catalog,{},policy).maxTokens,undefined);
assert.equal(requestModel(catalog,{maxTokens:8192},policy).maxTokens,undefined);
assert.equal(requestModel(catalog,{maxTokens:12000},policy).maxTokens,undefined);
assert.equal(catalog.maxTokens,8192);
assert.throws(()=>requestModel(catalog,{reasoning:'off'},policy),/does not support thinking off/);
assert.throws(()=>requestModel(catalog,{},policy,'off'),/does not support thinking off/);
assert.throws(()=>requestModel(catalog,{},policy,' OFF '),/does not support thinking off/);
assert.equal(requestModel(catalog,{reasoning:'low'},policy,'off').maxTokens,undefined);
for(const level of ['low','high','max'])assert.equal(requestModel(catalog,{reasoning:level},policy).maxTokens,undefined);
const real={...policy,implicit_model_limit:false};
assert.equal(requestModel(catalog,{maxTokens:12000},real),catalog);
assert.equal(requestModel({...catalog,params:{max_tokens:12000}},{},real).params.max_tokens,8192);
assert.equal(requestModel(catalog,{},real).maxTokens,undefined);
assert.equal(requestModel(catalog,{},undefined),catalog);
for(const value of [{},null,{supports_thinking_off:0},{output_budget:8192},{implicit_model_limit:true}]){
  assert.throws(()=>checkedRequestPolicy(value),/Unsupported/);
}
console.log('OPENCLAW_CANONICAL_PARAMETERS_OK');
