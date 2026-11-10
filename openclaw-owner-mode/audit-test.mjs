// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomBytes } from 'node:crypto';
import { createAdapter } from './index.mjs';

const root = mkdtempSync(join(tmpdir(), 'flashalign-audit-'));
try {
  const key = join(root, 'control.key');
  writeFileSync(key, randomBytes(32), {mode:0o600});
  const hooks = {};
  const api = {on(name, callback) { hooks[name] = callback; }, registerProvider() {}};
  const config = {bridgeUrl:'http://127.0.0.1:8030', controlKeyFile:key, model:{id:'bench'}};
  createAdapter(api, config, async url => {
    if (url.pathname === '/_owner/claim') return {ok:true, json:async () => ({capability:'fixture-run'})};
    throw new Error('unlogged detail');
  });
  const context = {runId:'run', sessionKey:'session', channelId:'rcs', senderId:'+10000000000',
    accountId:'default', messageId:'message'};
  await hooks.inbound_claim({content:'Hello', channel:'rcs', isGroup:false}, context);
  await hooks.before_model_resolve({}, context);
  await assert.rejects(hooks.before_tool_call({toolName:'fixture_tool'}, context), /control_unavailable/);
  const rows = readFileSync(join(root, 'owner-mode.log'), 'utf8').trim().split('\n').map(JSON.parse);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].decision, 'control_unavailable');
  assert.deepEqual(Object.keys(rows[0]).sort(), ['decision', 'ts']);
  const rejected = createAdapter(api, config, async () => ({ok:false,
    json:async () => ({error:'unlogged detail', reason:'missing_native_message_claims'})}));
  await assert.rejects(rejected.ready(), error => error.reason === 'missing_native_message_claims');
} finally {
  rmSync(root, {recursive:true, force:true});
}
