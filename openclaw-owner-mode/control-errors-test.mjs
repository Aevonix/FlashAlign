// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createAdapter } from './index.mjs';

function fixture(request) {
  const root = mkdtempSync(join(tmpdir(), 'flashalign-control-'));
  const key = join(root, 'control.key');
  writeFileSync(key, randomBytes(32), {mode:0o600});
  const hooks = {};
  const api = {on(name, callback) { hooks[name] = callback; }, registerProvider() {}};
  const config = {bridgeUrl:'http://127.0.0.1:8030', controlKeyFile:key, model:{id:'bench'}};
  const adapter = createAdapter(api, config, request);
  const log = join(root, 'owner-mode.log');
  return {adapter, hooks, rows() {
    return existsSync(log) ? readFileSync(log, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [];
  }};
}

test('refused readiness polls remain silent before a successful start', async () => {
  let calls = 0;
  const {adapter, rows} = fixture(async () => {
    if (++calls <= 12) throw new Error('connection refused');
    return new Response(JSON.stringify({pid:1, instance:'fixture'}));
  });
  for (let i = 0; i < 12; i++) {
    await assert.rejects(adapter.ready(), /control_unavailable/);
  }
  assert.deepEqual(await adapter.ready(), {pid:1, instance:'fixture'});
  assert.deepEqual(rows(), []);
});

test('a non-JSON readiness rejection has a fixed error and no startup row', async () => {
  const {adapter, rows} = fixture(async () => new Response('<html>denied</html>', {status:403}));
  await assert.rejects(adapter.ready(), {name:'Error', message:'Owner Mode control request failed'});
  assert.deepEqual(rows(), []);
});

test('a non-JSON control rejection records one fixed decision without response content', async () => {
  const {hooks, rows} = fixture(async url => url.pathname === '/_owner/claim'
    ? new Response(JSON.stringify({capability:'fixture-run'}))
    : new Response('<html>denied</html>', {status:403}));
  const context = {runId:'run', sessionKey:'session', channelId:'rcs', senderId:'+10000000000',
    accountId:'default', messageId:'message'};
  await hooks.inbound_claim({content:'Hello', channel:'rcs', isGroup:false}, context);
  await hooks.before_model_resolve({}, context);
  await assert.rejects(hooks.before_tool_call({toolName:'fixture_tool'}, context),
    {name:'Error', message:'Owner Mode control request failed'});
  assert.equal(rows().length, 1);
  assert.equal(rows()[0].decision, 'control_request_rejected');
  assert.deepEqual(Object.keys(rows()[0]).sort(), ['decision', 'ts']);
});
