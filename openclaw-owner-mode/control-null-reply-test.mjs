// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import { randomBytes } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test, { after } from 'node:test';
import { createAdapter } from './index.mjs';

const root = mkdtempSync(join(tmpdir(), 'flashalign-control-'));
after(() => rmSync(root, {recursive:true, force:true}));

test('a failing control reply without a usable reason has the fixed error', async () => {
  const key = join(root, 'control.key');
  writeFileSync(key, randomBytes(32), {mode:0o600});
  const api = {on() {}, registerProvider() {}};
  const config = {bridgeUrl:'http://127.0.0.1:8030', controlKeyFile:key, model:{id:'bench'}};
  for (const body of ['null', '{}', '[]', '5', 'true', '"denied"', '{"reason":5}', '{"reason":"Not A Code"}']) {
    const adapter = createAdapter(api, config, async () => new Response(body, {status:403}));
    await assert.rejects(adapter.ready(), {name:'Error', reason:'control_request_rejected',
      message:'Owner Mode control request failed: control_request_rejected'}, body);
  }
});
