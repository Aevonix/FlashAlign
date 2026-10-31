// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import plugin from './index.mjs';

const configurations = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const services = [];
function service(configuration) {
  let registered;
  plugin.register({pluginConfig:configuration, config:{session:{dmScope:'per-account-channel-peer'}},
    on() {}, registerProvider() {}, registerService(value) { registered = value; }});
  assert.ok(registered);
  services.push(registered);
  return registered;
}
async function identity(configuration) {
  const key = readFileSync(configuration.controlKeyFile).toString('hex');
  const response = await fetch(new URL('/_owner/ready', configuration.bridgeUrl), {
    method:'POST', headers:{Authorization:`Bearer ${key}`,'Content-Type':'application/json'}, body:'{}',
    signal:AbortSignal.timeout(1000)});
  assert.equal(response.status, 200);
  return response.json();
}
try {
  const first = service(configurations[0]);
  await first.start();
  const before = await identity(configurations[0]);
  assert.ok(Number.isInteger(before.pid) && before.pid > 0);
  assert.match(before.instance, /^[0-9a-f-]{36}$/);

  // The same control key deliberately models a stale process from a prior
  // deployment. A models-200 or valid authentication alone cannot identify it.
  const stale = service(configurations[1]);
  await assert.rejects(stale.start(), /different FlashAlign bridge|exited during startup/);
  assert.deepEqual(await identity(configurations[0]), before);

  // Race two actual Python children on one fresh port. Only the process that
  // owns the listener may report successful startup.
  const rivals = configurations.slice(2).map(service);
  const outcomes = await Promise.allSettled(rivals.map(value => value.start()));
  assert.equal(outcomes.filter(value => value.status === 'fulfilled').length, 1);
  assert.equal(outcomes.filter(value => value.status === 'rejected').length, 1);
  console.log('BRIDGE_READINESS_IDENTITY_OK');
} finally {
  for (const value of services) value.stop();
  await new Promise(resolve => setTimeout(resolve, 250));
}
