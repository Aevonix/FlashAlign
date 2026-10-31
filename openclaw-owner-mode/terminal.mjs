// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
/* Trusted local TUI seam. A request body or channel label cannot create receipts. */
import { randomUUID } from 'node:crypto';
import { readFileSync, readdirSync, mkdtempSync, writeFileSync, rmSync, lstatSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:net';
import { spawn } from 'node:child_process';
import process from 'node:process';

const slot = Symbol.for('flashalign.native-terminal.v1');

export function terminalReceipt(run, session) {
  const state = process[slot];
  if (!state || state.pid !== process.pid) return undefined;
  const receipt = state.pending.get(run);
  if (!receipt || receipt.session !== session) return undefined;
  state.pending.delete(run);
  return {...receipt, terminalSession: state.session};
}

export function terminalSessionActive() {
  return process[slot]?.pid === process.pid;
}

export function terminalCleanup(stopBackend, stopBridge) {
  let closing;
  return () => closing ??= (async () => {
    try { await stopBackend(); }
    finally { await stopBridge(); }
  })();
}

export function wrapTerminalBackend(backend, session) {
  if (process[slot]) throw new Error('A terminal adapter is already active');
  const state = {pid: process.pid, session, pending: new Map()};
  process[slot] = state;
  const send = backend.sendChat.bind(backend);
  backend.sendChat = async (options) => {
    const runId = options.runId ?? randomUUID();
    // Media needs a separate native render binding. This initial seam is text-only.
    if (typeof options.message === 'string' && options.message.trim() && !options.images?.length && !options.attachments?.length) {
      state.pending.set(runId, {session: options.sessionKey, text: options.message});
      while (state.pending.size > 4096) state.pending.delete(state.pending.keys().next().value);
    }
    try { return await send({...options, runId}); }
    catch (error) { state.pending.delete(runId); throw error; }
  };
  return () => {
    backend.sendChat = send;
    if (process[slot] === state) delete process[slot];
  };
}

async function nativeModule(root, prefix, marker, name) {
  const dist = join(root, 'dist');
  const candidates = readdirSync(dist).filter(file => file.startsWith(prefix) && file.endsWith('.mjs') &&
    readFileSync(join(dist, file), 'utf8').includes(marker));
  for (const file of candidates) {
    const module = await import(pathToFileURL(join(dist, file)).href);
    if (typeof module[name] === 'function') return module;
  }
  throw new Error(`This OpenClaw installation has no supported native terminal ${name} interface`);
}

async function privateTerminalConfig() {
  const original = process.env.OPENCLAW_CONFIG_PATH;
  if (!original) throw new Error('OpenClaw configuration path is required');
  const config = JSON.parse(readFileSync(original, 'utf8'));
  const entry = config.plugins?.entries?.['owner-mode']?.config;
  if (!entry?.bridgeConfig || !entry.corePath || !entry.python || !entry.controlKeyFile) {
    throw new Error('Configure the FlashAlign OpenClaw plugin before opening a native terminal');
  }
  const keyInfo = lstatSync(entry.controlKeyFile);
  if (!keyInfo.isFile() || keyInfo.isSymbolicLink() || keyInfo.nlink !== 1 || (keyInfo.mode & 0o777) !== 0o600 ||
      process.getuid && keyInfo.uid !== process.getuid()) throw new Error('Invalid terminal control key permissions');
  const key = readFileSync(entry.controlKeyFile);
  if (key.length !== 32) throw new Error('Invalid terminal control key');
  if (!config.models?.providers?.['owner-mode']) throw new Error('The configured FlashAlign provider is missing');
  const bridge = JSON.parse(readFileSync(entry.bridgeConfig, 'utf8'));
  const port = await new Promise((accept, reject) => {
    const server = createServer(); server.once('error', reject);
    server.listen(0, '127.0.0.1', () => { const selected=server.address().port; server.close(() => accept(selected)); });
  });
  const temporary = mkdtempSync(join(dirname(original), '.flashalign-terminal-'));
  try {
    const bridgePath = join(temporary, 'bridge.json');
    bridge.port = port;
    writeFileSync(bridgePath, JSON.stringify(bridge), {mode:0o600, flag:'wx'});
    config.plugins.entries['owner-mode'].config = {...entry, bridgeConfig:bridgePath, bridgeUrl:`http://127.0.0.1:${port}`};
    config.models.providers['owner-mode'].baseUrl = `http://127.0.0.1:${port}/head/v1`;
    const configPath = join(temporary, 'openclaw.json');
    writeFileSync(configPath, JSON.stringify(config), {mode:0o600, flag:'wx'});
    process.env.OPENCLAW_CONFIG_PATH = configPath;
    return {temporary, original, entry, bridgePath, port, key};
  } catch (error) { rmSync(temporary, {recursive:true, force:true}); throw error; }
}

async function startTerminalBridge(prepared) {
  const instance = randomUUID();
  const child = spawn(prepared.entry.python, [join(prepared.entry.corePath, 'serve.py'), '--config', prepared.bridgePath],
    {stdio:'ignore', env:{...process.env, FLASHALIGN_BRIDGE_INSTANCE:instance}});
  let failed = false; child.on('error', () => { failed=true; });
  let stopping;
  const stop = () => stopping ??= (async () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.kill('SIGTERM');
    for (let i=0;i<100 && child.exitCode === null && child.signalCode === null;i++) await new Promise(r=>setTimeout(r,50));
    if (child.exitCode === null && child.signalCode === null) {
      child.kill('SIGKILL');
      for (let i=0;i<100 && child.exitCode === null && child.signalCode === null;i++) await new Promise(r=>setTimeout(r,50));
    }
    if (child.exitCode === null && child.signalCode === null) throw new Error('Owned terminal bridge did not stop');
  })();
  try {
    for (let i=0;i<100;i++) {
      if (failed || child.exitCode !== null || child.signalCode !== null) throw new Error('Terminal bridge did not start');
      try {
        const response=await fetch(`http://127.0.0.1:${prepared.port}/_owner/ready`, {method:'POST',
          headers:{'Content-Type':'application/json',Authorization:`Bearer ${prepared.key.toString('hex')}`},
          body:'{}',signal:AbortSignal.timeout(500)});
        const value=response.ok ? await response.json() : {};
        if (value.pid===child.pid && value.instance===instance) return stop;
      } catch {}
      await new Promise(r=>setTimeout(r,100));
    }
    throw new Error('Terminal bridge readiness identity was not confirmed');
  } catch (error) { await stop(); throw error; }
}

export async function runNativeTerminal(root, session) {
  if (!process.stdin.isTTY || !process.stdout.isTTY) throw new Error('Native terminal sessions require an interactive terminal');
  const prepared = await privateTerminalConfig();
  try {
    const ui = await nativeModule(root, 'tui-', 'runTui', 'runTui');
    const runtime = await nativeModule(root, 'embedded-backend-', 'EmbeddedTuiBackend', 'EmbeddedTuiBackend');
    if (typeof ui.withEmbeddedTuiStateLock !== 'function') throw new Error('OpenClaw has no isolated embedded TUI lifecycle');
    await ui.withEmbeddedTuiStateLock(async () => {
      const stop = await startTerminalBridge(prepared);
      let release = () => {};
      let stopBackend = async () => {};
      const cleanup = terminalCleanup(() => stopBackend(), stop);
      const interrupted = () => { void cleanup().then(() => {
        rmSync(prepared.temporary, {recursive:true, force:true}); process.exit(143);
      }, () => { process.exitCode=1; }); };
      process.once('SIGTERM', interrupted); process.once('SIGHUP', interrupted);
      try {
        const backend = new runtime.EmbeddedTuiBackend();
        const mcp = await import(pathToFileURL(join(root, 'dist/agents/agent-bundle-mcp-manager-api.js')).href);
        if (typeof mcp.setSessionMcpRuntimeScheduler !== 'function' || !backend.scheduler) {
          throw new Error('OpenClaw has no owned local MCP scheduler interface');
        }
        await mcp.setSessionMcpRuntimeScheduler(backend.scheduler);
        const backendStop = backend.stop.bind(backend);
        let stopped;
        backend.stop = () => stopped ??= (async () => {
          try { await mcp.disposeAllSessionMcpRuntimes(); }
          finally { await backendStop(); }
        })();
        stopBackend = () => backend.stop();
        release = wrapTerminalBackend(backend, session);
        await ui.runTui({local:true, backend, session:`agent:main:flashalign-${session}`, forceProcessExitOnReturn:false});
      }
      finally { process.removeListener('SIGTERM', interrupted); process.removeListener('SIGHUP', interrupted); release(); await cleanup(); }
    });
  } finally {
    process.env.OPENCLAW_CONFIG_PATH = prepared.original;
    rmSync(prepared.temporary, {recursive:true, force:true});
  }
}
