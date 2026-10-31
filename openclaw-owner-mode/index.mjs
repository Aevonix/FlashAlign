// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import { readFileSync, lstatSync } from "node:fs";
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { canonicalModelParams, canonicalTokenDefault, checkedRequestPolicy, requestModel } from './model-params.mjs';
import { terminalReceipt, terminalSessionActive } from './terminal.mjs';

const clean = (v) => typeof v === "string" ? v.trim() : "";

export function createAdapter(api, config, request = fetch) {
  if (!api.on || !api.registerProvider) throw new Error("Unsupported OpenClaw plugin API");
  const requestPolicy = checkedRequestPolicy(config.requestPolicy);
  const endpoint = new URL(config.bridgeUrl);
  if (!['127.0.0.1', '[::1]', 'localhost'].includes(endpoint.hostname) || endpoint.protocol !== 'http:') {
    throw new Error("Owner Mode bridge must use loopback HTTP");
  }
  const keyInfo = lstatSync(config.controlKeyFile);
  if (!keyInfo.isFile() || keyInfo.isSymbolicLink() || keyInfo.nlink !== 1 || (keyInfo.mode & 0o777) !== 0o600 ||
      (process.getuid && keyInfo.uid !== process.getuid())) throw new Error("Invalid control key permissions");
  const key = readFileSync(config.controlKeyFile);
  if (key.length !== 32) throw new Error("Invalid control key length");
  const runs = new Map(), ingress = new Map(), aliases = new Map(), pendingTools = new Map(), seen = new Map();
  const endedRuns = new Set();
  const agentEvents = api.agent?.events;
  const registerLifecycle = typeof agentEvents?.registerAgentEventSubscription === 'function'
    ? (subscription) => agentEvents.registerAgentEventSubscription(subscription)
    : typeof api.registerAgentEventSubscription === 'function'
      ? (subscription) => api.registerAgentEventSubscription(subscription) : undefined;

  async function control(action, body, timeout) {
    const response = await request(new URL(`/_owner/${action}`, endpoint), {
      method: 'POST', headers: {'Content-Type': 'application/json', Authorization: `Bearer ${key.toString('hex')}`},
      body: JSON.stringify(body), signal: AbortSignal.timeout(timeout ?? (action === 'bind' ? 30000 : 5000)),
    });
    if (!response.ok) throw new Error("Owner Mode control request failed");
    return response.json();
  }

  function remember(id, value) {
    runs.set(id, value);
    while (runs.size > 4096) runs.delete(runs.keys().next().value);
  }

  api.on('inbound_claim', async (event, context) => {
    const id = clean(context.runId || event.runId);
    const session = clean(context.sessionKey || event.sessionKey);
    const channel = clean(context.channelId);
    const contact = clean(context.senderId || event.senderId || event.from);
    const text = typeof event.content === 'string' ? event.content : '';
    const messageId = clean(context.messageId || event.messageId || event.providerUpdate?.messageId);
    const account = clean(context.accountId || event.accountId) || 'default';
    const media = [...(Array.isArray(event.media) ? event.media : []),
      ...(Array.isArray(event.originalMedia) ? event.originalMedia : [])];
    const hasMedia = media.some(item => item && typeof item === 'object' && (clean(item.path) || clean(item.url))) ||
      Boolean(event.location && Number.isFinite(event.location.latitude) && Number.isFinite(event.location.longitude));
    const valid = id && session && messageId && contact && channel && (text || hasMedia) && event.isGroup === false &&
      event.channel === channel && !(context.senderId && event.senderId && context.senderId !== event.senderId) &&
      !(context.runId && event.runId && context.runId !== event.runId) &&
      !(context.sessionKey && event.sessionKey && context.sessionKey !== event.sessionKey) &&
      !(context.accountId && event.accountId && context.accountId !== event.accountId) &&
      !(context.messageId && event.messageId && context.messageId !== event.messageId) && event.metadata?.reaction !== true &&
      !String(event.providerUpdate?.kind || '').toLowerCase().includes('reaction');
    if (!id || endedRuns.has(id)) return;
    // This callback receives native host ingress. No identity is read from prompt text.
    const group = ingress.get(id) || {session, messages: [], invalid: false};
    const record = {id: messageId, channel, account, contact, text, hasMedia,
      fingerprint: createHash('sha256').update(JSON.stringify({text, media, location:event.location})).digest('hex')};
    const previous = group.messages.find(item => item.id === messageId);
    const messageKey = JSON.stringify([channel,account,contact,messageId]);
    const priorRun = seen.get(messageKey);
    if (!valid || group.session !== session || group.messages.length >= 64 ||
        priorRun && priorRun !== id ||
        previous && JSON.stringify(previous) !== JSON.stringify(record) ||
        group.messages.some(item => item.channel !== channel || item.account !== account || item.contact !== contact) ||
        runs.get(id)?.bound && !previous) group.invalid = true;
    else if (!previous) group.messages.push(record);
    if (valid && !group.invalid) {
      seen.set(messageKey,id);
      while (seen.size > 8192) seen.delete(seen.keys().next().value);
    }
    ingress.set(id, group);
    while (ingress.size > 4096) ingress.delete(ingress.keys().next().value);
    if (group.invalid && runs.has(id)) {
      const run = runs.get(id); runs.delete(id);
      try { await control('end', {capability:run.capability}); } catch {}
    }
  });

  // Native message hooks have no runId in OpenClaw 2026.10.8. Bind host
  // ingress to an explicitly isolated contact session, then require the same
  // protected sender tuple from the later model hook. Never parse a session key.
  const nativeMessages = new Map(), nativeSessions = new Map();
  function isolatedSessions() {
    try {
      const cfg = api.runtime?.config?.current?.() ?? api.config;
      return ['per-channel-peer','per-account-channel-peer'].includes(cfg?.session?.dmScope);
    } catch { return false; }
  }
  const nativeKey = (session, message) => JSON.stringify([session,message]);
  api.on('message_received', (event, context) => {
    const session = clean(context.sessionKey || event.sessionKey);
    const message = clean(context.messageId || event.messageId);
    if (!session || !message) return;
    nativeMessages.set(nativeKey(session,message), structuredClone({event,context}));
    while (nativeMessages.size > 4096) nativeMessages.delete(nativeMessages.keys().next().value);
  });
  api.on('before_dispatch', (event, context) => {
    if (!isolatedSessions()) return;
    const session = clean(context.sessionKey || event.sessionKey);
    const message = clean(context.messageId || event.messageId);
    const channel = clean(context.channelId), account = clean(context.accountId) || 'default';
    const contact = clean(context.senderId || event.senderId);
    const source = nativeMessages.get(nativeKey(session,message));
    nativeMessages.delete(nativeKey(session,message));
    if (!source || !session || !message || !channel || !contact || event.isGroup !== false ||
        event.channel !== channel || source.context.channelId !== channel ||
        clean(source.context.sessionKey || source.event.sessionKey) !== session ||
        clean(source.context.senderId || source.event.senderId) !== contact ||
        (clean(source.context.accountId) || 'default') !== account ||
        (event.senderId && event.senderId !== contact)) {
      if (nativeSessions.has(session)) nativeSessions.set(session,{invalid:true});
      return;
    }
    const text = typeof event.content === 'string' ? event.content : '';
    const media = [...(Array.isArray(source.event.media) ? source.event.media : []),
      ...(Array.isArray(source.event.originalMedia) ? source.event.originalMedia : [])];
    const location = source.event.location;
    const hasMedia = media.some(item => item && typeof item === 'object' && (clean(item.path) || clean(item.url))) ||
      Boolean(location && Number.isFinite(location.latitude) && Number.isFinite(location.longitude));
    if (!text && !hasMedia || source.event.metadata?.reaction === true ||
        String(source.event.providerUpdate?.kind || '').toLowerCase().includes('reaction')) return;
    const prior = nativeSessions.get(session);
    if (prior?.invalid || prior && (prior.channel !== channel || prior.account !== account || prior.contact !== contact)) {
      nativeSessions.set(session,{invalid:true});return;
    }
    nativeSessions.set(session,{channel,account,contact,message,text,hasMedia,
      fingerprint:createHash('sha256').update(JSON.stringify({text,media,location})).digest('hex')});
    while (nativeSessions.size > 4096) nativeSessions.delete(nativeSessions.keys().next().value);
  });

  api.on('before_model_resolve', async (_event, context) => {
    const id = clean(context.runId), session = clean(context.sessionKey || context.sessionId);
    if (context.modelProviderId && context.modelProviderId !== 'owner-mode') return;
    if (!id || !session) return {providerOverride: 'owner-mode', modelOverride: config.model.id};
    if (endedRuns.has(id)) return {providerOverride: 'owner-mode', modelOverride: config.model.id};
    const terminal = terminalReceipt(id, session);
    if (terminal && !runs.has(id) && typeof _event.prompt === 'string' && _event.prompt) {
      try {
        const result = await control('terminal_claim', {session, run:id,
          terminal_session:terminal.terminalSession, text:terminal.text, rendered:_event.prompt});
        if (result.capability) remember(id, {capability:result.capability, session, bound:false,
          terminal:true, runId:id, messages:[{text:terminal.text,hasMedia:false}]});
      } catch {}
    }
    const binding = nativeSessions.get(session);
    const nativeChannel = clean(context.channel || context.messageProvider);
    if (isolatedSessions() && !ingress.has(id) && binding && !binding.invalid && context.trigger === 'user' &&
        nativeChannel === binding.channel && clean(context.senderId) === binding.contact &&
        (clean(context.accountId) || 'default') === binding.account) {
      ingress.set(id,{session,invalid:false,nativeSession:true,messages:[{id:binding.message,channel:binding.channel,
        account:binding.account,contact:binding.contact,text:binding.text,hasMedia:binding.hasMedia,
        fingerprint:binding.fingerprint}]});
    }
    const group = ingress.get(id);
    if (group && !group.invalid && !runs.has(id) && group.session === session) {
      try {
        const source = group.messages[0];
        const result = group.nativeSession
          ? await control('claim_session', {session, run:id, channel:source.channel,
              account:source.account, contact:source.contact})
          : await control('claim', {session, run:id, messages:group.messages.map(({id,channel,account,contact}) =>
              ({id,channel,account,contact}))});
        if (result.capability) remember(id, {capability: result.capability, session, bound:false, nativeSession:group.nativeSession===true, messages:group.messages});
      } catch {}
    }
    const origin = context.inputProvenance;
    if (!group && !runs.has(id) && origin?.kind === 'inter_session' && origin.sourceTool) {
      const parents = [...runs.entries()].filter(([parent, value]) => value.session === origin.sourceSessionKey &&
        [...(pendingTools.get(parent)?.values() || [])].includes(origin.sourceTool));
      if (parents.length === 1) {
        try {
          const result = await control('adopt', {parent: parents[0][1].capability,
            session, run: id, tool: origin.sourceTool});
          if (result.capability) remember(id, {capability: result.capability, session, bound:true});
        } catch {}
      }
    }
    const run = runs.get(id);
    if (!run || run.session !== session) return {providerOverride: 'owner-mode', modelOverride: config.model.id};
    const alias = `owner-run-${randomUUID()}`;
    aliases.set(alias, {id, ...run, agentId:clean(context.agentId)});
    while (aliases.size > 4096) aliases.delete(aliases.keys().next().value);
    return {providerOverride: 'owner-mode', modelOverride: alias};
  });

  api.on('before_tool_call', async (event, context) => {
    const id = clean(context.runId || event.runId);
    if (!id || !event.toolName) return;
    const call = clean(context.toolCallId || event.toolCallId) || `fallback:${event.toolName}`;
    if (!pendingTools.has(id)) pendingTools.set(id, new Map());
    pendingTools.get(id).set(call, event.toolName);
    const run = runs.get(id);
    if (run) {
      // A disabled tool temporarily makes model calls on this run stock.
      await control('tool_begin', {capability: run.capability, call, tool: event.toolName});
    }
    // Never grants or bypasses a host tool approval.
  });
  api.on('after_tool_call', async (event, context) => {
    const id = clean(context.runId || event.runId);
    const call = clean(context.toolCallId || event.toolCallId) || `fallback:${event.toolName}`;
    pendingTools.get(id)?.delete(call);
    const run = runs.get(id);
    if (run) { try { await control('tool_end', {capability: run.capability, call}); } catch {} }
  });
  // OpenClaw emits this awaited native lifecycle hook before the spawning
  // tool returns. Its child run/session and requester session are host values.
  // The child may start after its parent's tool scope and run have ended.
  api.on('subagent_spawned', async (event, context) => {
    const id = clean(context.runId || event.runId);
    const session = clean(context.childSessionKey || event.childSessionKey);
    const parentSession = clean(context.requesterSessionKey);
    if (!id || !session || !parentSession || session === parentSession || runs.has(id) || endedRuns.has(id) ||
        (context.runId && event.runId && context.runId !== event.runId) ||
        (context.childSessionKey && event.childSessionKey && context.childSessionKey !== event.childSessionKey) ||
        !['run','session'].includes(event.mode)) return;
    const parents = [...runs.entries()].filter(([parent,value]) => value.session === parentSession && value.bound &&
      [...(pendingTools.get(parent)?.values() || [])].includes('sessions_spawn'));
    if (parents.length !== 1) return;
    const parent = parents[0][1];
    if (parent.nativeSession && (!isolatedSessions() ||
        nativeSessions.get(parent.sourceSession || parent.session)?.invalid)) return;
    try {
      const result = await control('adopt', {parent:parent.capability, session, run:id, tool:'sessions_spawn'});
      if (result.capability) remember(id, {capability:result.capability, session, bound:true,
        nativeSession:parent.nativeSession === true, sourceSession:parent.sourceSession || parent.session,
        childLifecycle:{requesterSession:parentSession}});
    } catch {}
  });
  async function endRun(id) {
    if (!id) return;
    const run = runs.get(id);
    endedRuns.add(id);
    while (endedRuns.size > 4096) endedRuns.delete(endedRuns.values().next().value);
    runs.delete(id); ingress.delete(id); pendingTools.delete(id);
    for (const [alias, value] of aliases) if (value.id === id) aliases.delete(alias);
    if (run) { try { await control('end', {capability: run.capability}); } catch {} }
  }
  if (registerLifecycle) registerLifecycle({
    id:'flashalign-run-completion', streams:['lifecycle'],
    async handle(event) {
      if (event.stream !== 'lifecycle' || !['end','error'].includes(event.data?.phase)) return;
      const id = clean(event.runId), run = runs.get(id);
      if (!run || clean(event.sessionKey) !== run.session) return;
      await endRun(id);
    },
  });
  api.on('agent_end', async (_event, context) => {
    const id = clean(context.runId);
    // Native recovery emits agent_end between attempts of the same run. The
    // host lifecycle end/error event follows its final attempt.
    if (registerLifecycle || runs.get(id)?.childLifecycle) return;
    await endRun(id);
  });
  api.on('subagent_ended', async (event, context) => {
    const id = clean(context.runId || event.runId), run = runs.get(id);
    if (!run?.childLifecycle || event.targetKind !== 'subagent' ||
        clean(context.childSessionKey) !== run.session || clean(event.targetSessionKey) !== run.session ||
        clean(context.requesterSessionKey) !== run.childLifecycle.requesterSession ||
        (context.runId && event.runId && context.runId !== event.runId)) return;
    await endRun(id);
  });

  function wrapStream({streamFn, config:current, modelId, thinkingLevel}) {
    if (typeof streamFn !== 'function') return undefined;
    const alias = aliases.get(modelId);
    const tokenDefault = alias && runs.get(alias.id)?.capability === alias.capability
      ? canonicalTokenDefault(current || api.config, config.model.id, alias.agentId) : undefined;
    const wrapped = (model, context, options) => {
      if (options?.maxTokens === undefined && tokenDefault !== undefined) options = {...options, maxTokens:tokenDefault};
      if (model.id === config.model.id) model = requestModel(model, options, requestPolicy, thinkingLevel);
      const capability = model.headers?.['X-Owner-Mode-Run'];
      const run = [...runs.values()].find(value => value.capability === capability);
      if (!run) {
        if (!capability) return streamFn(model, context, options);
        const headers = {...model.headers}; delete headers['X-Owner-Mode-Run'];
        return streamFn({...model,headers}, context, options);
      }
      if (run.nativeSession && (!isolatedSessions() || nativeSessions.get(run.sourceSession || run.session)?.invalid)) {
        const headers = {...model.headers}; delete headers['X-Owner-Mode-Run'];
        return streamFn({...model,headers}, context, options);
      }
      if (run.bound) return streamFn(model, context, options);
      const prior = options?.onPayload;
      return streamFn(model, context, {...options, onPayload: async (payload, payloadModel) => {
        const replacement = await prior?.(payload, payloadModel);
        const final = replacement === undefined ? payload : replacement;
        const last = Array.isArray(final?.messages) ? final.messages.at(-1) : undefined;
        if (last?.role === 'user') {
          const text = typeof last.content === 'string' ? last.content : Array.isArray(last.content)
            ? last.content.filter(part => part?.type === 'text').map(part => part.text).join('\n\n') : '';
          let cursor = 0;
          const complete = run.nativeSession || run.messages.every(item => {
            if (item.hasMedia) return true;
            const position = text.indexOf(item.text, cursor);
            if (position < 0) return false;
            cursor = position + item.text.length; return true;
          });
          if (complete) {
            try {
              if (run.terminal) await control('terminal_render', {capability,
                session:run.session, run:run.runId, content:last.content});
              run.bound = (await control('bind', {capability, content:last.content})).mode === 'owner';
            }
            catch {}
          }
        }
        return final;
      }});
    };
    wrapped.preservesGenericCompatibility = true;
    return wrapped;
  }

  api.registerProvider({
    id: 'owner-mode', label: 'Aevonix FlashAlign', auth: [],
    wrapStreamFn: wrapStream,
    wrapSimpleCompletionStreamFn: wrapStream,
    prepareExtraParams({config:current, modelId, extraParams}) {
      const alias = aliases.get(modelId);
      if (!alias || runs.get(alias.id)?.capability !== alias.capability) return extraParams;
      return canonicalModelParams(current || api.config, config.model.id, alias.agentId, extraParams);
    },
    resolveDynamicModel({modelId}) {
      const alias = aliases.get(modelId);
      if (modelId !== config.model.id && !alias) return undefined;
      const headers = alias ? {'X-Owner-Mode-Run': alias.capability} : {};
      return {...config.model, id: config.model.id, provider: 'owner-mode', api: 'openai-completions',
        baseUrl: new URL('/head/v1', endpoint).toString(), headers};
    },
    resolveSyntheticAuth() { return {apiKey: 'local-owner-mode', source: 'owner-mode-local', mode: 'api-key'}; },
  });

  return {
    async ready(timeout = 1000) { return control('ready', {}, timeout); },
    async toolEnabled(runId, toolName) {
      const run = runs.get(runId);
      if (!run) return false;
      try { return (await control('tool', {capability: run.capability, tool: toolName})).enabled === true; }
      catch { return false; }
    },
  };
}

export default {
  id: 'owner-mode', name: 'Aevonix FlashAlign',
  register(api) {
    const cfg = api.pluginConfig;
    const adapter = createAdapter(api, cfg);
    if (cfg.python && cfg.corePath && cfg.bridgeConfig && api.registerService && !terminalSessionActive()) {
      let child;
      api.registerService({id: 'owner-mode-bridge',
        async start() {
          let failed = false;
          const instance = randomUUID();
          child = spawn(cfg.python, [join(cfg.corePath, 'serve.py'), '--config', cfg.bridgeConfig],
            {stdio: ['ignore', 'ignore', 'ignore'],
              env: {...process.env, FLASHALIGN_BRIDGE_INSTANCE: instance}});
          child.on('error', () => { failed = true; });
          const deadline = Date.now() + 15000;
          try {
            while (Date.now() < deadline) {
              if (failed || child.exitCode !== null || child.signalCode !== null) {
                throw new Error('FlashAlign bridge exited during startup');
              }
              let identity;
              try { identity = await adapter.ready(Math.max(1, Math.min(1000, deadline - Date.now()))); } catch {}
              if (identity) {
                if (identity.pid !== child.pid || identity.instance !== instance) {
                  throw new Error('A different FlashAlign bridge is using the configured port');
                }
                if (failed || child.exitCode !== null || child.signalCode !== null) {
                  throw new Error('FlashAlign bridge exited during startup');
                }
                let models;
                try {
                  models = await fetch(new URL('/head/v1/models', cfg.bridgeUrl),
                    {signal: AbortSignal.timeout(Math.max(1, Math.min(1000, deadline - Date.now())))});
                } catch {}
                if (models?.ok && !failed && child.exitCode === null && child.signalCode === null) return;
              }
              await new Promise(resolve => setTimeout(resolve, 100));
            }
            throw new Error('FlashAlign bridge readiness identity was not confirmed');
          } catch (error) {
            if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
            throw error;
          }
        },
        stop() { if (child && child.exitCode === null) child.kill('SIGTERM'); },
      });
    }
  },
};
