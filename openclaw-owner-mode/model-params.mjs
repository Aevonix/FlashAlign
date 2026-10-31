// SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
// SPDX-License-Identifier: AGPL-3.0-only
// Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
// Native OpenClaw still validates and maps these portable request parameters.
// Transport addresses, headers, credentials and owner controls are never copied.
const portable = new Set([
  'temperature', 'topP', 'topK', 'minP', 'seed',
  'maxTokens', 'max_tokens', 'max_completion_tokens',
  'frequencyPenalty', 'frequency_penalty', 'presencePenalty', 'presence_penalty',
  'parallelToolCalls', 'parallel_tool_calls', 'responseFormat', 'response_format',
  'stop', 'streaming', 'transport', 'cacheRetention',
  'textVerbosity', 'text_verbosity', 'reasoningEffort', 'reasoning_effort',
]);

const record = value => value && typeof value === 'object' && !Array.isArray(value);
const selected = value => record(value)
  ? Object.fromEntries(Object.entries(value).filter(([key]) => portable.has(key))) : {};

// The native model-parameter lookup uses this provider/model key spelling.
function modelParams(models, id) {
  const key = id.toLowerCase().startsWith('owner-mode/') ? id : `owner-mode/${id}`;
  return selected(models?.[key]?.params);
}

function agentEntry(config, id) {
  if (!id) return undefined;
  const normalize = value => String(value || '').trim().toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 64) || 'main';
  const wanted = normalize(id);
  const agents = config?.agents;
  if (Object.hasOwn(agents || {}, 'entries') && agents.entries !== undefined) {
    return record(agents.entries)
      ? Object.entries(agents.entries).find(([key]) => normalize(key) === wanted)?.[1] : undefined;
  }
  return Array.isArray(agents?.list) ? agents.list.find(value => value && normalize(value.id) === wanted) : undefined;
}

function sources(config, modelId, agentId) {
  const agent = agentEntry(config, agentId);
  return [selected(config?.agents?.defaults?.params),
    modelParams(config?.agents?.defaults?.models, modelId),
    modelParams(agent?.models, modelId), selected(agent?.params)];
}

const tokenKeys = ['maxTokens', 'max_completion_tokens', 'max_tokens'];

export function canonicalTokenDefault(config, modelId, agentId) {
  let result;
  for (const source of sources(config, modelId, agentId)) {
    for (const key of tokenKeys) {
      const value = source[key];
      if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
        result = value; break;
      }
    }
  }
  return result;
}

export function canonicalModelParams(config, modelId, agentId, existing) {
  // Existing native extras include global defaults. Canonical model sources
  // override them; agent-wide settings retain native precedence. Explicit
  // request/session overrides are reapplied by OpenClaw's outer stream mapper.
  const [,model,agentModel,agent] = sources(config, modelId, agentId);
  const result = {...existing, ...model, ...agentModel, ...agent};
  // Leave all token aliases to the outer native override mapper. The inner
  // stream wrapper supplies the canonical default only if that mapper did not
  // supply an explicit limit. Restoring maxTokens here would shadow max_tokens.
  for (const key of tokenKeys) delete result[key];
  return result;
}

export function checkedRequestPolicy(value) {
  if (value === undefined) return undefined;
  const allowed = new Set(['output_budget', 'supports_thinking_off', 'implicit_model_limit']);
  if (!record(value) || !Object.keys(value).length || Object.keys(value).some(key => !allowed.has(key)) ||
      value.output_budget !== undefined && value.output_budget !== 'remaining_context' ||
      value.supports_thinking_off !== undefined && typeof value.supports_thinking_off !== 'boolean' ||
      value.implicit_model_limit !== undefined && (typeof value.implicit_model_limit !== 'boolean' ||
        value.output_budget !== 'remaining_context')) throw new Error('Unsupported FlashAlign request policy');
  return {...value};
}

export function requestModel(model, options, policy, thinkingLevel) {
  if (!policy) return model;
  const effort = options?.reasoningEffort ?? options?.reasoning_effort ?? options?.reasoning ?? thinkingLevel;
  if (policy.supports_thinking_off === false && (effort === false ||
      typeof effort === 'string' && ['off', 'none', 'disabled'].includes(effort.trim().toLowerCase()))) {
    throw new Error('The selected model does not support thinking off');
  }
  if (policy.output_budget !== 'remaining_context') return model;
  const positive = value => typeof value === 'number' && Number.isFinite(value) && value > 0;
  const parameterKey = tokenKeys.find(key => positive(model.params?.[key]));
  const explicit = positive(options?.maxTokens) || parameterKey !== undefined;
  // A catalog capacity is not a request budget. Real capacities still constrain
  // explicit requests; a synthetic capacity must never cap a task override.
  if (policy.implicit_model_limit !== true && explicit) {
    // Native model.params budgets bypass its normal explicit-option clamp.
    if (parameterKey && positive(model.maxTokens) && model.params[parameterKey] > model.maxTokens) {
      return {...model, params:{...model.params, [parameterKey]:model.maxTokens}};
    }
    return model;
  }
  const result = {...model};
  delete result.maxTokens;
  return result;
}
