# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Typed request options shared by setup and the authenticated request policy.

These rules validate wire values, not model support or caller identity. Engine
and model adapters can restrict the available rules. New executable validators
are registered by trusted application code, never by request/profile metadata.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import re
from types import MappingProxyType


STRUCTURAL_FIELDS = frozenset({'model', 'messages', 'tools', 'owner_mode', 'owner_mode_scale'})
EFFORTS = frozenset({'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra', 'auto'})
TEMPLATE_FIELDS = frozenset({'enable_thinking', 'thinking', 'preserve_thinking',
                             'reasoning_effort', 'thinking_budget'})


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _effort(value):
    return isinstance(value, str) and value in EFFORTS or _integer(value)


def _bounded_json(value, depth=0, budget=None):
    budget = [4096] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        return False
    if value is None or type(value) in (str, bool, int):
        return not isinstance(value, str) or len(value) <= 65536
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_bounded_json(item, depth+1, budget) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and len(key) <= 1024
                   and _bounded_json(item, depth+1, budget) for key, item in value.items())
    return False


def _response_format(value, _body):
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    kind = value.get('type')
    if kind in ('text', 'json_object'):
        return set(value) == {'type'}
    if kind != 'json_schema' or set(value) != {'type', 'json_schema'}:
        return False
    schema = value['json_schema']
    if (not isinstance(schema, dict) or not {'name', 'schema'} <= set(schema)
            or set(schema) - {'name', 'description', 'schema', 'strict'}
            or not isinstance(schema['name'], str)
            or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', schema['name']) is None
            or not isinstance(schema['schema'], dict)
            or 'description' in schema and (not isinstance(schema['description'], str)
                                             or len(schema['description']) > 8192)
            or 'strict' in schema and type(schema['strict']) is not bool):
        return False
    return _bounded_json(schema['schema'])


def _tool_choice(value, body):
    if value is None or isinstance(value, str) and value in ('auto', 'none', 'required'):
        return True
    if (not isinstance(value, dict) or set(value) != {'type', 'function'}
            or value['type'] != 'function' or not isinstance(value['function'], dict)
            or set(value['function']) != {'name'}):
        return False
    name = value['function']['name']
    if not isinstance(name, str) or not name or len(name) > 256:
        return False
    tools = body.get('tools', [])
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get('type') == 'function'
        and isinstance(tool.get('function'), dict) and tool['function'].get('name') == name
        for tool in tools)


def _template(value, _body):
    if not isinstance(value, dict) or set(value) - TEMPLATE_FIELDS:
        return False
    for key, item in value.items():
        if key in ('enable_thinking', 'thinking', 'preserve_thinking'):
            if type(item) is not bool:
                return False
        elif key == 'reasoning_effort':
            if not _effort(item):
                return False
        elif not (_integer(item, -1) or isinstance(item, str) and item in EFFORTS | {'unlimited'}):
            return False
    return True


@dataclass(frozen=True)
class FieldRule:
    validator: object
    error: str = 'bad_generation_option'


@dataclass(frozen=True)
class RequestCapabilities:
    """An immutable trusted registry of option validators."""
    rules: object
    schema: str = 'flashalign.request-capabilities.v1'

    def __post_init__(self):
        rules = dict(self.rules)
        if any(not isinstance(name, str) or not name or name in STRUCTURAL_FIELDS
               or not isinstance(rule, FieldRule) or not callable(rule.validator)
               for name, rule in rules.items()):
            raise ValueError('Invalid trusted request capability')
        object.__setattr__(self, 'rules', MappingProxyType(rules))

    def with_field(self, name, validator, *, error='bad_generation_option'):
        if name in self.rules:
            raise ValueError('Request capability already registered')
        return RequestCapabilities({**self.rules, name: FieldRule(validator, error)})

    def restricted_to(self, fields):
        fields = frozenset(fields)
        if fields - self.rules.keys():
            raise ValueError('Unknown request capability')
        return RequestCapabilities({name: self.rules[name] for name in fields})

    def check(self, body):
        if not isinstance(body, dict) or set(body) - STRUCTURAL_FIELDS - self.rules.keys():
            return 'unknown_request_field'
        for name, value in body.items():
            rule = self.rules.get(name)
            if rule is None:
                continue
            try:
                valid = rule.validator(value, body)
            except Exception:
                valid = False
            if valid is not True:
                return rule.error
        return ''


def _rule(check, error='bad_generation_option'):
    return FieldRule(lambda value, _body: value is None or check(value), error)


_rules = {name: _rule(_number) for name in ('temperature', 'presence_penalty',
          'frequency_penalty', 'repetition_penalty', 'length_penalty')}
_rules.update({name: _rule(lambda value: _number(value) and 0 <= value <= 1)
               for name in ('top_p', 'min_p')})
_rules.update({name: _rule(lambda value: _integer(value, 1))
               for name in ('max_tokens', 'max_completion_tokens', 'max_output_tokens')})
_rules.update({name: _rule(lambda value: _integer(value)) for name in ('min_tokens', 'top_logprobs')})
_rules.update({
    'top_k': _rule(lambda value: _integer(value, -1)),
    'seed': _rule(lambda value: type(value) is int),
    'n': FieldRule(lambda value, _body: type(value) is int and value == 1, 'n_not_1'),
    'stream': FieldRule(lambda value, _body: type(value) is bool, 'bad_stream'),
    'stream_options': FieldRule(lambda value, _body: isinstance(value, dict)
                               and not set(value)-{'include_usage'}
                               and all(type(item) is bool for item in value.values()), 'bad_stream_options'),
    'ignore_eos': _rule(lambda value: type(value) is bool),
    'logprobs': _rule(lambda value: type(value) is bool),
    'parallel_tool_calls': _rule(lambda value: type(value) is bool),
    'reasoning_effort': _rule(_effort),
    'stop': _rule(lambda value: isinstance(value, str)
                 or isinstance(value, list) and len(value) <= 4096
                 and all(isinstance(item, str) for item in value)),
    'stop_token_ids': _rule(lambda value: isinstance(value, list) and len(value) <= 4096
                           and all(_integer(item) for item in value)),
    'chat_template_kwargs': FieldRule(_template, 'bad_chat_template_kwargs'),
    'response_format': FieldRule(_response_format, 'response_format'),
    'tool_choice': FieldRule(_tool_choice, 'tool_choice'),
})
DEFAULT_REQUEST_CAPABILITIES = RequestCapabilities(_rules)
GENERATION_FIELDS = frozenset(_rules)


def portable_generation_settings(extra, capabilities=DEFAULT_REQUEST_CAPABILITIES):
    """Copy valid generation options while removing provider/transport metadata."""
    result = {}
    for key, value in extra.items():
        if key not in capabilities.rules or key in {'stream', 'stream_options', 'tool_choice'}:
            continue
        value = deepcopy(value)
        if key == 'chat_template_kwargs' and isinstance(value, dict):
            value = {name: item for name, item in value.items() if name in TEMPLATE_FIELDS}
        if key == 'response_format' and isinstance(value, dict):
            value = {name: item for name, item in value.items() if name in {'type', 'json_schema'}}
            schema = value.get('json_schema')
            if isinstance(schema, dict):
                value['json_schema'] = {name: item for name, item in schema.items()
                                        if name in {'name', 'description', 'schema', 'strict'}}
        if capabilities.check({key: value}) == '':
            result[key] = value
    return result
