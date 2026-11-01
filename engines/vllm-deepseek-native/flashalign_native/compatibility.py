# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Explicit request controls for the pinned DeepSeek renderer."""
from .protocol import FieldError

EFFORTS={'none','low','high','xhigh','max'}


def normalize_chat_request(body):
    supplied=body.get('chat_template_kwargs')
    if supplied is not None and not isinstance(supplied,dict):
        raise FieldError('chat_template_kwargs must be an object')
    kwargs=dict(supplied or {})
    choices=[kwargs[name] for name in ('thinking','enable_thinking') if name in kwargs]
    if any(type(value) is not bool for value in choices):
        raise FieldError('DeepSeek thinking controls must be booleans')
    if choices and any(value!=choices[0] for value in choices):
        raise FieldError('DeepSeek thinking and enable_thinking controls conflict')
    top=body.get('reasoning_effort')
    nested=kwargs.get('reasoning_effort')
    if top is not None and (not isinstance(top,str) or top not in EFFORTS):
        raise FieldError('DeepSeek reasoning_effort must be none, low, high, xhigh or max')
    if nested is not None and not (type(nested) is int and 1<=nested<=100 or
                                  isinstance(nested,str) and nested in EFFORTS):
        raise FieldError('Unsupported DeepSeek chat-template reasoning effort')
    if top is not None and nested is not None and top!=nested:
        raise FieldError('DeepSeek reasoning effort controls conflict')
    effort=top if top is not None else nested
    thinking=choices[0] if choices else None
    if effort=='none':
        if thinking is True:
            raise FieldError('Thinking true conflicts with reasoning_effort none')
        thinking=False
    elif effort is not None and thinking is None:
        thinking=True
    if thinking is not None:
        # Native build_chat_params may insert enable_thinking from top-level
        # effort. The tokenizer ORs both aliases after default merging, so both
        # must explicitly agree when either was selected by the caller.
        kwargs.update(thinking=thinking,enable_thinking=thinking)
    if effort is not None:
        kwargs['reasoning_effort']=effort
    if kwargs or supplied is not None:
        body['chat_template_kwargs']=kwargs
    return body
