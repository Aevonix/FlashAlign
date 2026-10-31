# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Optional lm-evaluation-harness adapter. No evaluation dependency at import time."""
import asyncio
import copy

from .benchmark import RESERVED_OPTIONS, validate_options


class SignedEvaluationMixin:
    """Use the upstream task/tokenization logic and sign each exact request body.

    Error bodies are never logged. A failed or downgraded request aborts the
    evaluation rather than creating a blank answer or silently retrying stock.
    """
    def __init__(self, *, flashalign_client, flashalign_profile="off", flashalign_scale=1.,
                 flashalign_options=None, **kwargs):
        if flashalign_client.mode != "verified" or flashalign_client.gateway:
            raise ValueError("Evaluation requires a direct verified-mode Client")
        self.flashalign_client = flashalign_client
        self.flashalign_profile = flashalign_profile
        self.flashalign_scale = flashalign_scale
        self.flashalign_options = copy.deepcopy(flashalign_options or {})
        if set(self.flashalign_options) & RESERVED_OPTIONS:
            raise ValueError("Evaluation options may not override model, prompts or mode")
        validate_options(self.flashalign_options)
        kwargs.update(model=flashalign_client.model, num_concurrent=1, batch_size=1)
        super().__init__(**kwargs)

    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        payload = self._create_payload(self.create_message(messages), generate=generate,
                                       gen_kwargs=copy.deepcopy(gen_kwargs or {}), seed=self._seed,
                                       eos=self.eos_string, **kwargs)
        payload.update(self.flashalign_options)
        payload["stream"] = False
        raw, headers = self.flashalign_client.prepare(payload, profile=self.flashalign_profile,
                                                      scale=self.flashalign_scale)
        response = self.flashalign_client.http.post(self.base_url, content=raw, headers=headers)
        self.flashalign_client._check(response, self.flashalign_profile)
        return response.json()

    async def amodel_call(self, session, sem, messages, *, generate=True, cache_keys=None,
                         ctxlens=None, gen_kwargs=None, **kwargs):
        async with sem:
            outputs = await asyncio.to_thread(self.model_call, messages, generate=generate,
                                              gen_kwargs=gen_kwargs, **kwargs)
        if generate:
            answers = self.parse_generations(outputs=outputs)
        else:
            answers = self.parse_logprobs(outputs=outputs, tokens=messages, ctxlens=ctxlens)
        if any(answer is None for answer in answers):
            raise ValueError("Engine returned an empty evaluation answer")
        if cache_keys:
            for answer, key in zip(answers, cache_keys):
                self.cache_hook.add_partial("generate_until" if generate else "loglikelihood", key, answer)
        return answers


def model_class(api="chat"):
    """Return a class compatible with the pinned upstream API adapter.

    Ordinary MMLU loglikelihood requires the completions API with echo and
    logprobs. A chat-only engine cannot provide that metric through this adapter.
    """
    if api not in ("chat", "completions"):
        raise ValueError("Choose chat or completions")
    from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
    parent = LocalChatCompletion if api == "chat" else LocalCompletionsAPI
    return type("FlashAlignEvaluation", (SignedEvaluationMixin, parent), {})
