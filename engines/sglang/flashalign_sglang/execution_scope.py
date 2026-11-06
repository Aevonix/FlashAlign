# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Target decoder ownership, including shared draft embeddings."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_owner = ContextVar("flashalign_target_decoder_owner", default=None)


class TargetExecutionScope:
    def active(self):
        value = _owner.get()
        return value is not None and value[0] is self

    def mode(self):
        value = _owner.get()
        return value[1] if value is not None and value[0] is self else None

    @contextmanager
    def enter(self, mode=None):
        token = _owner.set((self, mode))
        try:
            yield
        finally:
            _owner.reset(token)

    def guard(self, output, apply):
        return apply(output) if self.active() else output

    def bind_forward(self, module):
        if hasattr(module, "_flashalign_target_scope"):
            raise ValueError("Target decoder already has an ownership scope")
        original = module.forward

        @wraps(original)
        def scoped(*args, **kwargs):
            batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
            mode = getattr(getattr(batch, "forward_mode", None), "name", None)
            with self.enter(mode):
                return original(*args, **kwargs)

        module.forward = scoped
        module._flashalign_target_scope = self
        return original
