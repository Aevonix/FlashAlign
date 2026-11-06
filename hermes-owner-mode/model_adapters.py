# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Trusted model request declarations, separate from generic setup policy."""
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from types import MappingProxyType


@dataclass(frozen=True)
class ModelAdapter:
    id: str
    model_ids: tuple[str, ...]
    request_kind: str | None = None
    request_efforts: tuple[str, ...] = ()
    hermes_efforts: tuple[str, ...] = ()
    hermes_default_effort: str | None = None
    legacy_request_name: str | None = None

    def __post_init__(self):
        if (not isinstance(self.id, str) or not self.id or not self.model_ids
                or any(not isinstance(value, str) or not value for value in self.model_ids)
                or len(set(self.model_ids)) != len(self.model_ids)):
            raise ValueError('Invalid model adapter identity')
        object.__setattr__(self, 'model_ids', tuple(self.model_ids))
        object.__setattr__(self, 'request_efforts', tuple(self.request_efforts))
        object.__setattr__(self, 'hermes_efforts', tuple(self.hermes_efforts))
        if self.request_kind is not None and (not isinstance(self.request_kind, str) or not self.request_kind):
            raise ValueError('Invalid model request codec')
        if self.hermes_default_effort is not None and self.hermes_default_effort not in self.hermes_efforts:
            raise ValueError('Model adapter default effort is not supported')

    def request_declaration(self):
        if self.request_kind is None:
            return None
        result = {'kind': self.request_kind}
        if self.request_efforts:
            result['efforts'] = list(self.request_efforts)
        return result


class ModelAdapterRegistry:
    def __init__(self, adapters=()):
        entries, models = {}, {}
        for adapter in adapters:
            if not isinstance(adapter, ModelAdapter) or adapter.id in entries:
                raise ValueError('Duplicate or invalid model adapter')
            entries[adapter.id] = adapter
            for model in adapter.model_ids:
                if model in models:
                    raise ValueError('Ambiguous model adapter')
                models[model] = adapter
        self._entries = MappingProxyType(entries)
        self._models = MappingProxyType(models)

    def get(self, model_id):
        return self._models.get(model_id) if isinstance(model_id, str) else None

    def with_adapter(self, adapter):
        return ModelAdapterRegistry((*self._entries.values(), adapter))


DEFAULT_MODEL_ADAPTERS = ModelAdapterRegistry((
    ModelAdapter('deepseek-v41', ('deepseek-ai/DeepSeek-V4.1-Flash-1017',),
                 request_kind='boolean_thinking_v1',
                 hermes_efforts=('low', 'high', 'xhigh', 'max'),
                 hermes_default_effort='low', legacy_request_name='deepseek_v41'),
))

_registry = DEFAULT_MODEL_ADAPTERS
_lock = threading.RLock()


def model_adapters():
    with _lock:
        return _registry


def register_model_adapter(adapter):
    """Register trusted installed code without permitting duplicate model IDs."""
    global _registry
    with _lock:
        _registry = _registry.with_adapter(adapter)


@contextmanager
def using_model_adapters(registry):
    global _registry
    if not isinstance(registry, ModelAdapterRegistry):
        raise TypeError('Expected trusted model registry')
    with _lock:
        previous, _registry = _registry, registry
        try:
            yield registry
        finally:
            _registry = previous
