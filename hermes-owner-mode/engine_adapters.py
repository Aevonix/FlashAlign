# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Trusted engine registration for setup orchestration and read-only probes.

Metadata never supplies executable callbacks. Installed application code builds
this registry explicitly or supplies installed entry points; immutable snapshots
can also be injected in tests.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import re
import threading
from types import MappingProxyType

from .setup_runtime import SetupError


@dataclass(frozen=True)
class EngineAdapter:
    id: str
    check: object
    prepare: object
    select: object = None
    probe: object = None
    auto_priority: int | None = None
    uses_backend_kits: bool = True
    requires_local_supervisor: bool = True

    def __post_init__(self):
        if (not isinstance(self.id, str) or re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', self.id) is None
                or not callable(self.check) or not callable(self.prepare)
                or self.select is not None and not callable(self.select)
                or self.probe is not None and not callable(self.probe)
                or self.auto_priority is not None and type(self.auto_priority) is not int
                or type(self.uses_backend_kits) is not bool
                or type(self.requires_local_supervisor) is not bool):
            raise ValueError('Invalid trusted engine adapter')


class EngineAdapterRegistry:
    def __init__(self, adapters=()):
        entries = {}
        for adapter in adapters:
            if not isinstance(adapter, EngineAdapter) or adapter.id in entries:
                raise ValueError('Duplicate or invalid engine adapter')
            entries[adapter.id] = adapter
        self._entries = MappingProxyType(entries)

    def get(self, name):
        adapter = self._entries.get(name) if isinstance(name, str) else None
        if adapter is None:
            raise SetupError('Engine adapter is not installed: '+str(name))
        return adapter

    def ids(self):
        return tuple(self._entries)

    def automatic(self):
        return tuple(sorted((adapter for adapter in self._entries.values()
                             if adapter.auto_priority is not None and adapter.select is not None),
                            key=lambda adapter: (adapter.auto_priority, adapter.id)))

    def with_adapter(self, adapter):
        return EngineAdapterRegistry((*self._entries.values(), adapter))


def _python_check(*args, **kwargs):
    from .backend_setup import _check_python_backend
    return _check_python_backend(*args, **kwargs)


def _python_prepare(*args, **kwargs):
    from .backend_setup import _prepare_python_backend
    return _prepare_python_backend(*args, **kwargs)


def _select(kind):
    def choose(model, profile, variant=None):
        from .backend_setup import _select_kit
        return _select_kit(kind, model, profile, variant)
    return choose


def _native_check(*args, **kwargs):
    from .native_setup import check
    kwargs.pop('proc_root', None)
    return check(*args, **kwargs)


def _native_prepare(*args, **kwargs):
    from .native_setup import prepare
    return prepare(*args, **kwargs)


DEFAULT_ENGINE_ADAPTERS = EngineAdapterRegistry((
    EngineAdapter('tensorfold', _native_check, _native_prepare, uses_backend_kits=False),
    EngineAdapter('vllm', _python_check, _python_prepare, select=_select('vllm'), auto_priority=100),
    EngineAdapter('sglang', _python_check, _python_prepare, select=_select('sglang'), auto_priority=200),
    EngineAdapter('llamacpp', _python_check, _python_prepare, select=_select('llamacpp')),
))
_registry = DEFAULT_ENGINE_ADAPTERS
_loaded = False
_lock = threading.RLock()


def engine_adapters():
    global _registry, _loaded
    with _lock:
        if not _loaded:
            from .channel_plugins import load_adapters
            registry = _registry
            for adapter in load_adapters('flashalign.engine_adapters', EngineAdapter):
                registry = registry.with_adapter(adapter)
            _registry, _loaded = registry, True
        return _registry


def register_engine_adapter(adapter):
    """Register trusted installed code; duplicate IDs cannot replace a backend."""
    global _registry
    with _lock:
        _registry = _registry.with_adapter(adapter)


@contextmanager
def using_engine_adapters(registry):
    """Temporarily install a registry snapshot during setup/testing."""
    global _registry, _loaded
    if not isinstance(registry, EngineAdapterRegistry):
        raise TypeError('Expected trusted engine registry')
    with _lock:
        previous = _registry, _loaded
        _registry, _loaded = registry, True
        try:
            yield registry
        finally:
            _registry, _loaded = previous
