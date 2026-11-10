# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Trusted artifact and operator extensions, registered by installed code only.

Manifest strings select existing registrations. They never name imports,
entry points, or executable validators.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import re
import threading
from types import MappingProxyType
from typing import Callable


def _id(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', value) is None:
        raise ValueError('Invalid profile adapter ID')
    return value


@dataclass(frozen=True)
class TensorRule:
    shape: tuple[int, ...]
    validation: str = 'finite'

    def __post_init__(self):
        shape = tuple(self.shape)
        if not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('Invalid profile tensor shape')
        if self.validation not in ('finite', 'positive', 'basis'):
            raise ValueError('Unknown profile tensor validation')
        if self.validation == 'basis' and len(shape) != 2:
            raise ValueError('Basis validation requires a matrix')
        object.__setattr__(self, 'shape', shape)


@dataclass(frozen=True)
class OperatorLayout:
    tensors: dict[str, TensorRule]

    def __post_init__(self):
        tensors = dict(self.tensors)
        if not tensors or len(tensors) > 16384:
            raise ValueError('Invalid operator tensor inventory')
        for name, rule in tensors.items():
            if (not isinstance(name, str) or not name or len(name) > 512
                    or not isinstance(rule, TensorRule)):
                raise ValueError('Invalid operator tensor rule')
        object.__setattr__(self, 'tensors', MappingProxyType(tensors))


@dataclass(frozen=True)
class OperatorContext:
    profile: dict
    hidden_size: int
    sites: list


@dataclass(frozen=True)
class OperatorAdapter:
    id: str
    requirements: frozenset[str]
    layout: Callable[[OperatorContext], OperatorLayout]
    schemas: frozenset = frozenset({None})

    def __post_init__(self):
        _id(self.id)
        requirements = frozenset(self.requirements)
        schemas = frozenset(self.schemas)
        for item in requirements:
            _id(item)
        for item in schemas - {None}:
            _id(item)
        if not schemas or not callable(self.layout):
            raise ValueError('An operator requires schemas and a trusted layout callback')
        object.__setattr__(self, 'requirements', requirements)
        object.__setattr__(self, 'schemas', schemas)


class OperatorRegistry:
    def __init__(self, adapters):
        entries = {}
        for adapter in adapters:
            if not isinstance(adapter, OperatorAdapter) or adapter.id in entries:
                raise ValueError('Duplicate or invalid operator adapter')
            entries[adapter.id] = adapter
        self._entries = MappingProxyType(entries)

    def get(self, name):
        return self._entries.get(name) if isinstance(name, str) else None

    def ids(self):
        return frozenset(self._entries)

    def with_adapter(self, adapter):
        return OperatorRegistry((*self._entries.values(), adapter))


@dataclass(frozen=True)
class ArtifactFormatAdapter:
    id: str
    format: str | None
    validate: Callable

    def __post_init__(self):
        _id(self.id)
        if self.format is not None:
            _id(self.format)
        if not callable(self.validate):
            raise ValueError('An artifact format requires a trusted validator')


class ArtifactFormatRegistry:
    def __init__(self, adapters):
        entries = {}
        names = set()
        for adapter in adapters:
            if (not isinstance(adapter, ArtifactFormatAdapter) or adapter.format in entries
                    or adapter.id in names):
                raise ValueError('Duplicate or invalid artifact format')
            entries[adapter.format] = adapter
            names.add(adapter.id)
        self._entries = MappingProxyType(entries)

    def get(self, name):
        return self._entries.get(name) if name is None or isinstance(name, str) else None

    def with_adapter(self, adapter):
        return ArtifactFormatRegistry((*self._entries.values(), adapter))

    def adapters(self):
        return tuple(self._entries.values())


def _builtin_layout(context):
    from .profiles import _builtin_operator_layout
    return _builtin_operator_layout(context)


def _dense(context):
    from .profiles import _validate_dense_profile
    return _validate_dense_profile(context)


def _native(context):
    from .profiles import _validate_native_profile
    return _validate_native_profile(context.root, context.manifest, context.manifest_bytes,
                                    include_audit=context.include_audit,
                                    allow_missing_payload=context.allow_missing_payload)


_REQUIREMENTS = {
    'projection': set(),
    'row_gain_projection_v1': {'row_gain'},
    'row_normalized_projection_v1': {'row_gain', 'row_norm_preconditioning'},
    'input_low_rank_delta_v1': {'matrix_input'},
    'weighted_output_projection_v1': {'per_direction_strength'},
    'residual_writer_group_projection_v1': {'writer_groups', 'text_embedding_before_multimodal_merge',
                                           'ple_value_projection', 'native_mtp_writer_groups'},
}
DEFAULT_OPERATORS = OperatorRegistry(
    OperatorAdapter(name, frozenset(requirements), _builtin_layout,
                    frozenset({None, 'flashalign.grouped_projection.v1'}) if name == 'projection' else frozenset({None}))
    for name, requirements in _REQUIREMENTS.items())
DEFAULT_ARTIFACT_FORMATS = ArtifactFormatRegistry((
    ArtifactFormatAdapter('dense-safetensors', None, _dense),
    ArtifactFormatAdapter('native-wo-b', 'flashalign.native_wo_b.v1', _native),
))
_operators = DEFAULT_OPERATORS
_formats = DEFAULT_ARTIFACT_FORMATS
_loaded = False
_lock = threading.RLock()


def _load_installed():
    global _operators, _formats, _loaded
    if not _loaded:
        from .channel_plugins import entry_point_context, load_adapters
        operators, formats = _operators, _formats
        for entry, adapter in load_adapters('flashalign.profile_adapters', (OperatorAdapter, ArtifactFormatAdapter),
                                           with_sources=True):
            with entry_point_context(entry):
                if isinstance(adapter, OperatorAdapter):
                    operators = operators.with_adapter(adapter)
                else:
                    formats = formats.with_adapter(adapter)
        _operators, _formats, _loaded = operators, formats, True


def operator_adapters():
    with _lock:
        _load_installed()
        return _operators


def artifact_formats():
    with _lock:
        _load_installed()
        return _formats


def register_operator(adapter):
    global _operators
    with _lock:
        _operators = _operators.with_adapter(adapter)


def register_artifact_format(adapter):
    global _formats
    with _lock:
        _formats = _formats.with_adapter(adapter)


@contextmanager
def using_profile_adapters(*, operators=None, formats=None):
    """Scoped registry injection for trusted setup code and contract tests."""
    global _operators, _formats, _loaded
    if operators is not None and not isinstance(operators, OperatorRegistry):
        raise ValueError('Invalid operator registry')
    if formats is not None and not isinstance(formats, ArtifactFormatRegistry):
        raise ValueError('Invalid artifact registry')
    with _lock:
        previous = _operators, _formats, _loaded
        _operators, _formats = operators or _operators, formats or _formats
        _loaded = True
        try:
            yield
        finally:
            _operators, _formats, _loaded = previous
