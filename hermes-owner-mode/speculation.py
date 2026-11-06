# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Resolve declared speculative model requirements without inspecting a model."""
from dataclasses import dataclass
import re

from .setup_runtime import SetupError


@dataclass(frozen=True)
class SpeculationPlan:
    method: str
    source: str
    draft_model_architectures: tuple[str, ...] = ()

    @property
    def requires_draft_model(self):
        return self.source == 'external'

    def draft_manifest(self):
        return {'requires_draft_model': self.requires_draft_model,
                'min_parallel': 1,
                'draft_model_architectures': list(self.draft_model_architectures)}


def _methods(value, label, *, empty=False):
    if (not isinstance(value, (list, tuple)) or not empty and not value
            or any(not isinstance(item, str) or re.fullmatch(r'[A-Z][A-Z0-9_]*', item) is None
                   for item in value) or len(set(value)) != len(value)):
        raise SetupError('Invalid SGLang ' + label + ' speculative capability')
    return tuple(value)


def sglang_speculation(metadata, settings=None):
    """One resolver for the wizard and explicit backend preparation.

    Metadata comes from a trusted installed backend. Merely naming a method in
    a request cannot add an algorithm or change whether it needs a draft.
    """
    settings = settings or {}
    methods = _methods(metadata.get('speculative_algorithms', ['DFLASH']), 'algorithm')
    draftless = _methods(metadata.get('draftless_speculative_algorithms', []),
                         'draftless', empty=True)
    native = _methods(metadata.get('native_speculative_algorithms', []), 'native', empty=True)
    if (not set(draftless) <= set(methods) or not set(native) <= set(methods)
            or set(draftless) & set(native)):
        raise SetupError('Invalid SGLang draftless/native speculative capability')
    selected = settings.get('draft_algorithm', methods[0])
    if not isinstance(selected, str):
        raise SetupError('A speculative algorithm must be a string')
    selected = selected.upper()
    if selected not in methods:
        raise SetupError('The active speculative method is unsupported by this SGLang integration; it will not be disabled')
    if selected in draftless:
        source = 'none'
    elif selected in native or 'native_mtp_writer_groups' in metadata.get('required_features', []):
        source = 'target'
    else:
        source = 'external'
    architectures = metadata.get('draft_model_architectures', ['DFlash2DraftModel', 'DFlashDraftModel'])
    if (not isinstance(architectures, (list, tuple))
            or any(not isinstance(item, str) or not item for item in architectures)
            or len(set(architectures)) != len(architectures)):
        raise SetupError('Invalid draft model architecture declaration')
    return SpeculationPlan(selected, source, tuple(architectures))
