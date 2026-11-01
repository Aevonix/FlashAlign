# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Early import bootstrap for the exact pinned native engine.

Spawned EngineCore processes import graph classes before general plugins. The
finder must therefore exist before the real vLLM initializer runs. The real
package identity and resource loader are retained; no installed file is edited.
"""
import hashlib as _fa_hashlib
import importlib.util as _fa_importlib_util
import os as _fa_os
from pathlib import Path as _FaPath


_fa_expected_init = 'fd3708e5a13abe98c566afd79a6b1987cef70c0548bb19c1218ff6a7f9d43346'
_fa_root_value = _fa_os.environ.get('FLASHALIGN_NATIVE_ENGINE_ROOT', '')
if not _fa_root_value or not _FaPath(_fa_root_value).is_absolute():
    raise RuntimeError('Explicit pinned FLASHALIGN_NATIVE_ENGINE_ROOT is required')
_fa_engine_root = _FaPath(_fa_root_value).resolve(strict=True)
_fa_original_init = _fa_engine_root / '__init__.py'
if (_fa_original_init.is_symlink() or not _fa_original_init.is_file()
        or _fa_original_init.stat().st_size > 1024 * 1024):
    raise RuntimeError('Expected a regular pinned vLLM initializer')
_fa_source = _fa_original_init.read_bytes()
if _fa_hashlib.sha256(_fa_source).hexdigest() != _fa_expected_init:
    raise RuntimeError('Unrecognized pinned vLLM initializer')

_fa_original_spec = _fa_importlib_util.spec_from_file_location(
    __name__, _fa_original_init, submodule_search_locations=[str(_fa_engine_root)])
if _fa_original_spec is None or _fa_original_spec.loader is None:
    raise RuntimeError('Could not preserve the original vLLM package loader')
__file__ = str(_fa_original_init)
__path__ = [str(_fa_engine_root)]
__spec__ = _fa_original_spec
__loader__ = _fa_original_spec.loader
__cached__ = _fa_original_spec.cached
__package__ = __name__

from flashalign_native.overlay import install as _fa_install

_fa_finder = _fa_install(_fa_engine_root)
__flashalign_early_overlay__ = {
    'initializer_sha256': _fa_expected_init,
    'overlay_pins_sha256': _fa_finder.pins_sha256,
}
exec(compile(_fa_source, __file__, 'exec'), globals(), globals())
