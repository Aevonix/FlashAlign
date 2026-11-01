# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Load pinned model overlays without editing installed engine files."""
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


class OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, engine_root):
        self.engine_root = Path(engine_root).resolve(strict=True)
        package = Path(__file__).resolve().parent
        self.paths = {}
        raw_pins = (package / 'overlay-pins.json').read_bytes()
        self.pins_sha256 = hashlib.sha256(raw_pins).hexdigest()
        pins = json.loads(raw_pins)
        for name, record in pins.items():
            module = 'vllm.' + name[:-3].replace('/', '.')
            original = self.engine_root / name
            overlay = package / 'overlays' / name
            if record['original_sha256'] is None:
                if name != 'flashalign_native_graph.py' or original.exists():
                    raise RuntimeError('Unexpected native helper source: '+name)
            elif hashlib.sha256(original.read_bytes()).hexdigest() != record['original_sha256']:
                raise RuntimeError('Unrecognized native model source: '+name)
            if hashlib.sha256(overlay.read_bytes()).hexdigest() != record['overlay_sha256']:
                raise RuntimeError('Changed FlashAlign model overlay: '+name)
            if module in sys.modules:
                loaded = sys.modules[module]
                filename = getattr(loaded, '__file__', None)
                origin = getattr(getattr(loaded, '__spec__', None), 'origin', None)
                expected = overlay.resolve(strict=True)
                if (filename is None or origin is None
                        or Path(filename).resolve(strict=True) != expected
                        or Path(origin).resolve(strict=True) != expected):
                    raise RuntimeError('Target module was loaded outside the pinned overlay: '+module)
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != record['overlay_sha256']:
                    raise RuntimeError('Loaded overlay source changed: '+module)
            self.paths[module] = overlay

    def find_spec(self, fullname, path=None, target=None):
        source = self.paths.get(fullname)
        return None if source is None else importlib.util.spec_from_file_location(fullname, source)


def install(engine_root):
    finder = OverlayFinder(Path(engine_root))
    for existing in sys.meta_path:
        if isinstance(existing, OverlayFinder):
            if (existing.engine_root != finder.engine_root
                    or existing.pins_sha256 != finder.pins_sha256
                    or existing.paths != finder.paths):
                raise RuntimeError('Conflicting FlashAlign overlay finder')
            return existing
    sys.meta_path.insert(0, finder)
    return finder
