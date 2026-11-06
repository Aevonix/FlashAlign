# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
"""Load the in-tree owner_mode package for standalone adapter tests."""
from pathlib import Path
import sys
import types

if 'owner_mode' not in sys.modules:
    module = types.ModuleType('owner_mode')
    module.__path__ = [str(Path(__file__).resolve().parents[3] / 'hermes-owner-mode')]
    sys.modules['owner_mode'] = module
