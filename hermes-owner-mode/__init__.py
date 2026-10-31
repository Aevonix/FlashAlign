# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Aevonix FlashAlign: native contact authorization and signed model requests."""

__version__ = "1.0.0"

def register(ctx):
    from .plugin import register as install
    return install(ctx)
