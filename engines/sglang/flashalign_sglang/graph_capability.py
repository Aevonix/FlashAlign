# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Explicit graph execution capabilities."""


def check_graph_capabilities(args, profiles, *, config=None):
    for name in ("enable_two_batch_overlap", "enable_single_batch_overlap", "enable_pdmux"):
        if getattr(args, name, False):
            raise ValueError("FlashAlign graph path rejects overlapping target streams")
    if config is None:
        config = getattr(args, "cuda_graph_config", None)
    enabled = False
    result = {}
    for phase in ("decode", "prefill"):
        entry = config.get(phase) if isinstance(config, dict) else getattr(config, phase, None)
        backend = entry.get("backend") if isinstance(entry, dict) else getattr(entry, "backend", None)
        # Early ServerArgs are intentionally raw in the pinned engine. Explicit
        # JSON takes precedence over convenience selectors and off switches.
        # The launcher supplies both selectors; worker installation separately
        # checks the published, normalized configuration.
        if backend is None:
            backend = getattr(args, "cuda_graph_backend_" + phase, None)
        if backend is None and (getattr(args, "disable_cuda_graph", False)
                                or getattr(args, "disable_" + phase + "_cuda_graph", False)):
            backend = "disabled"
        if backend is None:
            raise ValueError("Select explicit decode and prefill CUDA graph backends")
        if backend not in ("full", "breakable", "disabled"):
            raise ValueError("Only full/breakable CUDA graphs are accepted by the graph adapter")
        enabled |= backend != "disabled"
        result[phase] = backend
    if not enabled:
        raise ValueError("Graph path requires an enabled CUDA graph phase; use the existing eager path")
    # Higher-rank generations remain supported through per-batch eager routing.
    return result


def require_capture_coverage(capture_counts):
    if not capture_counts or any(count <= 0 for count in capture_counts.values()):
        raise ValueError("FlashAlign graph activation refused: expected sites were not all observed during CUDA capture")
