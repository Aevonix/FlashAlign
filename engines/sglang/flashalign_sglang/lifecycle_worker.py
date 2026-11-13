# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Dedicated scheduler control path. Never invoked from model request fields."""
import gc
import json
import os
from pathlib import Path

from .flashalign_admin_server import runtime_counters

from .lifecycle_support import GENERATION, TRANSACTION, checked_profile, private_directory, recovery_generations, strict_json

_RUNNERS = []
_PREPARED = {}
_COMMITTED = set()


def register_runner(runner):
    from . import runtime
    if not hasattr(runtime, "_flashalign_model_contracts"):
        runtime._flashalign_model_contracts = {json.dumps(p.model, sort_keys=True) for p in runtime.registry()[0].values()}
    if all(value is not runner for value in _RUNNERS):
        _RUNNERS.append(runner)


def _sync():
    import torch
    devices = {str(runner.device) for runner in _RUNNERS}
    for device in devices:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)


def _profile(spec):
    if not isinstance(spec, dict) or set(spec) != {"id", "digest"} or not GENERATION.fullmatch(spec["id"]):
        raise ValueError("Invalid generation specification")
    root = private_directory(os.environ["FLASHALIGN_LIFECYCLE_STATE"])
    directory = private_directory(root / "generations" / spec["id"])
    return checked_profile(directory, spec["digest"])


def _prepare(spec):
    from . import runtime
    profiles, _ = runtime.registry()
    gid = spec["id"]
    if gid in _PREPARED:
        if _PREPARED[gid] != spec["digest"]:
            raise ValueError("Immutable generation identity changed")
        return
    if not _RUNNERS:
        raise ValueError("No model runners acknowledged lifecycle preparation")
    profile = _profile(spec)
    # Reuse the exact startup tensors when the initial staged snapshot matches.
    profile = next((p for p in profiles.values() if p.digest == profile.digest), profile)
    profiles[gid] = profile
    try:
        for runner in _RUNNERS:
            runtime.install(runner)
            graph = getattr(runner, "flashalign_graph", None)
            if graph is not None:
                graph.prepare_generation(gid, profile)
        _sync()
    except BaseException:
        for runner in _RUNNERS:
            graph = getattr(runner, "flashalign_graph", None)
            if graph is not None:
                graph.retire_generation(gid, profile.digest, quiescent=True)
        profiles.pop(gid, None)
        raise
    _PREPARED[gid] = spec["digest"]


def _selection(salt):
    from . import runtime
    return runtime.registry()[1].from_salt(salt)


def _retire_graphs(gid, digest):
    for runner in _RUNNERS:
        graph = getattr(runner, "flashalign_graph", None)
        if graph is not None:
            graph.retire_generation(gid, digest, quiescent=True)


def cache_references(cache, gid):
    """Inspect the actual prefix-cache owner, including the unified facade."""
    if cache is None:
        raise ValueError("Prefix-cache reference accounting is unavailable")
    tree = getattr(cache, "tree_core", cache)
    try:
        root = getattr(tree, "root_node", None)
    except NotImplementedError:
        root = None
    if root is None:
        if getattr(cache, "is_chunk_cache", lambda: False)():
            return 0  # Chunk caches retain only the requests counted below.
        total = getattr(cache, "total_size", getattr(tree, "total_size", None))
        if not callable(total):
            raise ValueError("Prefix-cache implementation has no reference accounting")
        sizes = total()
        sizes = (sizes,) if type(sizes) is int else sizes
        if (not isinstance(sizes, (tuple, list)) or not sizes
                or any(type(size) is not int or size < 0 for size in sizes)):
            raise ValueError("Invalid prefix-cache size accounting")
        # An opaque core cannot identify individual namespaces. Keep every
        # generation pinned until explicit cache eviction proves it empty.
        return int(any(sizes))
    count, pending, visited = 0, [root], set()
    while pending:
        node = pending.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        children = getattr(node, "children", None)
        if not isinstance(children, dict):
            raise ValueError("Prefix-cache node traversal is unavailable")
        pending.extend(children.values())
        selection = _selection(getattr(getattr(node, "key", None), "cache_salt", None))
        if selection is not None and selection.name == gid:
            count += 1
    return count


def references(scheduler, gid):
    """Count live scheduler and cache references, including disconnected streams."""
    count = 0
    for request in active_requests(scheduler, include_finished=True):
        selection = _selection(getattr(request, "cache_salt", None))
        if selection is not None and selection.name == gid:
            count += 1
    return count + cache_references(getattr(scheduler, "tree_cache", None), gid)


def dispatch_generate(scheduler, request):
    if getattr(scheduler, "_flashalign_fenced", False):
        pending = getattr(scheduler, "_flashalign_deferred", None)
        if pending is None:
            pending = scheduler._flashalign_deferred = []
        if len(pending) >= 4096:
            # Stop this scheduler instead of silently bypassing the update fence.
            raise RuntimeError("FlashAlign admission fence queue is full")
        pending.append(request)
        return None
    selection = _selection(getattr(request, "cache_salt", None))
    if selection is not None and os.environ.get("FLASHALIGN_LIFECYCLE_STATE"):
        if selection.name not in _COMMITTED:
            raise RuntimeError("Uncommitted FlashAlign generation reached scheduler admission")
    return scheduler.handle_generate_request(request)


def dispatch_batch_generate(scheduler, request):
    # Iteration yields the original tokenized requests, including each rid and
    # return route. Use the same per-request admission path as scalar input.
    for item in request:
        dispatch_generate(scheduler, item)


def dispatch_abort(scheduler, request):
    prefix = getattr(request, "rid", None)
    matched = [item for item in active_requests(scheduler)
               if getattr(request, "abort_all", False)
               or (isinstance(prefix, str) and isinstance(getattr(item, "rid", None), str)
                   and item.rid.startswith(prefix))]
    scheduler._flashalign_cancellation_matches = getattr(scheduler, "_flashalign_cancellation_matches", 0) + len(matched)
    if getattr(scheduler, "_flashalign_fenced", False):
        pending = getattr(scheduler, "_flashalign_deferred_aborts", None)
        if pending is None:
            pending = scheduler._flashalign_deferred_aborts = []
        if len(pending) >= 4096:
            raise RuntimeError("FlashAlign deferred abort queue is full")
        pending.append(request)
    # Abort existing requests immediately and replay after fenced admissions.
    return scheduler.abort_request(request)


def active_requests(scheduler, *, include_finished=False):
    """Distinct scheduler requests; retirement also retains pending finalization."""
    queues = [getattr(scheduler, "waiting_queue", ()), getattr(scheduler, "_flashalign_deferred", ()),
              getattr(getattr(scheduler, "grammar_manager", None), "grammar_queue", ())]
    dllm = getattr(scheduler, "dllm_manager", None)
    queues.extend((getattr(dllm, "waiting_queue", ()), getattr(dllm, "staging_queue", ())))
    receiver = getattr(scheduler, "mm_receiver", None)
    queues.append([item.recv_req for item in getattr(receiver, "waiting_list", ())])
    sparse = getattr(scheduler, "hisparse_coordinator", None)
    queues.append([item.req for item in getattr(sparse, "ack_staging_queue", ())])
    collect = getattr(scheduler, "collect_inflight_reqs", None)
    if callable(collect):
        queues.append(collect())
    for name in ("running_batch", "cur_batch", "last_batch", "cur_batch_for_debug"):
        queues.append(getattr(getattr(scheduler, name, None), "reqs", ()))
    for name in ("chunked_req", "_pending_chunked_abort_req"):
        item = getattr(scheduler, name, None)
        if item is not None:
            queues.append((item,))
    for result in getattr(scheduler, "result_queue", ()):
        queues.append(getattr(result[0], "reqs", ()))
    seen, output = set(), []
    for queue in queues:
        for request in queue:
            finished = getattr(request, "finished", None)
            if not include_finished and callable(finished) and finished():
                continue
            rid = getattr(request, "rid", None)
            salt = getattr(request, "cache_salt", None)
            identity = (("rid", rid, salt) if isinstance(rid, str) and (salt is None or isinstance(salt, str))
                        else ("object", id(request)))
            if identity not in seen:
                seen.add(identity)
                output.append(request)
    return output


def _apply(scheduler, payload):
    from . import runtime
    if not isinstance(payload, dict) or set(payload) - {"action", "transaction", "generation", "snapshot"}:
        raise ValueError("Invalid lifecycle command")
    action = payload.get("action")
    allowed = {"inspect", "prepare", "fence", "commit", "resume", "abort", "references", "retire", "reconcile", "evict_cache", "finish_bootstrap"}
    if action not in allowed:
        raise ValueError("Unsupported lifecycle command")
    tx = payload.get("transaction", {})
    if not isinstance(tx, dict) or not TRANSACTION.fullmatch(tx.get("id", "")):
        raise ValueError("Invalid lifecycle transaction")
    spec = payload.get("generation")
    gid = spec.get("id") if isinstance(spec, dict) else None
    profiles, _ = runtime.registry()
    detail = {}
    if action in {"prepare", "commit", "abort", "references", "retire"}:
        if not isinstance(spec, dict) or not GENERATION.fullmatch(gid or ""):
            raise ValueError("Missing generation identity")
    if action == "inspect":
        detail = {"runner_count": len(_RUNNERS), "roles": sorted({"native_mtp" if r.is_draft_worker else "target" for r in _RUNNERS}),
                  "fenced": getattr(scheduler, "_flashalign_fenced", False),
                  "active_requests": len(active_requests(scheduler)),
                  "idle": bool(scheduler.is_fully_idle() and not getattr(scheduler, "_flashalign_deferred", [])),
                  "cancellation_matches_total": getattr(scheduler, "_flashalign_cancellation_matches", 0),
                  **runtime_counters("sglang")}
        graphs = [runner.flashalign_graph.status() for runner in _RUNNERS if getattr(runner, "flashalign_graph", None) is not None]
        detail["graph"] = graphs or [{"graph_mode": "eager"}]
        installed = set().union(*(set(getattr(runner, "flashalign_installed_sites", {}).values()) for runner in _RUNNERS))
        detail["expected_sites_by_generation"] = {gid: len(profiles[gid].sites) for gid in _PREPARED}
        detail["installed_sites_by_generation"] = {gid: len(set(profiles[gid].sites) & installed) for gid in _PREPARED}
    elif action == "prepare":
        _prepare(spec)
    elif action == "fence":
        scheduler._flashalign_fenced = True
        _sync()
    elif action == "commit":
        if not getattr(scheduler, "_flashalign_fenced", False) or _PREPARED.get(gid) != spec["digest"]:
            raise ValueError("Commit lacks a matching prepared generation and fence")
        _COMMITTED.add(gid)
    elif action == "resume":
        scheduler._flashalign_fenced = False
        pending = getattr(scheduler, "_flashalign_deferred", [])
        scheduler._flashalign_deferred = []
        for request in pending:
            dispatch_generate(scheduler, request)
        aborts = getattr(scheduler, "_flashalign_deferred_aborts", [])
        scheduler._flashalign_deferred_aborts = []
        for request in aborts:
            scheduler.abort_request(request)
    elif action == "abort":
        if gid not in _COMMITTED:
            _sync()
            _retire_graphs(gid, spec["digest"])
            profiles.pop(gid, None)
            _PREPARED.pop(gid, None)
    elif action == "references":
        detail["references"] = references(scheduler, gid)
    elif action == "retire":
        _sync()
        if references(scheduler, gid):
            raise ValueError("Generation still has scheduler or cache references")
        _retire_graphs(gid, spec["digest"])
        profiles.pop(gid, None)
        _PREPARED.pop(gid, None)
        _COMMITTED.discard(gid)
        gc.collect()
    elif action == "finish_bootstrap":
        _sync()
        for name in tuple(profiles):
            if name not in _PREPARED:
                _retire_graphs(name, profiles[name].digest)
                profiles.pop(name)
        for runner in _RUNNERS:
            graph = getattr(runner, "flashalign_graph", None)
            if graph is not None:
                for gid in _PREPARED:
                    graph.prepare_generation(gid, profiles[gid])
        _sync()
    elif action == "evict_cache":
        if not scheduler.is_fully_idle() or getattr(scheduler, "_flashalign_deferred", []):
            raise ValueError("Cache eviction requires an idle scheduler")
        _sync()
        if not scheduler.flush_cache(empty_cache=False):
            raise ValueError("Scheduler did not evict its cache")
    elif action == "reconcile":
        scheduler._flashalign_fenced = True
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("generations"), dict):
            raise ValueError("Invalid recovery snapshot")
        if not scheduler.is_fully_idle():
            raise ValueError("Recovery requires an idle scheduler")
        wanted = {gid: row["artifact"]["digest"] for gid, row in recovery_generations(snapshot).items()}
        # Deferred requests have not entered scheduler allocation or execution.
        # Preserve them across the cache fence, provided their immutable
        # generation belongs to the restored committed set.
        for request in getattr(scheduler, "_flashalign_deferred", []):
            selection = _selection(getattr(request, "cache_salt", None))
            if selection is not None and selection.name not in wanted:
                raise ValueError("Deferred request refers to an uncommitted generation")
        _sync()
        # Recovery clears all stale cache namespaces before removing resources.
        if not scheduler.flush_cache(empty_cache=False):
            raise ValueError("Recovery cache fence failed")
        for generation in tuple(_PREPARED):
            if generation not in wanted:
                _retire_graphs(generation, _PREPARED[generation])
                profiles.pop(generation, None)
                _PREPARED.pop(generation, None)
        for generation, digest in wanted.items():
            _prepare({"id": generation, "digest": digest})
        _COMMITTED.clear()
        _COMMITTED.update(wanted)
    phase = {"prepare": "prepared", "fence": "fenced", "commit": "committed", "resume": "resumed",
             "abort": "aborted", "retire": "retired", "reconcile": "reconciled"}.get(action, action)
    return {"ok": True, "transaction_id": tx["id"], "generation_id": gid,
            "artifact_digest": spec.get("digest") if spec else None, "phase": phase, **detail}


def handle_control(scheduler, request):
    """Every rank participates, even after a local preparation failure."""
    import torch
    from sglang.srt.managers.io_struct import FlashAlignControlReqOutput
    rank = scheduler.ps.tp_rank
    try:
        payload = strict_json(request.payload)
        result = _apply(scheduler, payload)
    except Exception as error:
        # Exception text may contain local paths. The transport returns only type.
        result = {"ok": False, "error": type(error).__name__}
    result["worker_id"] = "tp-" + str(rank)
    size = scheduler.ps.tp_size
    cohort = [None] * size
    if size > 1:
        torch.distributed.all_gather_object(cohort, result, group=scheduler.tp_cpu_group)
    else:
        cohort[0] = result
    return FlashAlignControlReqOutput(rid=request.rid, payload=json.dumps({"workers": cohort}, separators=(",", ":")))
