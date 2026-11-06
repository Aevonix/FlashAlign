# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Fixed-address graph execution with immutable generation slots."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .device_state import DeviceRowState, build_site_bank
from .diagnostic_contract import check_counter_delta, check_text_input_boundary, expected_site_rows, sample_rows
from .execution_scope import TargetExecutionScope
from .fused_projection import project_inplace
from .graph_capability import check_graph_capabilities, require_capture_coverage
from .row_contract import (ProfileIdentity, SlotRegistry, graph_batch_reason,
                           graph_memory_bytes, graph_profile_reason, plan_from_batch)


class GraphAdapter:
    def __init__(self, runner, profiles, gate, positions):
        self.runner, self.gate, self.profiles = runner, gate, profiles
        self.activation_dtype = getattr(getattr(runner, "model_config", None), "dtype", None)
        if self.activation_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Graph execution requires a supported model activation dtype")
        slots = int(os.environ.get("FLASHALIGN_GRAPH_PROFILE_SLOTS", "8"))
        self.registry = SlotRegistry(capacity=slots)
        self.eligible, self.fallback = set(), {}
        self._submission = ContextVar("flashalign_graph_submission", default=None)
        self.graph_batches, self.eager_batches, self.last_mode = 0, 0, None
        self.scope = TargetExecutionScope()
        decoder = getattr(runner.model, "model", None)
        if decoder is None or not hasattr(decoder, "layers") or not hasattr(decoder, "embed_tokens"):
            raise ValueError("Unqualified target decoder scope")
        # Wrapping .forward itself covers __call__ and SGLang's direct inner
        # decoder.forward prefill captures. A standalone shared-embedding call
        # in DFlash does not enter this scope and never reaches projection.
        # Body-only prefill replay temporarily substitutes decoder.forward,
        # while the outer target forward still computes embeddings eagerly.
        # Scope both entry points; standalone draft embedding enters neither.
        self.scope.bind_forward(runner.model)
        self.scope.bind_forward(decoder)
        capacity = int(os.environ.get("FLASHALIGN_GRAPH_MAX_ROWS", "65536"))
        self.sites = tuple(sorted(set(positions.values())))
        widths = {}
        for site in self.sites:
            values = {p.hidden for p in profiles.values() if site in p.sites}
            if len(values) != 1:
                raise ValueError("Profile widths disagree at the same model site")
            widths[site] = values.pop()
        budget = int(os.environ.get("FLASHALIGN_GRAPH_BANK_BUDGET", str(64 * 1024**2)))
        self.resident_bytes = graph_memory_bytes(list(widths.values()), slots=slots, rows=capacity, budget=budget,
                                                 diagnostic=os.environ.get("FLASHALIGN_GRAPH_ATTEST") == "1")
        self.state = DeviceRowState(capacity, slots, device=runner.device)
        self.banks, self.active_slots = {}, {}
        self.capture_hook_calls = {site: 0 for site in self.sites}
        self.capture_by_mode, self.alias_metadata = {}, {}
        self.fired, self.before, self.after, self.cpu_banks = {}, {}, {}, {}
        self._capture_checked = False
        self.diagnostic = os.environ.get("FLASHALIGN_GRAPH_ATTEST") == "1"
        self.evidence = Path(os.environ["FLASHALIGN_GRAPH_ATTEST_FILE"]) if self.diagnostic else None
        self.sample_indices = torch.zeros(4, dtype=torch.int64, device=runner.device) if self.diagnostic else None
        self.forward_number = 0
        for site, width in widths.items():
            self.banks[site] = build_site_bank(self.registry, profiles, site, width, device=runner.device)
            self.active_slots[site] = set()
            if self.diagnostic:
                self.fired[site] = torch.zeros(1, dtype=torch.int32, device=runner.device)
                self.before[site] = torch.empty((4, width), dtype=self.activation_dtype, device=runner.device)
                self.after[site] = torch.empty_like(self.before[site])
                self.cpu_banks[site] = tuple(t.cpu() for t in self.banks[site])
        for name, profile in profiles.items():
            self.prepare_generation(name, profile)

    def prepare_generation(self, name, profile):
        identity = ProfileIdentity(name, profile.digest)
        if identity in self.eligible:
            return
        reason = graph_profile_reason(profile, self.sites)
        if reason is not None:
            self.fallback[identity] = reason
            return
        slot = self.registry.reserve(identity)
        if slot is None:
            self.fallback[identity] = "graph_slot_capacity"
            return
        try:
            for site, (bank, scales) in self.banks.items():
                operator = profile.sites.get(site)
                if operator is None:
                    bank[slot].zero_()
                    scales[slot] = 0.
                else:
                    bank[slot].copy_(operator['q'][0])
                    scales[slot] = operator['scale']
                    if operator['scale'] != 0.:
                        self.active_slots[site].add(slot)
                if self.diagnostic:
                    self.cpu_banks[site] = tuple(t.cpu() for t in self.banks[site])
        except BaseException:
            self.registry.release(identity, references=0, completed=True)
            raise
        self.eligible.add(identity)
        self.fallback.pop(identity, None)

    def retire_generation(self, name, digest, *, quiescent):
        if quiescent is not True:
            raise ValueError("Graph retirement requires a completed worker fence")
        identity = ProfileIdentity(name, digest)
        slot = self.registry.release(identity, references=0, completed=True)
        self.eligible.discard(identity)
        self.fallback.pop(identity, None)
        if slot is not None:
            self.state.clear_retired()
            for site, (bank, scales) in self.banks.items():
                bank[slot].zero_()
                scales[slot] = 0.
                self.active_slots[site].discard(slot)

    def status(self):
        return {"graph_mode": self.last_mode or "ready", "graph_batches": self.graph_batches,
                "eager_batches": self.eager_batches, "graph_profile_slots": self.registry.capacity,
                "graph_profile_slots_used": len(self.registry.identities),
                "graph_bank_bytes": self.resident_bytes}

    def _counts(self):
        return torch.cat([self.fired[site] for site in self.sites]).cpu().tolist()

    @contextmanager
    def submission(self, batch):
        selected = [self.gate.from_salt(salt) for salt in (getattr(batch, "flashalign_cache_salts", None) or [])]
        reason = graph_batch_reason(selected, self.eligible, rows=batch.input_ids.numel(), capacity=self.state.capacity)
        try:
            check_text_input_boundary(batch)
        except ValueError:
            reason = "input_layout"
        if reason is not None:
            record = SimpleNamespace(can_run_graph=False, force_eager=True)
            token = self._submission.set((batch, True))
            try:
                yield record
                self.eager_batches += 1
                self.last_mode = "eager"
            finally:
                self._submission.reset(token)
            return
        if not self._capture_checked:
            require_capture_coverage(self.capture_hook_calls)
            self._capture_checked = True
        plan = plan_from_batch(batch, registry=self.registry, decode_salt=self.gate.from_salt,
                               capacity=self.state.capacity)
        record = SimpleNamespace(can_run_graph=None, force_eager=False)
        selected = sample_rows(plan) if self.diagnostic else None
        before = self._counts() if self.diagnostic else None
        if self.diagnostic:
            self.sample_indices.copy_(torch.tensor([i if i is not None else 0 for i in selected], dtype=torch.int64))
        token = self._submission.set((batch, False))
        try:
            with self.state.submission(plan):
                yield record
            if record.can_run_graph:
                self.graph_batches += 1
                self.last_mode = "graph"
            else:
                self.eager_batches += 1
                self.last_mode = "eager"
        finally:
            self._submission.reset(token)
        if self.diagnostic:
            after = self._counts()
            expected = [expected_site_rows(plan, self.active_slots[site]) for site in self.sites]
            deltas = check_counter_delta(before, after, expected)
            parity = self._check_samples(plan, selected)
            self.forward_number += 1
            evidence = {
                "forward": self.forward_number,
                "mode": batch.forward_mode.name,
                "used_graph": record.can_run_graph,
                "live_rows": plan.live_rows,
                "row_roles": {"selected": plan.selected_rows,
                              "off": plan.live_rows - plan.selected_rows},
                "sites": {f"L{s[0]}.{s[1]}": {"expected": e, "actual": d}
                          for s, e, d in zip(self.sites, expected, deltas)},
                "capture_by_mode": {mode: {f"L{s[0]}.{s[1]}": n for s, n in sites.items()}
                                    for mode, sites in self.capture_by_mode.items()},
                "alias_metadata": {f"L{s[0]}.{s[1]}": item for s, item in self.alias_metadata.items()},
                "sample_parity": parity,
            }
            with self.evidence.open("a") as stream:
                stream.write(json.dumps(evidence, sort_keys=True) + "\n")

    def project(self, output, site):
        # Crucially before shape checks, sampling, or any capture test. DFlash
        # calls this target-owned embedding on rank-three block_ids tensors.
        if not self.scope.active():
            return output
        submission = self._submission.get()
        if submission is not None and (submission[1] or site not in self.banks):
            from .runtime import project
            return project(output, submission[0], site, self.profiles, self.gate)
        if torch.cuda.is_current_stream_capturing():
            self.capture_hook_calls[site] += 1
            mode = self.scope.mode() or "unknown"
            counts = self.capture_by_mode.setdefault(mode, {})
            counts[site] = counts.get(site, 0) + 1
        self.alias_metadata[site] = {"shape": list(output.shape), "strides": list(output.stride()),
                                     "is_view": output._base is not None, "dtype": str(output.dtype)}
        if self.diagnostic and output.shape[0]:
            if output.dtype != self.before[site].dtype:
                raise ValueError("Diagnostic activation dtype differs from expected target dtype")
            torch.index_select(output, 0, self.sample_indices, out=self.before[site])
        bank, scales = self.banks[site]
        result = project_inplace(output, self.state, bank, scales, self.fired.get(site))
        if self.diagnostic and output.shape[0]:
            torch.index_select(result, 0, self.sample_indices, out=self.after[site])
        return result

    def _check_samples(self, plan, selected):
        maximum = 0.
        checked = 0
        for site in self.sites:
            before, after = self.before[site].cpu(), self.after[site].cpu()
            q, scales = self.cpu_banks[site]
            for pos, row in enumerate(selected):
                if row is None:
                    continue
                slot = plan.slots[row]
                if slot not in self.active_slots[site]:
                    if not torch.equal(before[pos].view(torch.int8), after[pos].view(torch.int8)):
                        raise ValueError("Off-row sample byte identity failed")
                else:
                    value = before[pos].float()
                    coefficient = torch.tensor(plan.strengths[row], dtype=torch.float32) * scales[slot]
                    reference = (value - (coefficient * torch.sum(value * q[slot])) * q[slot]).to(after.dtype)
                    difference = (after[pos].float() - reference.float()).abs()
                    eps = torch.finfo(after.dtype).eps
                    bound = max(4 * eps, 2e-6) + max(2 * eps, 2e-6) * reference.float().abs()
                    if not torch.isfinite(difference).all() or not torch.all(difference <= bound):
                        raise ValueError("Selected-row aggregate reference parity failed")
                    maximum = max(maximum, difference.max().item())
                checked += 1
        return {"checked_row_sites": checked, "maximum_absolute_error": maximum,
                "off_samples_byte_exact": True, "raw_activations_logged": False}
