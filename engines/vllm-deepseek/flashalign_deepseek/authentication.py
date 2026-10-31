# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Persist validated nonce consumption without altering the HMAC protocol."""
import inspect
import time

from flashalign_vllm.core import Gate as MemoryGate


class PersistentGate(MemoryGate):
    def __init__(self, profiles, key, replay_store, clock=time.time):
        self._base_persists = 'replay_store' in inspect.signature(MemoryGate.__init__).parameters
        if self._base_persists:
            super().__init__(profiles, key, clock=clock, replay_store=replay_store)
        else:
            super().__init__(profiles, key, clock=clock)
        self._persistent_store = replay_store

    def authorize(self, raw, body, headers):
        selection = super().authorize(raw, body, headers)
        if self._base_persists or selection is None:
            return selection
        # The superclass has already checked the unique header's exact
        # grammar, timestamp window and HMAC over the unchanged request body.
        _, timestamp, nonce, _ = headers[0].split(':', 3)
        if not self._persistent_store.consume(nonce, int(timestamp)+61, now=self.clock()):
            return None
        return selection
