# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Sender IDs that the channel adapter rejects are silent, routine rejections."""
import unittest

import test_routine_diagnostics as routine


class RejectedSenderIds(unittest.TestCase):
    setUp = routine.RoutineDiagnostics.setUp
    native = routine.RoutineDiagnostics.native
    rejected = routine.RoutineDiagnostics.rejected

    def test_authorized_sms_short_code_is_silent_and_not_pending(self):
        event, gateway = self.native()
        self.store.set_channels(['sms'])
        self.store.set_contact('sms', 'default', '+10000000000', True)
        self.assertTrue(self.store.approved('sms', 'default', '+10000000000'))
        gateway.adapters['whatsapp'].platform = event.source.platform = 'sms'
        event.source.user_id = '00000'
        self.rejected(event, gateway)

    def test_whatsapp_sender_id_in_rejected_form_is_silent_and_not_pending(self):
        event, gateway = self.native()
        event.source.user_id = '10000000000@c.us'
        self.rejected(event, gateway)
