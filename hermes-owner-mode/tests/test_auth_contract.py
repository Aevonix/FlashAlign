# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
import os
from pathlib import Path
import tempfile
import unittest

import test_owner_mode
from owner_mode.auth import sign
from owner_mode.auth_contract import authenticate_header, read_signing_key


class AuthContractTests(unittest.TestCase):
    def test_wire_grammar_and_exact_window(self):
        key, raw, now = b'k' * 32, b'{"owner_mode":"test"}', 2_000_000_000
        for nonce in ('n' * 16, 'aBcD_019-' * 14):
            header = sign(key, raw, now=now, nonce=nonce)
            for at in (now - 60, now, now + 60):
                verified = authenticate_header(key, raw, [header], now=at)
                self.assertEqual(verified.nonce, nonce)
                self.assertEqual(verified.expires_at, now + 61)
            for at in (now - 60.01, now + 60.01, float('nan')):
                self.assertIsNone(authenticate_header(key, raw, [header], now=at))
            for headers in ([], [header, header], [header + '\n'], [None]):
                self.assertIsNone(authenticate_header(key, raw, headers, now=now))
            self.assertIsNone(authenticate_header(key, raw + b' ', [header], now=now))
        for nonce in ('n' * 15, 'n' * 129, 'x:y' * 8, 'é' * 16):
            self.assertIsNone(authenticate_header(key, raw, [sign(key, raw, now=now, nonce=nonce)], now=now))

    def test_key_refuses_links_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'key'
            path.write_bytes(b'k' * 32)
            path.chmod(0o600)
            self.assertEqual(read_signing_key(path), b'k' * 32)
            path.chmod(0o400)
            with self.assertRaises(ValueError): read_signing_key(path)
            path.chmod(0o600)
            link = Path(tmp) / 'linked'
            os.link(path, link)
            with self.assertRaises(ValueError): read_signing_key(path)
            link.unlink()
            link.symlink_to(path)
            with self.assertRaises(OSError): read_signing_key(link)
