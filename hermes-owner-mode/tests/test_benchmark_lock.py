# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("benchmark_lock_io", ROOT / "flashalign/benchmark_io.py")
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


class EvaluationLockTests(unittest.TestCase):
    def test_busy_is_specific_and_retains_value_error_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            with io.run_lock(temp):
                with self.assertRaises(io.EvaluationBusyError) as raised:
                    with io.run_lock(temp):
                        self.fail("Entered a busy evaluation")
                self.assertIsInstance(raised.exception, ValueError)
                self.assertEqual(str(raised.exception), "Another process owns this evaluation")
            with io.run_lock(temp):
                pass

    def test_unsafe_lock_stays_a_validation_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / ".run.lock"
            lock.write_bytes(b"")
            lock.chmod(0o666)
            with self.assertRaises(ValueError) as raised:
                with io.run_lock(temp):
                    self.fail("Entered an unsafe evaluation")
            self.assertNotIsInstance(raised.exception, io.EvaluationBusyError)
            self.assertEqual(str(raised.exception), "Unsafe evaluation lock")


if __name__ == "__main__":
    unittest.main()
