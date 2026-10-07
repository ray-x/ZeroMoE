"""Tests for the numeric oracle and malformed-dump rejection."""

import array
import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_decode_logits", Path(__file__).parents[1] / "bench/compare_decode_logits.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class LogitsOracleTest(unittest.TestCase):
    def test_uniform_offset_preserves_distribution(self):
        m = oracle.metrics([1.0, 2.0, -1.0], [4.0, 5.0, 2.0], token=1)
        self.assertTrue(m["top1_agrees"])
        self.assertAlmostEqual(m["kl_ref_candidate"], 0.0)
        self.assertAlmostEqual(m["total_variation"], 0.0)
        self.assertAlmostEqual(m["teacher_nll_delta"], 0.0)
        self.assertAlmostEqual(m["centered_rms_delta"], 0.0)
        self.assertEqual(m["max_abs_delta"], 3.0)

    def test_known_binary_distribution(self):
        m = oracle.metrics([math.log(0.75), math.log(0.25)],
                           [math.log(0.25), math.log(0.75)], token=0)
        self.assertFalse(m["top1_agrees"])
        self.assertAlmostEqual(m["kl_ref_candidate"], 0.5 * math.log(3))
        self.assertAlmostEqual(m["total_variation"], 0.5)
        self.assertAlmostEqual(m["teacher_nll_delta"], math.log(3))
        self.assertEqual(m["reference_top1_candidate_rank"], 2)

    def test_extreme_logits_do_not_underflow_kl(self):
        m = oracle.metrics([1000.0, -1000.0], [-1000.0, 1000.0])
        self.assertAlmostEqual(m["kl_ref_candidate"], 2000.0)
        self.assertEqual(m["total_variation"], 1.0)

    def test_dump_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logits"
            good = struct.pack("=II", 0, 2) + array.array("f", [1.0, 2.0]).tobytes()
            path.write_bytes(good)
            self.assertEqual(len(list(oracle.records(path))), 1)
            for bad in (good[:-1], good[:7], struct.pack("=II", 1, 2) + good[8:],
                        struct.pack("=II", 0, 2) + array.array("f", [math.nan, 2.0]).tobytes()):
                path.write_bytes(bad)
                with self.assertRaises(ValueError):
                    list(oracle.records(path))

    def test_missing_steps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            record = struct.pack("=II", 0, 2) + array.array("f", [1.0, 2.0]).tobytes()
            a.write_bytes(record)
            b.write_bytes(record)
            with self.assertRaises(ValueError):
                oracle.compare(a, b, expected_steps=2)
            with self.assertRaises(ValueError):
                oracle.compare(a, b, tokens=[])
            b.write_bytes(b"")
            with self.assertRaises(ValueError):
                oracle.compare(a, b)


if __name__ == "__main__":
    unittest.main()
