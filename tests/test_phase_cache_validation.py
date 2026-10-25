"""Ensure the transition gate checks decode frames, not only prefill logits."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("phase_validation", Path(__file__).resolve().parents[1] / "bench/phase_cache_transition.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def write(path, frames):
    with path.open("wb") as f:
        for step, values in enumerate(frames):
            f.write(struct.pack("<II", step, len(values)))
            f.write(struct.pack("<" + "f"*len(values), *values))


class ValidationTest(unittest.TestCase):
    def test_changed_decode_is_detected_even_when_prefill_matches(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/"a.bin", Path(d)/"b.bin"
            write(a, [[1, 0], [1, 0]])
            write(b, [[1, 0], [0, 1]])
            result = validation.compare_requests(a, b)
            self.assertEqual(result["steps"], 2)
            self.assertLess(result["cosine"], 0.999)
            write(b, [[1, 0]])
            with self.assertRaises(AssertionError):
                validation.compare_requests(a, b)

    def test_nonfinite_decode_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.bin"
            write(p, [[1, 0], [float("nan"), 1]])
            with self.assertRaises(AssertionError):
                validation.all_logits(p)


if __name__ == "__main__":
    unittest.main()
