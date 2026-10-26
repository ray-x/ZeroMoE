import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "compare_mtp_graph", Path(__file__).parents[1] / "tools/compare_mtp_graph.py"
)
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def fixture_dump():
    lines = ["ZEROMOE_MTP_DUMP_V1 123 2 4 3 32"]
    for phase in range(3):
        for row in range(32):
            for stage in sorted(PARITY.STAGES):
                values = ([0, 1] if stage == "router_ids" else
                          [1, 2, 3] if stage == "logits" else
                          [0.25, 0.75] if stage == "router_weights" else
                          [1, 2, 3, 4])
                lines.append(f"{phase} {row} {stage} {len(values)} " +
                             " ".join(map(str, values)))
    return "\n".join(lines) + "\n"


class MtpParityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.reference = Path(self.temp.name) / "reference"
        self.actual = Path(self.temp.name) / "actual"
        self.text = fixture_dump()
        self.reference.write_text(self.text)
        self.actual.write_text(self.text)

    def test_exact(self):
        _, failures = PARITY.compare(self.reference, self.actual)
        self.assertEqual(failures, [])

    def test_fixture_identity(self):
        self.actual.write_text(self.text.replace("V1 123", "V1 124"))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            PARITY.compare(self.reference, self.actual)

    def test_missing_stage(self):
        self.actual.write_text("\n".join(self.text.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "missing"):
            PARITY.compare(self.reference, self.actual)

    def test_replay_drift(self):
        self.actual.write_text(self.text.replace(
            "1 0 hidden 4 1 2 3 4", "1 0 hidden 4 1 2 3 5"))
        with self.assertRaisesRegex(ValueError, "reset/chunk"):
            PARITY.compare(self.reference, self.actual)

    def test_per_row_gate(self):
        for phase in range(3):
            self.text = self.text.replace(
                f"{phase} 0 hidden 4 1 2 3 4",
                f"{phase} 0 hidden 4 1 2 3 5")
        self.actual.write_text(self.text)
        _, failures = PARITY.compare(self.reference, self.actual)
        self.assertEqual(len(failures), 3)

    def test_routes_are_exact(self):
        self.actual.write_text(self.text.replace("router_ids 2 0 1",
                                                 "router_ids 2 1 0"))
        _, failures = PARITY.compare(self.reference, self.actual)
        self.assertEqual(len(failures), 96)

    def test_non_finite(self):
        self.actual.write_text(self.text.replace("hidden 4 1 2 3 4",
                                                 "hidden 4 nan 2 3 4"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            PARITY.compare(self.reference, self.actual)

    def test_zero_metrics(self):
        self.assertEqual(PARITY.metrics([0, 0], [0, 0]), (1, 0, 0))
        self.assertEqual(PARITY.metrics([0, 0], [1, 0])[0], 0)


if __name__ == "__main__":
    unittest.main()
