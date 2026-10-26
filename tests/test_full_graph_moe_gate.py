import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tools" / "check_full_graph_moe_run.py"


class FullGraphMoeGateTests(unittest.TestCase):
    def run_gate(self, log_text, mode, *extra_args):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.log"
            log_path.write_text(log_text)
            return subprocess.run(
                [sys.executable, str(GATE), str(log_path), mode, *extra_args],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_qwen_full_fit_requires_every_expert_layer_on_device(self):
        prefix = (
            "[GRAPH-CAPS] qwen35moe-full-graph: prefill=0 decode=1 cpu_lane=0\n"
            "[MOE-GRAPH-GATE] family=qwen35moe-full-graph policy=required "
            "memory_plan=fits graph_decode_runs=8 layered_s1_forwards=0 "
            "legacy_s1_expert_dispatches=0\n"
        )
        complete = prefix + (
            "[QWEN35MOE-GRAPH] decode ready: expert_layers_on_gpu=40/40 "
            "experts_per_layer=256 expert_arena_bytes=9000000000 "
            "dense_arena_bytes=2000000000 kv_capacity=4096\n"
        )
        args = ("--full-fit", "--require-all-expert-layers-on-device")
        self.assertEqual(self.run_gate(complete, *args).returncode, 0)
        partial = complete.replace("expert_layers_on_gpu=40/40",
                                   "expert_layers_on_gpu=25/40")
        self.assertNotEqual(self.run_gate(partial, *args).returncode, 0)

    def test_full_fit_requires_graph_execution_and_zero_layered_dispatch(self):
        log = (
            "[GRAPH-CAPS] qwen35moe: prefill=1 decode=1 cpu_lane=0\n"
            "[MOE-GRAPH-GATE] family=qwen35moe policy=required memory_plan=fits "
            "graph_decode_runs=31 layered_s1_forwards=0 "
            "legacy_s1_expert_dispatches=0\n"
        )
        result = self.run_gate(log, "--full-fit")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_fit_rejects_legacy_layered_execution(self):
        log = (
            "[GRAPH-CAPS] qwen35moe: prefill=1 decode=1 cpu_lane=0\n"
            "[MOE-GRAPH-GATE] family=qwen35moe policy=required memory_plan=fits "
            "graph_decode_runs=31 layered_s1_forwards=1 "
            "legacy_s1_expert_dispatches=8\n"
        )
        result = self.run_gate(log, "--full-fit")
        self.assertNotEqual(result.returncode, 0)

    def test_full_fit_requires_explicit_memory_fit(self):
        log = (
            "[GRAPH-CAPS] qwen35moe: prefill=1 decode=1 cpu_lane=0\n"
            "[MOE-GRAPH-GATE] family=qwen35moe policy=required memory_plan=unknown "
            "graph_decode_runs=31 layered_s1_forwards=0 "
            "legacy_s1_expert_dispatches=0\n"
        )
        result = self.run_gate(log, "--full-fit")
        self.assertNotEqual(result.returncode, 0)

    def test_memory_fallback_requires_explicit_capacity_reason(self):
        log = (
            "[GRAPH-CAPS] qwen35moe: prefill=1 decode=1 cpu_lane=1\n"
            "[MOE-GRAPH-GATE] family=qwen35moe policy=memory-fallback "
            "memory_plan=insufficient graph_decode_runs=0 layered_s1_forwards=31 "
            "legacy_s1_expert_dispatches=248\n"
            "[MOE-GRAPH-FALLBACK] planner_reason=required_vram_exceeds_available\n"
        )
        result = self.run_gate(log, "--allow-memory-fallback")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
