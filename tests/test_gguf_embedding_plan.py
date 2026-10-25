"""Check the read-only planner on a tiny synthetic MoE, without loading a model or using CUDA."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


def gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def write_fixture(path):
    tensors = [
        ("token_embd.weight", (64, 100)),
        ("output.weight", (64, 100)),
        ("output_norm.weight", (64,)),
        ("blk.0.ffn_gate_exps.weight", (64, 32, 4)),
        ("blk.0.ffn_up_exps.weight", (64, 32, 4)),
        ("blk.0.ffn_down_exps.weight", (32, 64, 4)),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    header += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("llama")
    offset = 0
    for name, dims in tensors:
        header += gguf_string(name) + struct.pack("<I", len(dims))
        header += struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", 0, offset)  # F32
        n = 4
        for d in dims:
            n *= d
        offset += n
    with path.open("wb") as f:
        f.write(header)
        f.seek((len(header) + 31) // 32 * 32 + offset - 1)
        f.write(b"\0")


class EmbeddingPlanTest(unittest.TestCase):
    def test_cpu_default_and_gpu_override(self):
        binary = Path(os.environ.get("GGUF_TOOL", "build_mimo/gguf_tool")).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "tiny.gguf"
            write_fixture(model)
            reports = {}
            for device in (None, "cpu", "gpu"):
                args = [str(binary), "plan", str(model)]
                if device:
                    args += ["--embedding-device", device]
                p = subprocess.run(args, capture_output=True, text=True, check=True)
                reports[device] = next(s for s in p.stdout.splitlines() if s.startswith("[EMBEDDING-PLAN]"))
            self.assertEqual(reports[None], reports["cpu"])
            self.assertIn("reclaimed_vram_bytes=25600 additional_expert_slots=1", reports[None])
            self.assertIn("device=gpu", reports["gpu"])
            self.assertIn("reclaimed_vram_bytes=0 additional_expert_slots=0", reports["gpu"])
            bad = subprocess.run([str(binary), "plan", str(model), "--embedding-device", "typo"],
                                 capture_output=True, text=True)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("embedding device must be", bad.stderr)


if __name__ == "__main__":
    unittest.main()
