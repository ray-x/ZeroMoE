#!/usr/bin/env python3
"""Compare GPU EXL3 bridge logits against CPU-reference FFN outputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from compare_decode_logits import compare


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/data/sdb/models/qwen3.8-flash-next/exl3_3bpw"
DEFAULT_CONFIG = ROOT / "configs/qwen4exp_iq1m_fullgraph_22_26.toml"


def require_idle(min_ram_gib):
    competing = subprocess.run(
        ["pgrep", "-x", "zeromoe"], capture_output=True, text=True
    ).stdout.strip()
    if competing:
        raise RuntimeError(f"another ZeroMoE process is active (PID {competing})")
    try:
        subprocess.run(
            ["nvidia-smi", "-L"], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("NVIDIA management/driver is unavailable") from error
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    available_gib = available_kib / (1024 * 1024)
    if available_gib < min_ram_gib:
        raise RuntimeError(
            f"only {available_gib:.1f} GiB RAM available; "
            f"need at least {min_ram_gib:.1f} GiB"
        )


def make_config(base_config, model_path, out_dir):
    bridge_layers = ", ".join(str(layer) for layer in range(48))
    prefix = (
        f"exl3_ggml_bridge_model_path = {json.dumps(model_path)}\n"
        f"exl3_ggml_bridge_layers = [{bridge_layers}]\n"
        "exl3_gpu_cache_gb = 8.0\n"
        "exl3_cpu_expert_cache_gb = 0.0\n"
    )
    config = out_dir / "qwen4exp_exl3_bridge_all48.toml"
    config.write_text(prefix + base_config.read_text())
    return config


def run_decode(engine, config, out_dir, label, prompt, tokens, extra_env,
               min_ram_gib):
    require_idle(min_ram_gib)
    print(f"Running {label} inference ({tokens} token(s))...", flush=True)
    env = os.environ.copy()
    for key in (
        "ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER",
        "ZEROMOE_EXL3_GGML_CPU_REFERENCE_OUTPUT",
        "ZEROMOE_TEACHER_FORCE_TOKEN_FILE",
        "ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING",
        "ZEROMOE_DECODE_LOGITS_DUMP",
        "ZEROMOE_GENERATED_TOKEN_FILE",
    ):
        env.pop(key, None)
    env["ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS"] = "0"
    env.update(extra_env)
    log_path = out_dir / f"{label}.log"
    stdout_path = out_dir / f"{label}.stdout"
    command = [
        str(engine), "--config", str(config), "--ngen", str(tokens),
        "--raw-prompt", "--log-path", str(log_path),
    ]
    try:
        with stdout_path.open("w") as stdout:
            subprocess.run(
                command, input=f"{prompt}\n:q\n", text=True, cwd=ROOT,
                env=env, stdout=stdout, stderr=subprocess.STDOUT, check=True,
            )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{label} inference failed; inspect {stdout_path} and {log_path}"
        ) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "build_ggmlcuda/zeromoe")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="This is an EXL3 bridge parity check.")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--min-ram-gib", type=float, default=110.0)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if not args.engine.is_file() or not args.base_config.is_file():
        parser.error("engine or base config does not exist")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    require_idle(args.min_ram_gib)
    config = make_config(args.base_config, args.model, args.out_dir)

    token_path = args.out_dir / "teacher.tokens"
    run_decode(
        args.engine, config, args.out_dir, "seed", args.prompt, args.tokens,
        {"ZEROMOE_GENERATED_TOKEN_FILE": str(token_path)}, args.min_ram_gib,
    )
    teacher_tokens = [int(value) for value in token_path.read_text().split()]
    if not teacher_tokens:
        raise RuntimeError("seed run produced no teacher-forcing tokens")
    print(f"Captured {len(teacher_tokens)} teacher token(s).", flush=True)

    for label, reference_output in (("cpu_reference", True), ("gpu_bridge", False)):
        extra_env = {
            "ZEROMOE_TEACHER_FORCE_TOKEN_FILE": str(token_path),
            "ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING": "1",
            "ZEROMOE_DECODE_LOGITS_DUMP": str(args.out_dir / f"{label}.logits"),
        }
        if reference_output:
            extra_env["ZEROMOE_EXL3_GGML_VERIFY_CPU_LAYER"] = "all"
            extra_env["ZEROMOE_EXL3_GGML_CPU_REFERENCE_OUTPUT"] = "1"
        run_decode(
            args.engine, config, args.out_dir, label, args.prompt,
            len(teacher_tokens), extra_env, args.min_ram_gib,
        )

    result = compare(
        args.out_dir / "cpu_reference.logits",
        args.out_dir / "gpu_bridge.logits",
        teacher_tokens,
        expected_steps=len(teacher_tokens),
    )
    result_path = args.out_dir / "logit_comparison.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Detailed comparison: {result_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"EXL3 logits parity: {error}", file=sys.stderr)
        sys.exit(1)
