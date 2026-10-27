#!/usr/bin/env python3
"""Capture A, then teacher-force the same prompt and continuation per layout.

Run from the repository root, on an idle GPU. The model must pass gguf_tool
verify first. This is a correctness diagnostic; dump I/O affects throughput.
The supplied prompt must be a single line (the engine's stdin prompt format).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from compare_decode_logits import compare, records

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "configs/testing/qwen4exp_q2_0_option_b_rightsize_split44_x0_cov80.toml"
B = ROOT / "configs/testing/qwen4exp_q2_0_option_b_global_topk48.toml"


def setting(config, key, value):
    pattern = rf"^{re.escape(key)}\s*=.*$"
    if re.search(pattern, config, re.MULTILINE):
        return re.sub(pattern, f"{key} = {value}", config, flags=re.MULTILINE)
    # Root keys must precede the first [[gpus]] table.
    return f"{key} = {value}\n" + config


def layout(name):
    base = A.read_text() if name in ("A", "A_repeat", "C") else B.read_text()
    if name == "C":
        base = setting(base, "qwen4exp_option_b_right_size_top_n", "false")
    elif name == "D":
        base = setting(base, "qwen4exp_option_b_global_top_k", "11264")
    elif name == "V1":
        layers = re.search(r"^qwen4exp_option_b_selected_layers\s*=\s*(\[.*\])",
                           A.read_text(), re.MULTILINE).group(1)
        base = setting(base, "qwen4exp_option_b_selected_layers", layers)
    elif name == "floor1":
        base = setting(base, "qwen4exp_option_b_global_top_k", "48")
        base = setting(base, "qwen4exp_option_b_global_floor_per_layer", "1")
    elif name not in ("A", "A_repeat", "B"):
        raise ValueError(f"unknown layout {name}")
    return base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, default=ROOT.parent / "moestream-build-upstream-master/moestream")
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ngen", type=int, default=64)
    p.add_argument("--gpu", default="0")
    p.add_argument("--cpu-threads", type=int, default=10)
    p.add_argument("--layouts", default="A_repeat,B,C,D,V1,floor1")
    p.add_argument("--resume", action="store_true", help="reuse complete runs in the output directory")
    p.add_argument("--wait-idle", type=int, default=0,
                   help="seconds to wait for competing inference processes before each launch")
    args = p.parse_args()
    if args.ngen <= 0:
        p.error("--ngen must be positive")
    prompt = args.prompt_file.read_text().strip()
    if not prompt or "\n" in prompt:
        p.error("prompt must be a nonempty single line")
    args.out_dir = args.out_dir.resolve()
    args.engine = args.engine.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    (args.out_dir / "prompt.txt").write_text(prompt + "\n")
    manifest = {
        "engine": str(args.engine),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "ngen": args.ngen, "gpu": args.gpu, "cpu_threads": args.cpu_threads,
        "base_configs_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in (A, B)],
        "usage_sha256": hashlib.sha256((ROOT / "tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap").read_bytes()).hexdigest(),
    }
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("resume settings, input files or engine differ from the saved run")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = os.environ.copy()
    # Prevent ambient diagnostics from changing the token history or placement.
    for key in list(env):
        if key.startswith("ZEROMOE_"):
            del env[key]
    env.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS": "0",
        "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING": "1",
        "ZEROMOE_QWEN_CPU_THREADS": str(args.cpu_threads),
        "ZEROMOE_BENCH_IGNORE_STOP": "1",
    })
    prompt_ids = args.out_dir / "prompt.tokens"
    teacher = args.out_dir / "A.tokens"
    summary_path = args.out_dir / "summary.json"
    results = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else {}
    for name in ["A", *args.layouts.split(",")]:
        if hashlib.sha256(args.engine.read_bytes()).hexdigest() != manifest["engine_sha256"]:
            raise RuntimeError("engine changed during validation; use a saved binary for the whole matrix")
        logits = args.out_dir / f"{name}.logits"
        captured = args.out_dir / f"{name}.tokens"
        cfg = args.out_dir / f"{name}.toml"
        config = setting(layout(name), "log_path", json.dumps(str(args.out_dir / f"{name}.engine.log")))
        cfg.write_text(config)
        run_env = env.copy()
        run_env["ZEROMOE_DECODE_LOGITS_DUMP"] = str(logits)
        run_env["ZEROMOE_GENERATED_TOKEN_FILE"] = str(captured)
        if name == "A":
            run_env["ZEROMOE_PROMPT_TOKEN_FILE"] = str(prompt_ids)
        else:
            run_env["ZEROMOE_PROMPT_TOKEN_INPUT_FILE"] = str(prompt_ids)
            run_env["ZEROMOE_TEACHER_FORCE_TOKEN_FILE"] = str(teacher)
            run_env["ZEROMOE_TEACHER_FORCE_SKIP_SAMPLING"] = "1"
        if not (args.resume and logits.exists() and captured.exists()):
            # Recheck before every launch: another session can start after A.
            deadline = time.monotonic() + args.wait_idle
            waiting = False
            while True:
                gpu_procs = subprocess.check_output([
                    "nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader"], text=True).strip()
                engines = subprocess.run(["pgrep", "-x", "-a", "zeromoe"],
                                         capture_output=True, text=True).stdout.strip()
                if not gpu_procs and not engines:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"competing inference process; resume after it exits: {gpu_procs or engines}")
                if not waiting:
                    print(f"Waiting for idle GPU before {name}: {gpu_procs or engines}", flush=True)
                    waiting = True
                time.sleep(5)
            print(f"Running {name} ({args.ngen} tokens)...", flush=True)
            started = time.monotonic()
            with (args.out_dir / f"{name}.stdout").open("w") as stdout:
                subprocess.run([str(args.engine), "--verbosity", "5", "--config", str(cfg), "--predict", str(args.ngen)],
                               input=prompt + "\n:q\n", text=True, stdout=stdout,
                               stderr=subprocess.STDOUT, cwd=ROOT, env=run_env, check=True)
            print(f"Finished {name} in {time.monotonic() - started:.1f}s", flush=True)
        if len(list(records(logits))) != args.ngen:
            raise RuntimeError(f"{name}: incomplete logits dump")
        tokens = [int(x) for x in teacher.read_text().split()]
        if len(tokens) != args.ngen or captured.read_text() != teacher.read_text():
            raise RuntimeError(f"{name}: continuation differs or is incomplete")
        if name != "A":
            result = compare(args.out_dir / "A.logits", logits, tokens, args.ngen)
            (args.out_dir / f"{name}.comparison.json").write_text(json.dumps(result, indent=2) + "\n")
            results[name] = result["summary"]
            s = result["summary"]
            print(f"{name}: top1={s['top1_agreement']:.2%} "
                  f"KL={s['mean_kl_ref_candidate']:.6g} "
                  f"max_delta={s['max_max_abs_delta']:.6g}", flush=True)
            summary_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
