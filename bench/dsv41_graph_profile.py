#!/usr/bin/env python3
"""Run a single-line DeepSeek V4.1 graph prefill with durable logs and RAM guardrails."""

import argparse
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import time


def wait_idle(seconds: int, required_host_gb: float,
              emergency_available_gib: float) -> None:
    start = time.monotonic()
    clear_since = None
    while time.monotonic() - start < seconds:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True)
        processes = subprocess.run(["pgrep", "-x", "zeromoe"],
                                   capture_output=True, text=True)
        available = next(
            (int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
             if line.startswith("MemAvailable:")), 0)
        reserve_gib = max(20.0, emergency_available_gib)
        minimum_available = int(required_host_gb * 1.0e9 + reserve_gib * 1024**3)
        now = time.monotonic()
        if (gpu.returncode == 0 and not gpu.stdout.strip() and
                not processes.stdout.strip() and available >= minimum_available):
            if clear_since is None:
                clear_since = now
            if now - clear_since >= 10:
                return
        else:
            clear_since = None
        time.sleep(2)
    raise TimeoutError("no idle dual-GPU and RAM window")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-budget-gb", type=float, default=100.0,
                        help="temporary process cap for this safe run (50..100 GB)")
    parser.add_argument("--native-reference", action="store_true",
                        help="run the host reference path for a paired trace")
    parser.add_argument("--native-f32-graph", action="store_true",
                        help="use the native F32 quant hook in graph kernels while keeping native prefill")
    parser.add_argument("--quantized-graph", action="store_true",
                        help="run graph expert/dense matmuls on the quantized kernels "
                             "instead of the exact-F32 parity hook (speed mode)")
    parser.add_argument("--trace-layer", type=int,
                        help="save native/graph activations for one layer")
    parser.add_argument("--trace-pos", type=int, default=0,
                        help="save native/graph activations at this position")
    parser.add_argument("--prompt-file", type=Path,
                        help="use this text (one line) instead of the repeated default prompt")
    parser.add_argument("--repetitions", type=int, default=43)
    parser.add_argument("--ngen", type=int, default=1)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument("--run-seconds", type=int, default=600)
    parser.add_argument("--emergency-available-gib", type=float, default=26.0)
    args = parser.parse_args()
    if args.repetitions < 1 or args.ngen < 1:
        parser.error("repetitions and ngen must be positive")
    if not 50.0 <= args.host_budget_gb <= 100.0:
        parser.error("host-budget-gb must stay between 50 and 100 GB in this safe runner")
    if args.emergency_available_gib < 26.0:
        parser.error("emergency MemAvailable floor cannot be lowered below 26 GiB")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = args.config.read_text()
    if not re.search(r"^host_ram_gb\s*=\s*[0-9.]+", config, re.M):
        parser.error("config has no explicit host_ram_gb setting")
    config = re.sub(r"^host_ram_gb\s*=\s*[0-9.]+",
                    f"host_ram_gb = {args.host_budget_gb:.1f}", config,
                    count=1, flags=re.M)
    run_config = output / "config.toml"
    run_config.write_text(config)
    prompt = ("Explain how a mixture-of-experts transformer routes tokens, "
              "in a few clear paragraphs. " * args.repetitions).strip() + "\n"
    if args.prompt_file:
        prompt = " ".join(args.prompt_file.read_text().split()) + "\n"
    # The CLI reads one line per request. Keep the whole batch on one line.
    (output / "prompt.txt").write_text(prompt)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    wait_idle(args.wait_seconds, args.host_budget_gb,
              args.emergency_available_gib)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES="0,1", NVIDIA_TF32_OVERRIDE="0",
               LD_LIBRARY_PATH=str(args.backend.resolve()),
               ZEROMOE_DEEPSEEK_V41_GPU_PREFILL="0" if args.native_reference else "1",
               ZEROMOE_DEEPSEEK_V41_CACHE_MOE_GRAPH="0" if args.native_reference else "1",
               ZEROMOE_DEEPSEEK_V41_NATIVE_F32_GRAPH="0" if args.native_reference else "1")
    if args.native_f32_graph:
        env["ZEROMOE_DEEPSEEK_V41_NATIVE_F32_GRAPH"] = "1"
    if args.quantized_graph:
        env["ZEROMOE_DEEPSEEK_V41_NATIVE_F32_GRAPH"] = "0"
    if args.trace_layer is not None:
        trace_dir = output / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        env["ZEROMOE_DEEPSEEK_V41_TRACE_DIR"] = str(trace_dir)
        env["ZEROMOE_DEEPSEEK_V41_TRACE_LAYER"] = str(args.trace_layer)
        env["ZEROMOE_DEEPSEEK_V41_TRACE_POS"] = str(args.trace_pos)
    print("starting model run", flush=True)
    command = [str(args.binary.resolve()), "--config", str(run_config),
               "--raw-prompt", "--ngen", str(args.ngen)]
    proc = None
    emergency_stop = False
    started = time.monotonic()
    with ((output / "prompt.txt").open("r") as stdin,
          (output / "stdout.log").open("w") as stdout,
          (output / "stderr.log").open("w") as stderr,
          (output / "memory.csv").open("w") as memory):
        memory.write("elapsed_s,mem_available_gib,process_rss_gib,process_locked_gib\n")
        proc = subprocess.Popen(command, stdin=stdin, text=True, env=env,
                                stdout=stdout, stderr=stderr)
        try:
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                status = Path(f"/proc/{proc.pid}/status")
                values = {key: 0 for key in ("VmRSS", "VmLck")}
                if status.exists():
                    for line in status.read_text(errors="replace").splitlines():
                        for key in values:
                            if line.startswith(key + ":"):
                                values[key] = int(line.split()[1]) * 1024
                available = next(
                    (int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:")), 0)
                memory.write(f"{elapsed:.1f},{available/1024**3:.3f},"
                             f"{values['VmRSS']/1024**3:.3f},"
                             f"{values['VmLck']/1024**3:.3f}\n")
                memory.flush()
                if available < args.emergency_available_gib * 1024**3:
                    emergency_stop = True
                    proc.send_signal(signal.SIGTERM)
                    break
                if elapsed > args.run_seconds:
                    proc.send_signal(signal.SIGTERM)
                    break
                time.sleep(2)
        except BaseException:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return_code = proc.wait()
    lines = (output / "stderr.log").read_text(errors="replace").splitlines()
    summary = {"exit_code": return_code, "host_budget_gb": args.host_budget_gb,
               "emergency_stop": emergency_stop,
               "emergency_available_gib": args.emergency_available_gib,
               "repetitions": args.repetitions,
               "prefill": [], "errors": [], "layers": 0,
               "route_ms": 0.0, "upload_ms": 0.0, "payload_ms": 0.0,
               "cache_misses": 0}
    layer_pattern = re.compile(
        r"\[DSV41-CACHE-MOE\] layer=\d+ tokens=\d+ held=\d+ misses=(\d+) "
        r"route_ms=([\d.]+) upload_ms=([\d.]+) payload_ms=([\d.]+)")
    for line in lines:
        if "[PREFILL-DONE]" in line and "tokens in" in line:
            summary["prefill"].append(line)
        if "Runtime Critical Failure" in line or "refusing graph" in line:
            summary["errors"].append(line)
        match = layer_pattern.search(line)
        if match:
            summary["layers"] += 1
            summary["cache_misses"] += int(match[1])
            for field, value in zip(("route_ms", "upload_ms", "payload_ms"), match.groups()[1:]):
                summary[field] += float(value)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
