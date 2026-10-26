#!/usr/bin/env python3
"""Gate full-fit MoE onboarding on measured steady decode vs llama-bench."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zeromoe_log", type=Path, help="full-fit ZeroMoE run log")
    parser.add_argument("llama_bench_json", type=Path, help="llama-bench JSON output for a combined pp/tg run")
    parser.add_argument("--n-prompt", type=int, required=True)
    parser.add_argument("--n-gen", type=int, required=True)
    parser.add_argument("--device", required=True,
                        help="llama-bench device name, e.g. CUDA1")
    parser.add_argument(
        "--minimum-ratio", type=float, default=0.90,
        help="minimum ZeroMoE steady-state decode / llama.cpp decode rate (default: 0.90)",
    )
    parser.add_argument(
        "--minimum-tps", type=float,
        help="optional absolute minimum ZeroMoE steady-state decode rate in tokens/s",
    )
    args = parser.parse_args()

    try:
        lines = args.zeromoe_log.read_text(errors="replace").splitlines()
    except OSError as error:
        return fail(f"cannot read ZeroMoE log: {error}")
    gate_lines = [line for line in lines if "[MOE-GRAPH-GATE]" in line]
    if not gate_lines:
        return fail("no [MOE-GRAPH-GATE] summary in ZeroMoE log")
    gate = fields(gate_lines[-1])
    model_lines = [line for line in lines if "[CONFIG] Model:" in line]
    if not model_lines:
        return fail("ZeroMoE log has no model path")
    zero_model = model_lines[-1].split("[CONFIG] Model:", 1)[1].strip()
    decode_prompt_lengths: list[int] = []
    for line in lines:
        marker = "[STATS] "
        if marker not in line:
            continue
        try:
            sample = json.loads(line.split(marker, 1)[1])
            if sample.get("status") == "ok" and int(sample.get("generated_tokens", 0)) >= args.n_gen:
                decode_prompt_lengths.append(int(sample["prompt_tokens"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    if not decode_prompt_lengths:
        return fail(
            f"ZeroMoE log has no successful decode request with at least "
            f"{args.n_gen} generated tokens"
        )
    if any(prompt != args.n_prompt for prompt in decode_prompt_lengths):
        return fail(
            f"ZeroMoE decode prompt lengths {sorted(set(decode_prompt_lengths))} "
            f"do not match --n-prompt {args.n_prompt}"
        )
    try:
        if gate.get("policy") != "required" or gate.get("memory_plan") != "fits":
            return fail("performance acceptance requires policy=required and memory_plan=fits")
        graph_runs = int(gate["graph_decode_runs"])
        graph_run_ms = float(gate["graph_decode_run_ms"])
        layered = int(gate["layered_s1_forwards"])
        legacy = int(gate["legacy_s1_expert_dispatches"])
    except (KeyError, ValueError):
        return fail("MOE-GRAPH-GATE is missing valid full-fit execution timing/counters")
    if graph_runs <= 0 or graph_run_ms <= 0:
        return fail("ZeroMoE recorded no graph decode time")
    if layered or legacy:
        return fail("full-fit performance run entered layered/per-expert decode")
    zeromoe_tps = graph_runs * 1000.0 / graph_run_ms

    try:
        llama_data = json.loads(args.llama_bench_json.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return fail(f"cannot parse llama-bench JSON: {error}")
    rows = llama_data if isinstance(llama_data, list) else [llama_data]
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("n_prompt") != args.n_prompt or row.get("n_gen") != args.n_gen:
            continue
        try:
            avg_ts = float(row.get("avg_ts", 0.0))
        except (TypeError, ValueError):
            continue
        if not row.get("model_filename") or not math.isfinite(avg_ts):
            continue
        if args.device not in str(row.get("devices", "")):
            continue
        if int(row.get("n_gpu_layers", 0)) <= 0 or int(row.get("n_cpu_moe", 0)) != 0:
            continue
        candidates.append(row)
    if not candidates:
        return fail(
            f"no llama-bench combined pp{args.n_prompt}+tg{args.n_gen} result found"
        )
    model_paths = {str(Path(row["model_filename"]).resolve()) for row in candidates}
    if len(model_paths) != 1:
        return fail("matching llama-bench rows use different GGUF paths")
    if str(Path(zero_model).resolve()) not in model_paths:
        return fail("ZeroMoE and llama-bench logs do not name the same GGUF path")
    reference_tps = float(candidates[-1]["avg_ts"])
    if reference_tps <= 0:
        return fail("llama-bench reported a non-positive decode rate")
    ratio = zeromoe_tps / reference_tps
    print(
        f"ZeroMoE full-graph steady decode: {zeromoe_tps:.2f} tok/s; "
        f"llama.cpp: {reference_tps:.2f} tok/s; ratio={ratio:.3f} "
        f"(required >= {args.minimum_ratio:.3f})"
    )
    if args.minimum_tps is not None:
        if args.minimum_tps <= 0:
            return fail("--minimum-tps must be positive")
        if zeromoe_tps < args.minimum_tps:
            return fail(
                f"ZeroMoE steady-state decode {zeromoe_tps:.2f} tok/s is below "
                f"the absolute target {args.minimum_tps:.2f} tok/s"
            )
    if ratio < args.minimum_ratio:
        return fail("performance bar not met; keep model onboarding in progress")
    print("PASS: full-fit steady decode meets the llama.cpp performance bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
