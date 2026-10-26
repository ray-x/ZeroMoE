#!/usr/bin/env python3
"""Validate the runtime evidence for a CUDA MoE full-graph onboarding run."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="ZeroMoE run log")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-fit", action="store_true",
                      help="require a fitting full graph and zero legacy decode calls")
    mode.add_argument("--allow-memory-fallback", action="store_true",
                      help="accept only an explicit insufficient-memory planner result")
    parser.add_argument("--require-all-expert-layers-on-device", action="store_true",
                        help="also require the Qwen full-fit placement proof line")
    args = parser.parse_args()

    try:
        lines = args.log.read_text(errors="replace").splitlines()
    except OSError as error:
        return fail(f"cannot read {args.log}: {error}")

    caps = [line for line in lines if "[GRAPH-CAPS]" in line]
    gate = [line for line in lines if "[MOE-GRAPH-GATE]" in line]
    fallback = [line for line in lines if "[MOE-GRAPH-FALLBACK]" in line]
    if not caps:
        return fail("no [GRAPH-CAPS] line; this family has no registered graph adapter")
    if not gate:
        return fail("no [MOE-GRAPH-GATE] summary; run did not report graph-path evidence")

    cap_fields = fields(caps[-1])
    gate_fields = fields(gate[-1])
    cap_name = re.search(r"\[GRAPH-CAPS\]\s+([^:]+):", caps[-1])
    if not cap_name or cap_name.group(1).strip() != gate_fields.get("family"):
        return fail("graph capability and execution-gate family do not match")
    try:
        graph_runs = int(gate_fields["graph_decode_runs"])
        layered_tokens = int(gate_fields["layered_s1_forwards"])
        legacy_dispatches = int(gate_fields["legacy_s1_expert_dispatches"])
    except (KeyError, ValueError):
        return fail("[MOE-GRAPH-GATE] is missing valid decode counters")

    policy = gate_fields.get("policy")
    memory_plan = gate_fields.get("memory_plan")
    if args.full_fit:
        if cap_fields.get("decode") != "1":
            return fail("graph decode is not enabled in [GRAPH-CAPS]")
        if policy != "required" or memory_plan != "fits":
            return fail("full-fit run must report policy=required and memory_plan=fits")
        if graph_runs <= 0:
            return fail("graph decode was enabled but no graph decode run was recorded")
        if layered_tokens or legacy_dispatches:
            return fail("full-fit run entered layered decode or the legacy per-expert executor")
    else:
        if policy != "memory-fallback" or memory_plan != "insufficient":
            return fail("fallback requires an explicit insufficient-memory planner result")
        reason = re.search(r"\[MOE-GRAPH-FALLBACK\] planner_reason=(.+)$",
                           fallback[-1]) if fallback else None
        if not reason or not reason.group(1).strip():
            return fail("memory fallback has no logged planner reason")

    if args.require_all_expert_layers_on_device:
        placement_lines = [line for line in lines
                           if "[QWEN35MOE-GRAPH] decode ready:" in line]
        if not placement_lines:
            return fail("missing Qwen all-expert device placement proof")
        placement_line = placement_lines[-1]
        placement = fields(placement_line)
        layer_match = re.search(
            r"expert_layers_on_gpu=(\d+)/(\d+)", placement_line)
        try:
            experts_per_layer = int(placement["experts_per_layer"])
        except (KeyError, ValueError):
            return fail("Qwen placement proof is missing experts_per_layer")
        if (not layer_match or int(layer_match.group(1)) <= 0 or
                int(layer_match.group(1)) != int(layer_match.group(2)) or
                experts_per_layer <= 0):
            return fail("Qwen full-fit run did not place every expert layer on device")

    print("PASS: graph decode policy and execution counters match the requested mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
