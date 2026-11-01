#!/usr/bin/env python3
"""Compare a measured full-graph profile with a candidate model TOML.

This is a review aid, not a performance or memory-fit gate. It highlights
placement/settings drift before loading a large model and shows the candidate
GPU layer set after CPU-affinity exclusions, before VRAM admission.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any


PROFILE_KEYS = (
    "cpu_affinity_layers",
    "graph_device_expert_layers",
    "graph_device_expert_trailing",
    "graph_cpu_threads_decode",
    "omp_max_threads",
    "context_size",
    "max_new_tokens",
    "prefill_batch_tokens",
    "host_ram_gb",
    "ram_preload_all_experts",
    "ram_pinned_budget_gb",
    "pin_ram_fraction",
    "expert_fetch_threads",
    "prefetch_worker_threads",
    "pin_file",
    "prefill_pin_file",
    "qwen4exp_graph_expert_vram_budget_gb",
)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> None:
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"override must be KEY=TOML_VALUE: {raw!r}")
        key, encoded = raw.split("=", 1)
        parsed = tomllib.loads(f"value = {encoded}")["value"]
        values[key] = parsed


def candidate_layers(config: dict[str, Any], n_layers: int) -> list[int] | None:
    count = int(config.get("graph_device_expert_layers", 0) or 0)
    if count <= 0:
        return []
    count = min(count, n_layers)
    trailing = bool(config.get("graph_device_expert_trailing", False))
    first = n_layers - count if trailing else 0
    affinity = {int(x) for x in config.get("cpu_affinity_layers", [])}
    return [layer for layer in range(first, first + count) if layer not in affinity]


def display(value: Any, key: str) -> str:
    if key == "cpu_affinity_layers" and value is not None:
        vals = sorted({int(x) for x in value})
        return f"{len(vals)} layers {vals}"
    if value is None:
        return "<unset>"
    if key in ("pin_file", "prefill_pin_file") and isinstance(value, str):
        return Path(value).name
    return repr(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path, help="measured/reference TOML")
    ap.add_argument("candidate", type=Path, help="new-model candidate TOML")
    ap.add_argument("--n-layers", type=int, required=True,
                    help="transformer layer count used for placement-set preview")
    ap.add_argument("--reference-override", action="append", default=[],
                    metavar="KEY=TOML_VALUE",
                    help="effective CLI/script override applied to the reference run")
    ap.add_argument("--candidate-override", action="append", default=[],
                    metavar="KEY=TOML_VALUE",
                    help="effective startup/hardware value for candidate profile")
    ap.add_argument("--strict-placement", action="store_true",
                    help="exit 1 if the pre-memory-fit candidate GPU layer sets differ")
    args = ap.parse_args()

    try:
        reference = load_toml(args.reference)
        candidate = load_toml(args.candidate)
        apply_overrides(reference, args.reference_override)
        apply_overrides(candidate, args.candidate_override)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    mismatches = []
    print(f"Reference: {args.reference}")
    print(f"Candidate: {args.candidate}")
    print("Setting                               Reference -> Candidate")
    for key in PROFILE_KEYS:
        ref = reference.get(key)
        cand = candidate.get(key)
        same = ref == cand
        marker = "MATCH" if same else "REVIEW"
        print(f"{marker:6} {key:35} {display(ref, key)} -> {display(cand, key)}")
        if not same:
            mismatches.append(key)

    ref_gpu = (reference.get("gpus") or [{}])[0].get("device_id")
    cand_gpu = (candidate.get("gpus") or [{}])[0].get("device_id")
    print(f"REVIEW {'GPU selection':35} {ref_gpu!r} -> {cand_gpu!r} (explicit profile vs portable auto-selection may be intentional)")

    ref_layers = candidate_layers(reference, args.n_layers)
    cand_layers = candidate_layers(candidate, args.n_layers)
    print(f"Reference eligible GPU layers before memory fit: {ref_layers}")
    print(f"Candidate eligible GPU layers before memory fit: {cand_layers}")
    placement_matches = ref_layers == cand_layers
    print("Candidate placement map matches reference before memory fit: "
          + ("yes" if placement_matches else "NO"))
    if args.strict_placement and not placement_matches:
        return 1
    if mismatches:
        print("Review differences above against the target profile; this tool does not decide whether they are safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
