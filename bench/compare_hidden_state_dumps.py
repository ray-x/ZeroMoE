#!/usr/bin/env python3
"""Compare per-layer Qwen hidden-state dumps from two decode runs.

Each directory contains ``<prefix>.step<position>.f32`` files and adjacent
``.json`` metadata written by the retired ``ZEROMOE_QWEN4EXP_ALLLAYERS_DUMP``
diagnostic. The tool
reports the earliest layer and token position whose hidden states exceed the
requested RMS or max-absolute tolerance. It uses only the Python standard
library.
"""

import argparse
import array
import json
import math
from pathlib import Path
import sys


def load_run(directory):
    result = {}
    for path in Path(directory).glob("*.step*.f32"):
        meta_path = Path(str(path) + ".json")
        if not meta_path.is_file():
            raise ValueError(f"missing metadata: {meta_path}")
        meta = json.loads(meta_path.read_text())
        position = int(meta["position"])
        layers = int(meta["layers"])
        layer_elements = int(meta["layer_elements"])
        missing_layers = int(meta.get("missing_layers", 0))
        if layers <= 0 or layer_elements <= 0:
            raise ValueError(f"invalid tensor shape in {meta_path}")
        if missing_layers:
            raise ValueError(
                f"{path} is incomplete: {missing_layers} layer snapshot(s) missing")
        raw = path.read_bytes()
        if len(raw) != layers * layer_elements * 4:
            raise ValueError(
                f"{path} has {len(raw)} bytes; expected "
                f"{layers * layer_elements * 4}")
        values = array.array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"non-finite hidden state in {path}")
        if position in result:
            raise ValueError(f"duplicate position {position} in {directory}")
        result[position] = (meta, values)
    if not result:
        raise ValueError(f"no .step*.f32 snapshots in {directory}")
    return result


def compare_layer(reference, candidate, start, count):
    squared = 0.0
    max_abs = 0.0
    mean_delta = 0.0
    for i in range(start, start + count):
        delta = float(candidate[i]) - float(reference[i])
        squared += delta * delta
        mean_delta += delta
        max_abs = max(max_abs, abs(delta))
    return {
        "rms": math.sqrt(squared / count),
        "max_abs": max_abs,
        "mean_delta": mean_delta / count,
    }


def compare_runs(reference_dir, candidate_dir, rms_tolerance, max_tolerance):
    reference = load_run(reference_dir)
    candidate = load_run(candidate_dir)
    positions = sorted(set(reference) & set(candidate))
    if not positions:
        raise ValueError("runs have no matching decode positions")
    missing_reference = sorted(set(candidate) - set(reference))
    missing_candidate = sorted(set(reference) - set(candidate))
    rows = []
    first_divergence = None
    for position in positions:
        ref_meta, ref_values = reference[position]
        cand_meta, cand_values = candidate[position]
        for key in ("layers", "layer_elements", "hidden", "hc", "sequence"):
            if ref_meta.get(key) != cand_meta.get(key):
                raise ValueError(
                    f"shape mismatch at position {position}: "
                    f"{key} {ref_meta.get(key)} != {cand_meta.get(key)}")
        layers = int(ref_meta["layers"])
        layer_elements = int(ref_meta["layer_elements"])
        layer_rows = []
        for layer in range(layers):
            metrics = compare_layer(
                ref_values, cand_values, layer * layer_elements,
                layer_elements)
            metrics["layer"] = layer
            metrics["diverged"] = (
                metrics["rms"] > rms_tolerance or
                metrics["max_abs"] > max_tolerance)
            layer_rows.append(metrics)
            if metrics["diverged"] and first_divergence is None:
                first_divergence = {
                    "position": position,
                    "layer": layer,
                    "rms": metrics["rms"],
                    "max_abs": metrics["max_abs"],
                }
        rows.append({"position": position, "layers": layer_rows})
    return {
        "reference": str(reference_dir),
        "candidate": str(candidate_dir),
        "positions_compared": positions,
        "positions_missing_from_reference": missing_reference,
        "positions_missing_from_candidate": missing_candidate,
        "rms_tolerance": rms_tolerance,
        "max_abs_tolerance": max_tolerance,
        "first_divergence": first_divergence,
        "per_position": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rms-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.reference, args.candidate,
                          args.rms_tolerance, args.max_abs_tolerance)
    serialized = json.dumps(result, indent=2)
    if args.json:
        args.json.write_text(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
