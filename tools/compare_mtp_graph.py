#!/usr/bin/env python3
"""Compare standalone Qwen MTP graph dumps, including reset/chunk invariants."""

import argparse
import math
from pathlib import Path
import struct
import sys


STAGES = {
    "bridge", "post_attention", "post_ffn", "hidden", "logits",
    "router_probabilities", "router_weights", "router_ids",
}


def read_dump(path, min_replays=32):
    with Path(path).open(encoding="ascii") as source:
        header = source.readline().split()
        if len(header) != 6 or header[0] != "ZEROMOE_MTP_DUMP_V1":
            raise ValueError(f"{path}: invalid dump header")
        fingerprint, d, w, v, rows = map(int, header[1:])
        if fingerprint < 0 or d <= 0 or w < d or v <= 0 or rows < min_replays:
            raise ValueError(f"{path}: invalid shape or insufficient replay rows")
        records = {}
        for line_number, line in enumerate(source, 2):
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: truncated record")
            phase, row, stage, count = (
                int(fields[0]), int(fields[1]), fields[2], int(fields[3])
            )
            key = phase, row, stage
            if (phase not in range(3) or row not in range(rows) or
                    stage not in STAGES or count <= 0 or key in records or
                    len(fields) != 4 + count):
                raise ValueError(f"{path}:{line_number}: invalid/duplicate record")
            values = tuple(map(float, fields[4:]))
            if not all(math.isfinite(x) for x in values):
                raise ValueError(f"{path}:{line_number}: non-finite diagnostic")
            required_size = v if stage == "logits" else (
                w if stage in {"bridge", "post_attention", "post_ffn", "hidden"}
                else count
            )
            if count != required_size:
                raise ValueError(f"{path}:{line_number}: diagnostic shape mismatch")
            records[key] = values
        expected = {(p, r, s) for p in range(3) for r in range(rows) for s in STAGES}
        if records.keys() != expected:
            raise ValueError(f"{path}: missing diagnostic rows")
        experts = len(records[0, 0, "router_probabilities"])
        topk = len(records[0, 0, "router_ids"])
        if topk > experts:
            raise ValueError(f"{path}: invalid router top-k")
        for phase, row, stage in records:
            values = records[phase, row, stage]
            if stage.startswith("router_"):
                size = experts if stage == "router_probabilities" else topk
                if len(values) != size:
                    raise ValueError(f"{path}: inconsistent router geometry")
            if stage == "router_ids" and (
                any(x != int(x) or x < 0 or x >= experts for x in values)
                or len(set(values)) != topk
            ):
                raise ValueError(f"{path}: invalid expert IDs")
            baseline = records[0, row, stage]
            if (len(values) != len(baseline) or
                    struct.pack(f"<{len(values)}f", *values) !=
                    struct.pack(f"<{len(baseline)}f", *baseline)):
                raise ValueError(
                    f"{path}: reset/chunk replay changed phase={phase} row={row} {stage}"
                )
        return tuple(header), records


def metrics(reference, actual):
    rr = math.fsum(x * x for x in reference)
    aa = math.fsum(x * x for x in actual)
    dot = math.fsum(x * y for x, y in zip(reference, actual))
    error = math.fsum((x - y) ** 2 for x, y in zip(reference, actual))
    cosine = dot / math.sqrt(rr * aa) if rr and aa else float(rr == aa)
    nrmse = math.sqrt(error / max(rr, 1e-30))
    return cosine, nrmse, max(abs(x - y) for x, y in zip(reference, actual))


def compare(reference, actual, cosine=0.999, nrmse=0.01, min_replays=32):
    rh, ref = read_dump(reference, min_replays)
    ah, got = read_dump(actual, min_replays)
    if rh != ah:
        raise ValueError("fixture fingerprint/dimensions/row count differ")
    worst = {stage: [1.0, 0.0, 0.0] for stage in STAGES - {"router_ids"}}
    failures = []
    for key, expected in ref.items():
        actual_values = got[key]
        if len(expected) != len(actual_values):
            raise ValueError(f"reference/actual diagnostic sizes differ: {key}")
        if key[2] == "router_ids":
            if expected != actual_values:
                failures.append(f"{key}: expert IDs differ")
            continue
        c, n, a = metrics(expected, actual_values)
        summary = worst[key[2]]
        summary[0], summary[1], summary[2] = (
            min(summary[0], c), max(summary[1], n), max(summary[2], a)
        )
        if c < cosine or n > nrmse:
            failures.append(f"{key}: cosine={c:.8f} nrmse={n:.8f} max_abs={a:.8g}")
    return worst, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("actual")
    args = parser.parse_args()
    try:
        worst, failures = compare(args.reference, args.actual)
        for stage, (cosine, nrmse, absolute) in sorted(worst.items()):
            print(f"{stage}: min_cosine={cosine:.8f} max_nrmse={nrmse:.8f} "
                  f"max_abs={absolute:.8g}")
        if failures:
            print(f"FAIL: {len(failures)} per-row parity failures", file=sys.stderr)
            for failure in failures[:10]:
                print(failure, file=sys.stderr)
            return 1
        print("PASS: every row/stage, exact routes, reset and chunked replay")
        return 0
    except (ValueError, OSError, OverflowError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
