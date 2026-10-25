#!/usr/bin/env python3
"""
diff_trace.py — localize where ZeroMoE's residual-stream |hidden| norm
diverges from colibri's, per layer and per sub-block.

Both engines emit, at log_level VERBOSE, one line per (layer, stage):
    [TRACE l=N] in       |hidden|=X        # residual before the layer's attn
    [TRACE l=N] post-attn|hidden|=X        # residual after the attn residual-add
    [TRACE l=N] post-ffn |hidden|=X        # residual after the FFN/MoE residual-add

colibri and ZeroMoE print the SAME lines (colibri lacks the |attn_out|/|moe_out|
inner magnitudes, but the |hidden| field is in the same place). So a single regex
parses both. We take the FIRST occurrence of each (layer, stage) — i.e. the
PREFILL pass of the shared prompt — so sampling divergence later in decode
doesn't pollute the comparison.

Usage:
    ./diff_trace.py <zeromoe_verbose.log> <colibri_trace.log>

Output: a per-layer table of |hidden| for each stage + the ZeroMoE/colibri
ratio, and the layer indices where the ratio first crosses 2x / 5x / 10x / 30x
(in either direction — colibri larger counts too).

CAVEAT: the two engines use DIFFERENT weight quantizations of the same base
model (ZeroMoE = GGUF IQ2_M; colibri = its own int4 requant of the FP8
checkpoint). So residual norms will NOT be bit-identical even with correct math
— expect per-layer ratios to wander within ~0.5x–2x. A SUSTAINED, ACCELERATING
drift toward 10x–30x is the bug signal (the colibri author's prior observation:
~2x at layer 0 compounding to ~30x by the last layers).
"""
import re
import sys

STAGES = ["in", "post-attn", "post-ffn"]
# Matches both ZeroMoE "[VERBOSE] [TRACE l=N] ..." and colibri "[TRACE l=N] ...".
# Note: ZeroMoE's post-attn/post-ffn lines carry an inner magnitude BEFORE
# |hidden| (e.g. "... post-ffn  |moe_out|=0 |hidden|=0.31"), so we can't use
# [^|]* between the stage and |hidden| — use non-greedy .*? anchored on |hidden|=.
LINE = re.compile(
    r"TRACE l=(\d+)\]\s*(in|post-attn|post-ffn).*?\|hidden\|=([0-9.eE+-]+)"
)


def parse(path):
    """Return {(layer:int, stage:str): float} for the FIRST occurrence of each."""
    seen = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            layer = int(m.group(1))
            stage = m.group(2)
            val = float(m.group(3))
            key = (layer, stage)
            if key not in seen:  # first occurrence == prefill pass
                seen[key] = val
    return seen


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    ms = parse(sys.argv[1])
    co = parse(sys.argv[2])
    if not ms:
        sys.exit(f"no [TRACE] lines found in {sys.argv[1]}")
    if not co:
        sys.exit(f"no [TRACE] lines found in {sys.argv[2]}")

    layers = sorted(set(l for (l, _) in ms) | set(l for (l, _) in co))
    print(f"ZeroMoE layers: {min(l for l,_ in ms)}–{max(l for l,_ in ms)} "
          f"({len(set(l for l,_ in ms))} layers)")
    print(f"colibri    layers: {min(l for l,_ in co)}–{max(l for l,_ in co)} "
          f"({len(set(l for l,_ in co))} layers)")
    print()
    print(f"{'layer':>5} {'stage':>9} {'zeromoe':>12} {'colibri':>12} {'ratio MS/co':>12}")
    print("-" * 54)

    thresholds = {2.0: None, 5.0: None, 10.0: None, 30.0: None}  # first layer crossing
    max_ratio = (0.0, -1, "")
    for l in layers:
        for s in STAGES:
            a = ms.get((l, s))
            b = co.get((l, s))
            if a is None or b is None:
                continue
            # guard against divide-by-near-zero at very early layers
            ratio = a / b if b != 0 else float("inf")
            print(f"{l:>5} {s:>9} {a:>12.5g} {b:>12.5g} {ratio:>12.3f}")
            rr = ratio if ratio != float("inf") else 0.0
            if rr > max_ratio[0]:
                max_ratio = (rr, l, s)
            for t in thresholds:
                if thresholds[t] is None and (ratio >= t or (b != 0 and b / a >= t)):
                    thresholds[t] = (l, s, ratio)
    print("-" * 54)
    print(f"\nMax |ratio|: {max_ratio[0]:.2f}x at layer {max_ratio[1]} {max_ratio[2]}")
    print("First (layer, stage) where ratio crosses each threshold (either direction):")
    for t in sorted(thresholds):
        v = thresholds[t]
        if v:
            print(f"  {t:>4.0f}x -> layer {v[0]:>3} {v[1]:>9}  (ratio {v[2]:.2f}x)")
        else:
            print(f"  {t:>4.0f}x -> never crossed")

    # Where does it accelerate? Print the per-layer post-ffn ratio series so the
    # compounding curve is visible at a glance.
    print("\npost-ffn |hidden| ratio per layer (MS/co) — the compounding curve:")
    series = []
    for l in layers:
        a = ms.get((l, "post-ffn"))
        b = co.get((l, "post-ffn"))
        if a is not None and b is not None and b != 0:
            series.append((l, a / b))
    if series:
        lo, hi = series[0][1], max(abs(r) for _, r in series)
        for l, r in series:
            bar = "#" * int(40 * min(abs(r), 30) / 30)
            print(f"  l={l:>3} {r:>7.2f}x {bar}")
        print(f"\n  first layer ratio: {series[0][1]:.2f}x   peak ratio: {hi:.2f}x")


if __name__ == "__main__":
    main()
