#!/usr/bin/env python3
"""Expected hit rate and VRAM for an Option-B global expert pool of K experts (Q2_0 Qwen3.8-Flash-Next).

usage: bench/option_b_survey_coverage.py [K ...]      (default: the K values in docs/option-b-vram-budget-2026-10-05.md)

Reads tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap (text lines "layer expert count"; every layer routes the
same total, 159,000, so counts compare across layers) and prints, for each K: the share of all 24,576 experts, the expert arena size
(1,382,400 bytes per expert), the free VRAM at decode predicted from the MEASURED baseline (K=8421 -> 9.36 GiB free once the legacy
cache is off), and the survey coverage = share of routed traffic whose expert is in the top-K pool. Coverage is a prediction from
a survey recorded on other prompts, not a live hit rate. Pure Python, no dependencies.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap"
SLAB_BYTES = 1_382_400            # per-expert slab (Q2_0, gate+up+down), ~1.318 MiB
BASELINE_K, BASELINE_FREE_GIB = 8421, 9.36   # measured: right-sized 0/44/4, legacy cache off, steady-state decode
LAYERS, EXPERTS = 48, 512


def main():
    ks = [int(a) for a in sys.argv[1:]] or [8421, 11264, 12700, 13100, 14458, 14554, 14746]
    counts = {}
    for line in SURVEY.read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            counts[(int(p[0]), int(p[1]))] = float(p[2])
    total = sum(counts.values())
    ranked = sorted(counts.values(), reverse=True)
    gib = SLAB_BYTES / 2**30
    print(f"survey: {len(counts)} (layer, expert) entries, {total:,.0f} routed picks; slab {SLAB_BYTES/2**20:.3f} MiB/expert")
    print(f"{'K':>7} {'% of 24576':>11} {'arena GiB':>10} {'free GiB (pred.)':>17} {'survey coverage':>16}")
    for k in ks:
        cov = sum(ranked[:k]) / total
        print(f"{k:7d} {k/(LAYERS*EXPERTS)*100:10.1f}% {k*gib:10.2f} {BASELINE_FREE_GIB-(k-BASELINE_K)*gib:17.2f} {cov*100:15.1f}%")


if __name__ == "__main__":
    main()
