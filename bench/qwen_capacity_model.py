#!/usr/bin/env python3
"""Capacity model for Qwen3.8-Flash-Next Q2_0 (48 layers x 512 experts, top-10, ~1.4 MB per expert).
time/token = L*t_fixed + TOPK*L*[p_cpu * t_cpu] with t_cpu = MB/eff_GBps (VRAM-resident routes are free, everything else is in DDR).
t_fixed is fitted to the measured early-ready decode (33.0 tok/s with 8,421 experts resident in VRAM).
usage: qwen_capacity_model.py"""
import sys
USAGE = 'tests/moe_usage_survey/moestream_usage_survey_qwen4exp_usage_tap'
L, TOPK, NE = 48, 10, 512
MB, GBPS = 1.40, 14.8               # Q2_0 expert, effective CPU GB/s from the quant benchmark (937 us per 10 experts)
counts = []
for line in open(USAGE):
    p = line.split()
    if len(p) == 3: counts.append(int(p[2]))
counts.sort(reverse=True)
total = float(sum(counts)); counts += [0] * (L * NE - len(counts))
cum = [0.0]
for c in counts: cum.append(cum[-1] + c / total)
t_cpu = MB / GBPS
def ms(nv, tf): return L * tf + TOPK * L * (1 - cum[min(nv, L * NE)]) * t_cpu
MEAS_NV, MEAS_MS = 8421, 1000 / 33.0
tf = (MEAS_MS - TOPK * L * (1 - cum[MEAS_NV]) * t_cpu) / L
print(f"measured: {MEAS_NV} experts in VRAM -> {MEAS_MS:.1f} ms/token; t_cpu={t_cpu*1000:.0f} us/expert -> fitted t_fixed={tf:.2f} ms/layer")
print(f"CPU-lane share at {MEAS_NV} resident: {(1-cum[MEAS_NV])*100:.1f}% of routes = {TOPK*L*(1-cum[MEAS_NV]):.0f} experts/token\n")
print("| experts in VRAM | GiB | route share VRAM/CPU | ms/token | tok/s | effective MoE/s |")
print("|---|---|---|---|---|---|")
for nv in (0, 1024, 2048, 4096, 6144, 8421, 12288, 16384, 24576):
    m = ms(nv, tf)
    print(f"| {nv} | {nv*MB/1024:.1f} | {cum[nv]*100:.0f}/{(1-cum[nv])*100:.0f}% | {m:.1f} | {1000/m:.1f} | {TOPK*L*1000/m:,.0f} |")
