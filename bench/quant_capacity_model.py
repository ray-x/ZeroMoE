#!/usr/bin/env python3
"""Capacity-aware decode model for a quant type: how many experts fit VRAM / DDR / SSD, the share of routed traffic each tier serves
(from the real per-expert usage counts), CPU cost per expert from the measured effective GB/s, and the SSD miss cost.

time/token = N_MOE_LAYERS * t_fixed + TOPK*N_MOE_LAYERS * [ p_ddr * t_cpu + p_ssd * (t_cpu + t_miss) ]   (VRAM-hot routes ~ free)
  t_cpu  = MB_per_expert / eff_GBps(type)            (CPU kernel, measured table in docs/ssd-offload/07-cpu-cost-by-quant.md)
  t_miss = MB_per_expert / ssd_GBps                  (synchronous SSD read in the miss node)
t_fixed (dense/attention/routing per layer) and ssd_GBps are FIT to measured AJ-IQ2_XXS decode runs (24/32/48 GiB DDR).
Placement is the static global ranking by usage (hottest -> VRAM, next -> DDR); a real LRU does a little worse.
Accuracy is NOT modelled: bits-per-weight is shown only as a rough proxy."""
import sys, itertools
USAGE = 'docs/ssd-offload/data/glm53flash_usage_combined_old_plus_survey5.txt'
NL, TOPK, NE = 42, 8, 288
PARAMS = 3 * 4096 * 1536                      # weights per GLM expert (gate+up+down)
# type: (bits per weight, effective CPU GB/s from the Qwen-shape benchmark)
TYPES = {'iq1_m': (1.75, 11.3), 'iq2_xxs': (2.0625, 11.8), 'iq2_xs': (2.3125, 13.0), 'q2_0': (2.5, 14.8), 'iq3_xxs': (3.0625, 13.6),
         'q3_k': (3.4375, 32.7), 'iq4_xs': (4.25, 37.0), 'q4_0': (4.5, 37.5), 'iq4_nl': (4.5, 35.1), 'q4_k': (4.5, 41.8),
         'q5_k': (5.5, 41.7), 'q6_k': (6.5625, 45.4), 'q8_0': (8.5, 48.0)}
counts = {}
for line in open(USAGE):
    p = line.split()
    if len(p) == 3: counts[(int(p[0]), int(p[1]))] = counts.get((int(p[0]), int(p[1])), 0) + int(p[2])
ranked = sorted(counts.values(), reverse=True)
total = float(sum(ranked))
n_experts = NL * NE
ranked += [0] * (n_experts - len(ranked))
cum = [0.0]
for c in ranked: cum.append(cum[-1] + c / total)
def shares(nv, nd):
    nv, nd = min(nv, n_experts), min(nd, n_experts - min(nv, n_experts))
    return cum[nv], cum[nv + nd] - cum[nv], 1.0 - cum[nv + nd]
def tok_ms(mb, gbps_cpu, vram_gib, ddr_gib, t_fixed, ssd_gbps):
    nv, nd = int(vram_gib * 1024 / mb), int(ddr_gib * 1024 / mb)
    pv, pd, ps = shares(nv, nd)
    t_cpu = mb / gbps_cpu                      # MB / (GB/s) = ms
    t_miss = mb / ssd_gbps
    return NL * t_fixed + TOPK * NL * (pd * t_cpu + ps * (t_cpu + t_miss)), (nv, nd, pv, pd, ps)
# --- fit t_fixed and ssd_gbps to the measured AJ runs (avg expert 6.59 MB, CPU 11.8 GB/s), VRAM 28.9 GiB
AJ_MB, VRAM = 6.59, 28.9 * 1.0
meas = [(24, 161.0), (32, 127.5), (48, 100.0)]        # DDR GiB, ms/token  (24/48: earlier summary numbers, 32: this session)
best = None
for tf in [x * 0.05 for x in range(0, 120)]:
    for sg in [x * 0.1 for x in range(3, 60)]:
        err = sum((tok_ms(AJ_MB, 11.8, VRAM, d, tf, sg)[0] - ms) ** 2 for d, ms in meas)
        if best is None or err < best[0]: best = (err, tf, sg)
_, TF, SG = best
print(f"fit: t_fixed={TF:.2f} ms/layer, ssd={SG:.2f} GB/s; AJ runs measured vs modelled ms/token:")
for d, ms in meas: print(f"   DDR {d} GiB: measured {ms:.0f}  modelled {tok_ms(AJ_MB, 11.8, VRAM, d, TF, SG)[0]:.0f}")
def table(ddr, vram):
    print(f"\nDDR {ddr} GiB, VRAM {vram:.1f} GiB of experts, uniform quant per row (static placement by usage)\n")
    print("| type | bpw | MB/expert | in VRAM | in DDR | on SSD | route share VRAM/DDR/SSD | CPU GB/s | ms/token | tok/s | effective MoE/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, (bpw, g) in sorted(TYPES.items(), key=lambda kv: kv[1][0]):
        mb = PARAMS * bpw / 8 / 1e6
        ms, (nv, nd, pv, pd, ps) = tok_ms(mb, g, vram, ddr, TF, SG)
        ns = n_experts - min(n_experts, nv + nd)
        tps = 1000.0 / ms
        print(f"| {t} | {bpw:.2f} | {mb:.1f} | {min(nv, n_experts)} | {min(nd, n_experts - min(nv, n_experts))} | {ns} | {pv*100:.0f}/{pd*100:.0f}/{ps*100:.0f}% | {g:.1f} | {ms:.0f} | {tps:.1f} | {TOPK*NL*tps:,.0f} |")
for v in (28.9, 15.8):          # both GPUs (as measured) / RTX 3090 alone (15.8 GB expert pool after dense weights)
    for d in (32, 64): table(d, v)
