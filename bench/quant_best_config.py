#!/usr/bin/env python3
"""Search the best (VRAM-tier type, DDR/SSD-tier type) pair for a machine with the capacity model in quant_capacity_model.py.
usage: quant_best_config.py <vram_gib_for_experts> <ddr_gib_pool>"""
import sys, runpy, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    M = runpy.run_path('bench/quant_capacity_model.py')
TYPES, PARAMS, NL, TOPK, NE = M['TYPES'], M['PARAMS'], M['NL'], M['TOPK'], 288
cum, TF, SG = M['cum'], M['TF'], M['SG']
n_experts = NL * NE
def run(tv, td, vram, ddr):
    mv, md = PARAMS * TYPES[tv][0] / 8 / 1e6, PARAMS * TYPES[td][0] / 8 / 1e6
    nv = min(n_experts, int(vram * 1024 / mv))
    nd = min(n_experts - nv, int(ddr * 1024 / md))
    pv, pd, ps = cum[nv], cum[nv + nd] - cum[nv], 1 - cum[nv + nd]
    t_cpu, t_miss = md / TYPES[td][1], md / SG
    ms = NL * TF + TOPK * NL * (pd * t_cpu + ps * (t_cpu + t_miss))
    return ms, nv, nd, pv, pd, ps, mv, md
vram, ddr = float(sys.argv[1]), float(sys.argv[2])
res = []
for tv in TYPES:
    for td in TYPES:
        # quality proxy: the DDR/SSD tier should not be lower precision than the VRAM tier is "hot"-prioritised; no accuracy model, so list all
        ms, nv, nd, pv, pd, ps, mv, md = run(tv, td, vram, ddr)
        res.append((ms, tv, td, nv, nd, pv, pd, ps, mv, md))
res.sort()
print(f"VRAM {vram} GiB of experts, DDR pool {ddr} GiB. fit: t_fixed={TF:.2f} ms/layer ssd={SG:.2f} GB/s\n")
print("| VRAM tier | DDR/SSD tier | experts V/D/SSD | route share V/D/SSD | MB/expert V,D | ms/token | tok/s | MoE/s |")
print("|---|---|---|---|---|---|---|---|")
def row(r):
    ms, tv, td, nv, nd, pv, pd, ps, mv, md = r
    print(f"| {tv} | {td} | {nv}/{nd}/{n_experts-nv-nd} | {pv*100:.0f}/{pd*100:.0f}/{ps*100:.0f}% | {mv:.1f},{md:.1f} | {ms:.0f} | {1000/ms:.1f} | {TOPK*NL*1000/ms:,.0f} |")
print("fastest overall:")
for r in res[:5]: row(r)
print("\nbest with >= 3.0 bits per weight in BOTH tiers (quality-oriented):")
for r in [r for r in res if TYPES[r[1]][0] >= 3.0 and TYPES[r[2]][0] >= 3.0][:5]: row(r)
print("\nbest with the hottest tier at least as precise as the DDR tier (VRAM bpw >= DDR bpw), any precision:")
for r in [r for r in res if TYPES[r[1]][0] >= TYPES[r[2]][0]][:5]: row(r)
print("\nbest with >= 2.5 bits per weight in both tiers:")
for r in [r for r in res if TYPES[r[1]][0] >= 2.5 and TYPES[r[2]][0] >= 2.5][:5]: row(r)
print("\nthe IQ2_XXS reference (same type both tiers):")
row((*run('iq2_xxs', 'iq2_xxs', vram, ddr)[:1], 'iq2_xxs', 'iq2_xxs', *run('iq2_xxs', 'iq2_xxs', vram, ddr)[1:]))
