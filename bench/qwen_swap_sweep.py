#!/usr/bin/env python3
"""Adaptive DDR swap-area simulation on real routing streams (Qwen router probs per token per layer).
Static pool = usage-tap global ranking (VRAM then DDR, real IQ3_S per-layer expert bytes); swap = byte-capped pool shared
across layers, filled on SSD misses. Policies: lru, lfu(decay half-life H tokens). Evaluated on the decode rows.
usage: qwen_swap_sim.py <size_model.gguf shard1> <usage_tap> <decode_rows> <vram_gib> <ddr_total_gib> <dump_dir>..."""
import sys, os, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min, glob
from qwen_expert_repack import TYPES
size_model, usage_path, DEC, VRAM, DDR_TOTAL = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
dirs = sys.argv[6:]
LAYERS, E, TOPK, GIB = 48, 512, 10, 1 << 30
PLAIN = [0, 1, 2, 22]
sz = np.zeros(LAYERS)
for sh in sorted(glob.glob(size_model.replace('00001-of', '0000*-of'))):
    for name, (dims, ty, off) in gguf_min.parse(sh).items():
        if name.startswith('blk.') and '_exps.weight' in name and any(p in name for p in ('gate', 'up', 'down')):
            bsz, bel = TYPES[ty]; sz[int(name.split('.')[1])] += dims[0] * dims[1] // bel * bsz
tap = np.zeros((LAYERS, E))
for line in open(usage_path):
    p = line.split()
    if len(p) == 3 and int(p[0]) < LAYERS: tap[int(p[0]), int(p[1])] = float(p[2])

def place(ddr_gib):
    ssd = np.ones((LAYERS, E), dtype=bool); ddr_left = ddr_gib * GIB; vram_left = VRAM * GIB
    for l in PLAIN: ssd[l] = False; ddr_left -= sz[l] * E
    cand = sorted(((tap[l, e], l, e) for l in range(LAYERS) if l not in PLAIN for e in range(E)), key=lambda t: -t[0])
    for _, l, e in cand:
        if vram_left >= sz[l]: ssd[l, e] = False; vram_left -= sz[l]
        elif ddr_left >= sz[l]: ssd[l, e] = False; ddr_left -= sz[l]
    return ssd

def routes(d):
    R = []
    for l in range(LAYERS):
        s = np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)
        R.append(np.argpartition(-s, TOPK, axis=1)[:, :TOPK])
    return np.stack(R, 1)          # [tokens, layers, TOPK]

def simulate(R, ssd, swap_gib, policy, half=64.0):
    cap = swap_gib * GIB; used = 0.0
    lru = collections.OrderedDict(); score = {}; last = {}
    miss = layers_miss = hit_swap = 0; rows = 0; T = R.shape[0]
    start = T - DEC
    decay = 0.5 ** (1.0 / half)
    for t in range(T):
        for l in range(LAYERS):
            m = 0
            for e in R[t, l]:
                if not ssd[l, e]: continue
                k = (l, int(e))
                if k in lru:
                    if t >= start: hit_swap += 1
                    if policy == 'lru': lru.move_to_end(k)
                    else: score[k] = score[k] * decay ** (t - last[k]) + 1.0; last[k] = t
                    continue
                if t >= start: miss += 1; m += 1
                if cap <= 0: continue
                while used + sz[l] > cap and lru:
                    if policy == 'lru': v, _ = lru.popitem(last=False)
                    else:
                        v = min(lru, key=lambda q: score[q] * decay ** (t - last[q])); del lru[v]
                    used -= sz[v[0]]; score.pop(v, None); last.pop(v, None)
                if used + sz[l] <= cap:
                    lru[k] = True; used += sz[l]; score[k] = 1.0; last[k] = t
            if t >= start:
                layers_miss += m > 0; rows += 1
    return miss / rows, layers_miss / rows, hit_swap / rows

streams = [(d, routes(d)) for d in dirs]
print('decode rows evaluated per stream: last %d; VRAM %.1f GiB; DDR total %.1f GiB (static + swap)' % (DEC, VRAM, DDR_TOTAL))
print('%-34s %10s %10s %12s' % ('policy / swap', 'SSD/layer', 'miss-layers', 'swap hits/layer'))
for label, swap, pol, half in [('static only', 0.0, 'lru', 0)]+[('LRU swap %g GiB (%.0f%% of DDR)' % (g, 100*g/DDR_TOTAL), g, 'lru', 0) for g in (1,2,4,8,13)]:
    ssd = place(DDR_TOTAL - swap)
    r = np.array([simulate(R, ssd, swap, pol, half or 64.0) for _, R in streams])
    print('%-34s %10.3f %9.1f%% %12.3f   per-stream SSD/layer: %s' % (
        label, r[:, 0].mean(), 100 * r[:, 1].mean(), r[:, 2].mean(), ' '.join('%.3f' % x for x in r[:, 0])))
