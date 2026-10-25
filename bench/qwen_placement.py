#!/usr/bin/env python3
"""Offline placement study: which experts live in VRAM / DDR-static / SSD under byte budgets (real per-layer expert
sizes of the given GGUF), evaluated on the real routing of the probe prompts' decode rows.
usage: qwen_placement.py <probe_dump_dir> <model_for_sizes shard1> [vram_gib=9] [ddr_gib=26]"""
import sys, glob, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
from qwen_expert_repack import TYPES
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model = sys.argv[2]
VRAM = float(sys.argv[3]) if len(sys.argv) > 3 else 9.0
DDR = float(sys.argv[4]) if len(sys.argv) > 4 else 26.0
GIB = 1 << 30; E = 512; LAYERS = 48; DECODE = 64; TOPK = 10
PLAIN = [0, 1, 2, 22]
USAGE = '/home/ray/work/moestream-graph/tests/moe_usage_survey/'

shards = sorted(glob.glob(model.replace('00001-of', '0000*-of')))
size = {}   # layer -> bytes per expert (gate+up+down)
for sh in shards:
    for name, (dims, ty, off) in gguf_min.parse(sh).items():
        if name.startswith('blk.') and '_exps.weight' in name and any(p in name for p in ('gate', 'up', 'down')):
            l = int(name.split('.')[1]); bsz, bel = TYPES[ty]
            size[l] = size.get(l, 0) + dims[0] * dims[1] // bel * bsz
sz = np.array([size[l] for l in range(LAYERS)], dtype=np.float64)
print('expert bytes per layer: min %.2f MB max %.2f MB mean %.2f MB; total experts %.1f GiB' % (
    sz.min() / 1e6, sz.max() / 1e6, sz.mean() / 1e6, sz.sum() * E / GIB))

def read_usage(path):
    c = np.zeros((LAYERS, E))
    for line in open(path):
        p = line.split()
        if len(p) == 3 and int(p[0]) < LAYERS: c[int(p[0]), int(p[1])] = float(p[2])
    return c
tap = read_usage(USAGE + 'zeromoe_usage_survey_qwen4exp_usage_tap')
pre = read_usage(USAGE + 'zeromoe_usage_survey_prefill')
dec = read_usage(USAGE + 'zeromoe_usage_survey')

routes = []   # per prompt: [LAYERS] list of bool [rows,E] np arrays; counts [LAYERS,E]
counts = []
for d in dumps:
    per, cnt = [], np.zeros((LAYERS, E))
    for l in range(LAYERS):
        s = torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E))[-DECODE:]
        top = s.topk(TOPK, -1).indices
        m = torch.zeros(s.shape[0], E, dtype=torch.bool); m.scatter_(1, top, True)
        per.append(m.numpy()); cnt[l] = m.numpy().sum(0)
    routes.append(per); counts.append(cnt)
rows = sum(routes[v][0].shape[0] for v in range(len(dumps)))

def place(score, vram_gib, ddr_gib, forced_ddr=()):
    """score [LAYERS,E] higher = more valuable. returns tier [LAYERS,E]: 0 VRAM, 1 DDR, 2 SSD."""
    tier = np.full((LAYERS, E), 2, dtype=np.int8)
    ddr_left = ddr_gib * GIB
    for l in forced_ddr: tier[l] = 1; ddr_left -= sz[l] * E
    cand = [(score[l, e], l, e) for l in range(LAYERS) if l not in forced_ddr for e in range(E)]
    cand.sort(key=lambda t: -t[0])
    vram_left = vram_gib * GIB
    for _, l, e in cand:
        if vram_left >= sz[l]: tier[l, e] = 0; vram_left -= sz[l]
        elif ddr_left >= sz[l]: tier[l, e] = 1; ddr_left -= sz[l]
    return tier

def evaluate(tier_for):
    tot = np.zeros(3); miss_layers = 0; n = 0
    for v in range(len(dumps)):
        tier = tier_for(v)
        for l in range(LAYERS):
            r = routes[v][l]
            t = tier[l]
            for k in range(3): tot[k] += (r & (t == k)).sum()
            miss_layers += ((r & (t == 2)).sum(1) > 0).sum(); n += r.shape[0]
    return tot / n, miss_layers / n          # routed per row-layer by tier, share of layers with >=1 SSD miss

def row(name, tier_for):
    r, ml = evaluate(tier_for)
    print('%-52s VRAM %.2f  DDR %.2f  SSD %.3f /layer-token | layers with SSD miss %4.1f%% | SSD MB/token %.0f' % (
        name, r[0], r[1], r[2], 100 * ml, r[2] * LAYERS * sz.mean() / 1e6))

print('\nbudget: VRAM %.1f GiB experts + DDR %.1f GiB static pool; routed experts per row-layer (10 total)' % (VRAM, DDR))
one = lambda t: (lambda v: t)
base = place(tap, VRAM, DDR, PLAIN)
row('1 usage tap, global rank (current), plain CPU layers pinned in DDR', one(base))
row('2 usage tap, global rank, plain layers join the pool', one(place(tap, VRAM, DDR)))
# per-layer uniform quota with the tap ranking
def uniform(score):
    tier = np.full((LAYERS, E), 2, dtype=np.int8)
    free = [l for l in range(LAYERS) if l not in PLAIN]
    for l in PLAIN: tier[l] = 1
    ddr_left = DDR * GIB - sum(sz[l] * E for l in PLAIN); vram_left = VRAM * GIB
    nv = int(vram_left / sum(sz[l] for l in free)); nd = int(ddr_left / sum(sz[l] for l in free))
    for l in free:
        o = np.argsort(-score[l]); tier[l, o[:nv]] = 0; tier[l, o[nv:nv + nd]] = 1
    return tier
row('3 usage tap, equal quota per layer', one(uniform(tap)))
row('4 usage tap / expert bytes (density)', one(place(tap / sz[:, None], VRAM, DDR, PLAIN)))
row('5 prefill usage file', one(place(pre, VRAM, DDR, PLAIN)))
row('6 decode survey usage file', one(place(dec, VRAM, DDR, PLAIN)))
row('7 tap + prefill + decode survey (normalized sum)', one(place(
    tap / tap.sum(1, keepdims=True) + pre / pre.sum(1, keepdims=True) + dec / dec.sum(1, keepdims=True), VRAM, DDR, PLAIN)))
tot = sum(counts)
row('8 probe-prompt decode counts, leave-one-prompt-out', lambda v: place(tot - counts[v], VRAM, DDR, PLAIN))
row('9 same as 8, plain layers join the pool', lambda v: place(tot - counts[v], VRAM, DDR))
row('10 oracle: this prompt\'s own counts (ceiling)', lambda v: place(counts[v], VRAM, DDR))
row('11 blend 0.5*tap + 0.5*LOO probe counts', lambda v: place(
    0.5 * tap / tap.sum(1, keepdims=True) + 0.5 * (tot - counts[v]) / (tot - counts[v]).sum(1, keepdims=True),
    VRAM, DDR, PLAIN))
print('\nVRAM budget sweep (strategy 1 vs 8):')
for vg in (6, 9, 12):
    for name, fn in (('1 tap', lambda v, vg=vg: place(tap, vg, DDR, PLAIN)),
                     ('8 LOO', lambda v, vg=vg: place(tot - counts[v], vg, DDR, PLAIN))):
        r, ml = evaluate(fn)
        print('  VRAM %2d GiB  %-6s SSD %.3f /layer-token  layers with miss %4.1f%%' % (vg, name, r[2], 100 * ml))
