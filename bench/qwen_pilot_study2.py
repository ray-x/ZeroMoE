#!/usr/bin/env python3
"""Pilot study v2: lookahead k and ridge, with BYTE-AWARE placement from the real IQ3_S per-layer expert sizes.
usage: qwen_pilot_study2.py <dump_dir> <router_model.gguf shard1 (Q2_0)> <size_model.gguf shard1 (IQ3_S)> <usage_tap> [vram_gib=9] [ddr_gib=26] [placement=tap|blend] [ks=1,2,3,4]"""
import sys, glob, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
from qwen_expert_repack import TYPES
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
router_model, size_model, usage_path = sys.argv[2], sys.argv[3], sys.argv[4]
VRAM = float(sys.argv[5]) if len(sys.argv) > 5 else 9.0
DDR = float(sys.argv[6]) if len(sys.argv) > 6 else 26.0
PLACEMENT = sys.argv[7] if len(sys.argv) > 7 else 'tap'
KS = [int(k) for k in (sys.argv[8] if len(sys.argv) > 8 else '1,2,3,4').split(',')]
D, E, TOPK, DECODE, LAYERS, LAM = 2560, 512, 10, 64, 48, 1e3
PLAIN = [0, 1, 2, 22]
dev = 'cuda'; GIB = 1 << 30

head, tail = router_model.split('-00001-of-'); n = int(tail.split('.')[0])
W = {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
sz = np.zeros(LAYERS)
for sh in sorted(glob.glob(size_model.replace('00001-of', '0000*-of'))):
    for name, (dims, ty, off) in gguf_min.parse(sh).items():
        if name.startswith('blk.') and '_exps.weight' in name and any(p in name for p in ('gate', 'up', 'down')):
            bsz, bel = TYPES[ty]; sz[int(name.split('.')[1])] += dims[0] * dims[1] // bel * bsz
tap = np.zeros((LAYERS, E))
for line in open(usage_path):
    p = line.split()
    if len(p) == 3 and int(p[0]) < LAYERS: tap[int(p[0]), int(p[1])] = float(p[2])

X, S = {}, {}
for v, d in enumerate(dumps):
    for l in range(LAYERS):
        X[(v, l)] = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev)
        S[(v, l)] = torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev)

def place(score, forced):
    tier = np.full((LAYERS, E), 2, dtype=np.int8); ddr_left = DDR * GIB; vram_left = VRAM * GIB
    for l in forced: tier[l] = 1; ddr_left -= sz[l] * E
    cand = sorted(((score[l, e], l, e) for l in range(LAYERS) if l not in forced for e in range(E)), key=lambda t: -t[0])
    for _, l, e in cand:
        if vram_left >= sz[l]: tier[l, e] = 0; vram_left -= sz[l]
        elif ddr_left >= sz[l]: tier[l, e] = 1; ddr_left -= sz[l]
    return tier
cnts = []
for v in range(len(dumps)):
    c = np.zeros((LAYERS, E))
    for l in range(LAYERS):
        top = S[(v, l)][-DECODE:].topk(TOPK, -1).indices.cpu().numpy()
        for r in top: c[l, r] += 1
    cnts.append(c)
tot = sum(cnts)
tiers = []
for v in range(len(dumps)):
    if PLACEMENT == 'tap': tiers.append(place(tap, PLAIN))
    else:
        loo = tot - cnts[v]
        tiers.append(place(0.5 * tap / tap.sum(1, keepdims=True) + 0.5 * loo / loo.sum(1, keepdims=True), PLAIN))
ssd = [{l: torch.tensor(t[l] == 2, device=dev) for l in range(LAYERS)} for t in tiers]
print('placement=%s VRAM %.1f GiB DDR %.1f GiB: SSD experts %d of %d (%.1f%% of experts)' % (
    PLACEMENT, VRAM, DDR, sum(int((t == 2).sum()) for t in tiers[:1]), LAYERS * E, 100 * (tiers[0] == 2).mean()))

def fit(Xtr, Ytr, lam):
    mu = Xtr.mean(0, keepdim=True) if False else Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Z = (Xtr - mu) / sd
    A = Z.T @ Z + lam * Z.shape[0] / Z.shape[1] * torch.eye(Z.shape[1], device=dev)
    return mu, sd, torch.linalg.solve(A, Z.T @ (Ytr - Ytr.mean(0, keepdim=True))), Ytr.mean(0)

CFG = [(10, 1), (10, 2), (10, 3), (16, 2), (16, 3), (24, 3), (24, 5)]
res = {}
for k in KS:
    for l in range(LAYERS):
        t = l + k
        if t >= LAYERS or t in PLAIN: continue
        ps = [torch.softmax(X[(v, l)] @ W[t].T, -1) for v in range(len(dumps))]
        for vname in ('pilot', 'ridge'):
            for v in range(len(dumps)):
                sl = slice(-DECODE, None); p, y = ps[v][sl], S[(v, t)][sl]
                if vname == 'pilot': score = p
                else:
                    tr = [i for i in range(len(dumps)) if i != v]
                    mu, sd, R, c = fit(torch.cat([X[(i, l)] for i in tr]), torch.cat([S[(i, t)] - ps[i] for i in tr]), LAM)
                    score = p + ((X[(v, l)][sl] - mu) / sd) @ R + c
                real = y.topk(TOPK, -1).indices
                real_ssd = ssd[v][t][real]; n_real = real_ssd.sum(-1)
                top = score.topk(24, -1).indices
                a = res.setdefault((k, vname), {'rows': 0, 'real': 0, 'miss_layers': 0, **{c: [0, 0, 0] for c in CFG}})
                a['rows'] += real.shape[0]; a['real'] += n_real.sum().item(); a['miss_layers'] += (n_real > 0).sum().item()
                cold_all = ssd[v][t][top]
                for (K, M) in CFG:
                    cand = cold_all[:, :K]; rank = cand.long().cumsum(-1); keep = cand & (rank <= M)
                    st = top[:, :K].masked_fill(~keep, -1)
                    hit = (st.unsqueeze(-1) == real.unsqueeze(-2)) & real_ssd.unsqueeze(-2) & keep.unsqueeze(-1)
                    cov = hit.any(-2); r = a[(K, M)]
                    r[0] += cov.sum().item(); r[1] += keep.sum().item()
                    r[2] += ((cov.sum(-1) == n_real) & (n_real > 0)).sum().item()
r0 = next(iter(res.values()))
print('SSD routed per layer-token %.3f ; layers with >=1 SSD miss %.1f%% (rows %d)' % (
    r0['real'] / r0['rows'], 100 * r0['miss_layers'] / r0['rows'], r0['rows']))
print('\nSSD-cold prefetch: recall / precision / miss-layers fully covered ; fetched per layer-token')
hdr = '%-3s %-6s ' % ('k', 'method') + ' '.join('%-27s' % ('pool %d M=%d' % c) for c in CFG)
print(hdr)
for (k, vname), a in res.items():
    cols = []
    for c in CFG:
        cov, st, full = a[c]
        cols.append('%4.0f%% %3.0f%% %3.0f%% %4.2f' % (100 * cov / max(a['real'], 1), 100 * cov / max(st, 1),
                                                       100 * full / max(a['miss_layers'], 1), st / a['rows']))
    print('%-3d %-6s ' % (k, vname) + ' '.join('%-27s' % c for c in cols))
