#!/usr/bin/env python3
"""Offline pilot study: lookahead depth k (predict layer l+k routing from layer l's FFN input) and feature variants.
SSD tiers as qwen_ssd_miss.py (IQ3_S, 2.66 MB/expert, 28 GiB DDR buffer incl. 2 GiB swap, VRAM expert GiB given).
usage: qwen_pilot_study.py <dump_dir> <model.gguf shard1> <hot_counts.json> <usage_file> [vram_gib=9] [ks=1,2,3,4]"""
import sys, glob, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model, hot_json, usage_path = sys.argv[2], sys.argv[3], sys.argv[4]
VRAM_GIB = float(sys.argv[5]) if len(sys.argv) > 5 else 9.0
KS = [int(k) for k in (sys.argv[6] if len(sys.argv) > 6 else '1,2,3,4').split(',')]
D, E, TOPK, DECODE, LAYERS = 2560, 512, 10, 64, 48
dev = 'cuda'; GIB = 1 << 30; B = 2.66e6; DDR_GIB = 26.0
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W = {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
Hn = {int(k): v for k, v in json.load(open(hot_json)).items()}
split_layers = sorted(Hn)
cnt = {}
for line in open(usage_path):
    p = line.split()
    if len(p) == 3: cnt.setdefault(int(p[0]), {})[int(p[1])] = float(p[2])
ranked = sorted(((c, l, e) for l in split_layers for e, c in cnt[l].items()), reverse=True)
seen = {(l, e) for _, l, e in ranked}
ranked += [(0.0, l, e) for l in split_layers for e in range(E) if (l, e) not in seen]
n_v = int(VRAM_GIB * GIB // B); n_d = int(DDR_GIB * GIB // B) - (LAYERS - len(split_layers)) * E
vram = {l: torch.zeros(E, dtype=torch.bool, device=dev) for l in split_layers}
ddr = {l: torch.zeros(E, dtype=torch.bool, device=dev) for l in split_layers}
for i, (_, l, e) in enumerate(ranked):
    if i < n_v: vram[l][e] = True
    elif i < n_v + n_d: ddr[l][e] = True
ssd = {l: ~(vram[l] | ddr[l]) for l in split_layers}
print('VRAM %d DDR %d SSD %d experts' % (n_v, n_d, sum(int(ssd[l].sum()) for l in split_layers)))

X, S = {}, {}
for v, d in enumerate(dumps):
    for l in range(LAYERS):
        fx, fs = f'{d}/L{l}.x.bin', f'{d}/L{l}.s.bin'
        if os.path.exists(fx) and os.path.exists(fs):
            X[(v, l)] = torch.tensor(np.fromfile(fx, dtype=np.float32).reshape(-1, D)).to(dev)
            S[(v, l)] = torch.tensor(np.fromfile(fs, dtype=np.float32).reshape(-1, E)).to(dev)
print('rows per dump (layer 5):', [X[(v, 5)].shape[0] for v in range(len(dumps))])

def feats(name, v, l):
    x = X[(v, l)]
    if name == 'x': return x
    if name == 'x+s': return torch.cat([x, S[(v, l)] * 50.0], -1)          # prev-layer router probs of this token
    if name == 'x+xprev': return torch.cat([x, X[(v, l - 1)] if (v, l - 1) in X else torch.zeros_like(x)], -1)
    raise ValueError(name)

def fit(Xtr, Ytr, lam):
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Z = (Xtr - mu) / sd
    A = Z.T @ Z + lam * Z.shape[0] / Z.shape[1] * torch.eye(Z.shape[1], device=dev)
    R = torch.linalg.solve(A, Z.T @ (Ytr - Ytr.mean(0, keepdim=True)))
    return mu, sd, R, Ytr.mean(0)

VARIANTS = [('pilot', None, 0), ('ridge x l=100', 'x', 100.0), ('ridge x l=1e3', 'x', 1e3), ('ridge x l=1e4', 'x', 1e4),
            ('ridge x+s l=1e3', 'x+s', 1e3), ('ridge x+xprev l=1e3', 'x+xprev', 1e3)]
CFG = [(10, 3), (16, 3), (24, 5)]
res = {}
for k in KS:
    for l in split_layers:
        t = l + k
        if t not in W or t not in ssd or not all((v, l) in X and (v, t) in S for v in range(len(dumps))): continue
        ps = [torch.softmax(X[(v, l)] @ W[t].T, -1) for v in range(len(dumps))]
        for vname, fname, lam in VARIANTS:
            F = [feats(fname, v, l) for v in range(len(dumps))] if fname else None
            for v in range(len(dumps)):
                sl = slice(-DECODE, None)
                p, y = ps[v][sl], S[(v, t)][sl]
                if fname is None:
                    score = p
                else:
                    tr = [i for i in range(len(dumps)) if i != v]
                    mu, sd, R, c = fit(torch.cat([F[i] for i in tr]), torch.cat([S[(i, t)] - ps[i] for i in tr]), lam)
                    score = p + ((F[v][sl] - mu) / sd) @ R + c
                real = y.topk(TOPK, -1).indices
                real_ssd = ssd[t][real]; n_real = real_ssd.sum(-1)
                top = score.topk(24, -1).indices
                a = res.setdefault((k, vname), {'rows': 0, 'real': 0, 'top10': 0, 'miss_layers': 0,
                                                **{c: [0, 0, 0] for c in CFG}})
                a['rows'] += real.shape[0]; a['real'] += n_real.sum().item(); a['miss_layers'] += (n_real > 0).sum().item()
                a['top10'] += (top[:, :TOPK].unsqueeze(-1) == real.unsqueeze(-2)).any(-1).sum().item()
                cold_all = ssd[t][top]
                for (K, M) in CFG:
                    cand = cold_all[:, :K]; rank = cand.long().cumsum(-1); keep = cand & (rank <= M)
                    st = top[:, :K].masked_fill(~keep, -1)
                    hit = (st.unsqueeze(-1) == real.unsqueeze(-2)) & real_ssd.unsqueeze(-2) & keep.unsqueeze(-1)
                    cov = hit.any(-2); r = a[(K, M)]
                    r[0] += cov.sum().item(); r[1] += keep.sum().item()
                    r[2] += ((cov.sum(-1) == n_real) & (n_real > 0)).sum().item()
print('\nIQ3_S, VRAM %.0f GiB; SSD-cold recall / precision / miss-layers fully covered; fetch/row' % VRAM_GIB)
print('%-3s %-22s %8s | %-26s | %-26s | %-26s' % ('k', 'variant', 'top10', 'pool10 M=3', 'pool16 M=3', 'pool24 M=5'))
for (k, vname), a in res.items():
    cols = []
    for c in CFG:
        cov, st, full = a[c]
        cols.append('%4.1f%% %4.1f%% %4.1f%% %4.2f' % (100 * cov / max(a['real'], 1), 100 * cov / max(st, 1),
                                                       100 * full / max(a['miss_layers'], 1), st / a['rows']))
    print('%-3d %-22s %7.1f%% | %-26s | %-26s | %-26s' % (k, vname, 100 * a['top10'] / (a['rows'] * TOPK), *cols))
