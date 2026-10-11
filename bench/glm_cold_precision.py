#!/usr/bin/env python3
"""Cold-only precision of the GLM next-layer pilot, by cold rank, with and without a ridge correction (offline, leave-one-prompt-out).
Mirrors the engine telemetry: candidates are the highest-scoring COLD experts of layer l+1 (hot ones masked), and the j-th candidate counts
as used when it is in the layer's real top-8. Hot set approximation: the globally most used N experts of the usage file (N = VRAM-hot count).
usage: glm_cold_precision.py <probe_dump_dir> <model first shard> <usage_file> [n_hot=2398]"""
import sys, glob, os
import numpy as np, torch, gguf
dumps = sorted(d for d in glob.glob(sys.argv[1] + '/d[0-9]*'))
model, usage_path = sys.argv[2], sys.argv[3]
N_HOT = int(sys.argv[4]) if len(sys.argv) > 4 else 2398
D, E, TOPK = 4096, 288, 8
layers = list(range(3, 44))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
shards = [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
W, NORM, BIAS = {}, {}, {}
for sh in shards:
    for t in gguf.GGUFReader(sh).tensors:
        for l in range(3, 45):
            if t.name == f'blk.{l}.ffn_gate_inp.weight': W[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E, D)
            elif t.name == f'blk.{l}.ffn_norm.weight': NORM[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(D)
            elif t.name == f'blk.{l}.exp_probs_b.bias': BIAS[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E)
cnt = {}
for line in open(usage_path):
    p = line.split()
    if len(p) == 3 and 3 <= int(p[0]) <= 44: cnt[(int(p[0]), int(p[1]))] = float(p[2])
hot_keys = sorted(cnt, key=lambda k: -cnt[k])[:N_HOT]
hot = {l: torch.zeros(E, dtype=torch.bool) for l in range(3, 45)}
for l, e in hot_keys: hot[l][e] = True
print('hot experts %d (%.1f per layer); layers with hot: %d' % (len(hot_keys), len(hot_keys) / 42, sum(int(h.any()) for h in hot.values())))

def load(d, l):
    x = np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)
    s = np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)
    return torch.tensor(x), torch.tensor(s)
def rms(x, eps=1e-5): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
def pilot(x, l1): return torch.sigmoid((rms(x) * NORM[l1]) @ W[l1].T) + BIAS[l1]

def ridge(X, Y, mu, lam):
    sd = X.std(0, keepdim=True).clamp_min(1e-3)
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * torch.eye(D, device=dev) * Z.shape[0] / D
    return torch.linalg.solve(A, Z.T @ (Y - Y.mean(0, keepdim=True))), sd

def cold_stats(score, real_s, mask_hot, acc):
    s = score.masked_fill(mask_hot.to(dev), -1e9)
    top = s.topk(8, -1).indices
    real = real_s.topk(TOPK, -1).indices
    hit = (top.unsqueeze(-1) == real.unsqueeze(-2)).any(-1)          # [rows, 8] j-th cold candidate used
    acc[0] += hit.sum(0).cpu().numpy(); acc[1] += hit.shape[0]
    # also: how many of the layer's real cold routes are covered by the first K candidates
    real_cold = ~mask_hot.to(dev)[real]
    for K in (1, 2, 3, 4, 8):
        cov = ((top[:, :K].unsqueeze(-1) == real.unsqueeze(-2)) & real_cold.unsqueeze(-2)).any(-2)
        acc[2][K] += cov.sum().item()
    acc[3] += real_cold.sum().item()

LAMS = (1e2, 1e3, 1e4)
res = {m: [np.zeros(8), 0, {K: 0 for K in (1, 2, 3, 4, 8)}, 0] for m in ('pilot', 'pilot+ridge')}
for l in layers:
    data = [load(d, l) for d in dumps]
    nxt = [load(d, l + 1)[1] for d in dumps]
    P = [pilot(x, l + 1) for x, _ in data]
    for t in range(len(dumps)):
        v = (t + 1) % len(dumps)
        tr = [i for i in range(len(dumps)) if i not in (t, v)]
        Xtr = torch.cat([rms(data[i][0]) for i in tr]).to(dev)
        Ttr = torch.cat([nxt[i] - P[i] for i in tr]).to(dev)
        mu = Xtr.mean(0, keepdim=True)
        Xv, Yv, Pv = rms(data[v][0]).to(dev), nxt[v].to(dev), P[v].to(dev)
        Xt, Yt, Pt = rms(data[t][0]).to(dev), nxt[t].to(dev), P[t].to(dev)
        best, bs = None, -1.0
        for lam in LAMS:
            R, sd = ridge(Xtr, Ttr, mu, lam)
            sv = Pv + ((Xv - mu) / sd) @ R + Ttr.mean(0, keepdim=True)
            a = [np.zeros(8), 0, {K: 0 for K in (1, 2, 3, 4, 8)}, 0]
            cold_stats(sv, Yv, hot[l + 1], a)
            sc = float(a[0][:3].sum()) / max(a[1], 1)
            if sc > bs: bs, best = sc, (R, sd)
        R, sd = best
        st = Pt + ((Xt - mu) / sd) @ R + Ttr.mean(0, keepdim=True)
        cold_stats(Pt, Yt, hot[l + 1], res['pilot'])
        cold_stats(st, Yt, hot[l + 1], res['pilot+ridge'])
print('\nprecision of the j-th COLD candidate (used = in the layer real top-8), engine telemetry live (decode): 92 80 65 51 40 29 22 18')
print('%-12s' % 'method' + ''.join('  r%-5d' % (j + 1) for j in range(8)) + '   cumulative@4')
for m, (hits, rows, cov, rc) in res.items():
    pr = 100.0 * hits / rows
    print('%-12s' % m + ''.join('  %5.1f%%' % x for x in pr) + '   %5.1f%%' % pr[:4].mean())
print('\nshare of the layer real cold routes covered by the first K cold candidates (recall):')
for m, (hits, rows, cov, rc) in res.items():
    print('%-12s' % m + ''.join('  K=%d %5.1f%%' % (K, 100.0 * cov[K] / rc) for K in (1, 2, 3, 4, 8)))
