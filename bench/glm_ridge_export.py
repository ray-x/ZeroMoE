#!/usr/bin/env python3
"""Fit and export the GLM pilot ridge correction (RDG1 file, see include/pilot_ridge.hpp).
For each layer pair l -> l+1: score(l+1) = pilot + (rms(x_l) - mu) R + c, fitted on the pilot residual (real l+1 selection score - pilot),
standardized features, lambda picked per layer on a held-out prompt by cold-only precision of the first 3 candidates (hot set = most used
experts of the usage file), then refit on every prompt. R is stored in raw feature space (R_std / sd).
usage: glm_ridge_export.py <probe_dump_dir> <model first shard> <usage_file> <out.bin> [n_hot=2398]"""
import sys, glob, struct
import numpy as np, torch, gguf
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model, usage_path, out = sys.argv[2], sys.argv[3], sys.argv[4]
N_HOT = int(sys.argv[5]) if len(sys.argv) > 5 else 2398
D, E, TOPK = 4096, 288, 8
layers = list(range(3, 44))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W, NORM, BIAS = {}, {}, {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for t in gguf.GGUFReader(sh).tensors:
        for l in range(3, 45):
            if t.name == f'blk.{l}.ffn_gate_inp.weight': W[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E, D)
            elif t.name == f'blk.{l}.ffn_norm.weight': NORM[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(D)
            elif t.name == f'blk.{l}.exp_probs_b.bias': BIAS[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E)
cnt = {}
for line in open(usage_path):
    p = line.split()
    if len(p) == 3 and 3 <= int(p[0]) <= 44: cnt[(int(p[0]), int(p[1]))] = float(p[2])
hot = {l: torch.zeros(E, dtype=torch.bool) for l in range(3, 45)}
for l, e in sorted(cnt, key=lambda k: -cnt[k])[:N_HOT]: hot[l][e] = True
SD_FLOOR = 1e-3
def load(d, l):
    x = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D))
    s = torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E))
    return x, s
def rms(x, eps=1e-5): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
def pilot(x, l1): return torch.sigmoid((rms(x) * NORM[l1]) @ W[l1].T) + BIAS[l1]
def fit(X, T, mu, lam):
    sd = X.std(0, keepdim=True).clamp_min(SD_FLOOR)
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * torch.eye(D, device=dev) * Z.shape[0] / D
    return torch.linalg.solve(A, Z.T @ (T - T.mean(0, keepdim=True))) / sd.T, T.mean(0)   # raw-space R [D,E], c [E]
def cold_first3(score, real_s, l1):
    s = score.masked_fill(hot[l1].to(dev), -1e9)
    top = s.topk(3, -1).indices
    real = real_s.topk(TOPK, -1).indices
    return (top.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).float().mean().item()
recs, base_tot, corr_tot = [], 0.0, 0.0
for l in layers:
    xs = [load(d, l)[0] for d in dumps]; ys = [load(d, l + 1)[1] for d in dumps]
    ps = [pilot(x, l + 1) for x in xs]
    va = len(dumps) - 1; tr = list(range(va))
    Xtr = torch.cat([rms(xs[i]) for i in tr]).to(dev); Ttr = torch.cat([ys[i] - ps[i] for i in tr]).to(dev)
    mu = Xtr.mean(0, keepdim=True)
    Xv, Yv, Pv = rms(xs[va]).to(dev), ys[va].to(dev), ps[va].to(dev)
    base = cold_first3(Pv, Yv, l + 1)
    best, bl = base, None
    for lam in (1e2, 1e3, 1e4):
        R, c = fit(Xtr, Ttr, mu, lam)
        sc = cold_first3(Pv + (Xv - mu) @ R + c, Yv, l + 1)
        if sc > best: best, bl = sc, lam
    Xa = torch.cat([rms(x) for x in xs]).to(dev); Ta = torch.cat([ys[i] - ps[i] for i in range(len(dumps))]).to(dev)
    mu_a = Xa.mean(0, keepdim=True)
    if bl is None:   # the correction does not help on the held-out prompt: ship a no-op
        R = torch.zeros(D, E, device=dev); c = torch.zeros(E, device=dev)
    else:
        R, c = fit(Xa, Ta, mu_a, bl)
    R = R.clamp(-60000, 60000)
    base_tot += base; corr_tot += best
    recs.append((l + 1, mu_a.squeeze(0).cpu().numpy().astype('<f4'), R.T.contiguous().cpu().numpy().astype('<f2'), c.cpu().numpy().astype('<f4')))
    print(f'layer {l+1:2d}: held-out cold first-3 precision {100*base:5.1f}% -> {100*best:5.1f}% (lambda {bl})', flush=True)
print('mean held-out cold first-3 precision: pilot %.1f%% corrected %.1f%%' % (100 * base_tot / len(layers), 100 * corr_tot / len(layers)))
with open(out, 'wb') as f:
    f.write(struct.pack('<4i', 0x31474452, D, E, len(recs)))
    for t, mu, R, c in recs:
        f.write(struct.pack('<i', t)); f.write(mu.tobytes()); f.write(R.tobytes()); f.write(c.tobytes())
import os
print('wrote', out, os.path.getsize(out) / 1e6, 'MB')
