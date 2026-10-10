#!/usr/bin/env python3
"""Fit and export the pilot ridge correction used by the Qwen in-graph predictor.

usage: qwen_ridge_export.py <probe_dump_dir> <model.gguf first shard> <out.bin>

For each layer pair (l -> l+1): corrected scores = softmax(W_{l+1} x_l) + (x_l - mu) @ R + c, with R,c fitted by
ridge on the pilot residual (target = real next-layer probs - pilot probs). lambda is chosen on a held-out prompt,
then R,c are refit on every prompt. File: int32 magic 'RDG1', D, E, n; per record: int32 target layer, f32 mu[D],
f16 R^T [E][D], f32 c[E].
"""
import sys, glob, os, struct
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model, out = sys.argv[2], sys.argv[3]
D, E, TOPK = 2560, 512, 10
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
shards = [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
W = {}
for sh in shards:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
layers = sorted(l for l in W if l + 1 in W)
def load(d, l):
    x = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev)
    s = torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev)
    return x, s
def pilot(x, l1): return torch.softmax(x @ W[l1].T, dim=-1)
def hits(score, real_s, k):
    real = real_s.topk(TOPK, dim=-1).indices
    pred = score.topk(k, dim=-1).indices
    return (pred.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).float().sum().item() / pred.numel()
def ridge(X, Y, mu, lam):
    Xc = X - mu
    A = Xc.T @ Xc + lam * torch.eye(D, device=dev) * Xc.shape[0] / D
    return torch.linalg.solve(A, Xc.T @ (Y - Y.mean(0, keepdim=True)))
lambdas = (0.1, 1.0, 10.0, 100.0)
recs, tot = [], {'pilot': 0.0, 'corr': 0.0}
for l in layers:
    xs = [load(d, l)[0] for d in dumps]; ys = [load(d, l + 1)[1] for d in dumps]
    ps = [pilot(x, l + 1) for x in xs]
    tr = list(range(len(dumps) - 1)); va = len(dumps) - 1
    Xtr = torch.cat([xs[i] for i in tr]); Ttr = torch.cat([ys[i] - ps[i] for i in tr]); mu = Xtr.mean(0, keepdim=True)
    best = (-1, 1.0)
    for lam in lambdas:
        R = ridge(Xtr, Ttr, mu, lam)
        sc = hits(ps[va] + (xs[va] - mu) @ R + Ttr.mean(0, keepdim=True), ys[va], 3)
        if sc > best[0]: best = (sc, lam)
    base = hits(ps[va], ys[va], 3)
    tot['pilot'] += base
    Xall = torch.cat(xs); Tall = torch.cat([ys[i] - ps[i] for i in range(len(dumps))]); mu = Xall.mean(0, keepdim=True)
    R = ridge(Xall, Tall, mu, best[1]); c = Tall.mean(0)
    if best[0] <= base:  # correction does not help on the held-out prompt: make it a no-op
        R = torch.zeros_like(R); c = torch.zeros_like(c); best = (base, best[1])
    tot['corr'] += best[0]
    recs.append((l + 1, mu.squeeze(0).cpu().numpy().astype('<f4'), R.T.contiguous().cpu().numpy().astype('<f2'), c.cpu().numpy().astype('<f4')))
    print(f'layer {l+1:2d}: holdout top-3 pilot {100*base:5.1f}% -> corrected {100*best[0]:5.1f}% (lam {best[1]})', flush=True)
print('mean holdout top-3: pilot %.1f%% corrected %.1f%%' % (100 * tot['pilot'] / len(layers), 100 * tot['corr'] / len(layers)))
with open(out, 'wb') as f:
    f.write(struct.pack('<4i', 0x31474452, D, E, len(recs)))
    for t, mu, R, c in recs:
        f.write(struct.pack('<i', t)); f.write(mu.tobytes()); f.write(R.tobytes()); f.write(c.tobytes())
print('wrote', out, os.path.getsize(out) / 1e6, 'MB')
