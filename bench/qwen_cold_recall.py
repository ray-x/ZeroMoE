#!/usr/bin/env python3
"""Cold-route recall/precision of the engine's staging candidates (offline, from probe dumps).
Engine candidates for layer l+1 = predicted top-10 (by score) filtered to experts NOT in the VRAM hot set, in rank
order, truncated to M (STAGE_MAX). recall = real cold routes covered / real cold routes; precision = covered / staged.
usage: qwen_cold_recall.py <dump_dir> <model.gguf shard1> <hot_counts.json {layer:H}> <usage_file>"""
import sys, glob, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model, hot_json, usage_path = sys.argv[2], sys.argv[3], sys.argv[4]
D, E, TOPK, LAM, DECODE = 2560, 512, 10, 100.0, 64
dev = 'cuda'
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W = {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
Hn = {int(k): v for k, v in json.load(open(hot_json)).items()}
cnt = {}
for line in open(usage_path):
    p = line.split()
    if len(p) == 3: cnt.setdefault(int(p[0]), {})[int(p[1])] = float(p[2])
hot = {}
for l, H in Hn.items():
    order = sorted(cnt[l], key=lambda e: -cnt[l][e])[:H]
    m = torch.zeros(E, dtype=torch.bool, device=dev); m[order] = True; hot[l] = m
def load(d, l):
    return (torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev),
            torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev))
def ridge(X, Y, mu):
    Xc = X - mu
    A = Xc.T @ Xc + LAM * torch.eye(D, device=dev) * Xc.shape[0] / D
    return torch.linalg.solve(A, Xc.T @ (Y - Y.mean(0, keepdim=True)))
Ms = (1, 2, 3, 5, 10)
acc = {(m, M): [0, 0, 0] for m in ('pilot', 'pilot+ridge') for M in Ms}  # covered, staged, real_cold
rows_total = 0
for l in sorted(Hn):
    t = l + 1
    if t not in W or t not in hot: continue
    xs, ss = zip(*[load(d, l) for d in dumps]); nxt = [load(d, t)[1] for d in dumps]
    ps = [torch.softmax(x @ W[t].T, -1) for x in xs]
    for v in range(len(dumps)):
        tr = [i for i in range(len(dumps)) if i != v]
        Xtr = torch.cat([xs[i] for i in tr]); T = torch.cat([nxt[i] - ps[i] for i in tr]); mu = Xtr.mean(0, keepdim=True)
        R = ridge(Xtr, T, mu); c = T.mean(0)
        sl = slice(-DECODE, None)
        x, y, p = xs[v][sl], nxt[v][sl], ps[v][sl]
        real = y.topk(TOPK, -1).indices
        real_cold = ~hot[t][real]
        for name, score in (('pilot', p), ('pilot+ridge', p + (x - mu) @ R + c)):
            top = score.topk(TOPK, -1).indices
            cold_c = ~hot[t][top]
            for M in Ms:
                # first M cold entries of the predicted top-10, in rank order
                rank = cold_c.long().cumsum(-1)
                keep = cold_c & (rank <= M)
                staged = top.masked_fill(~keep, -1)
                match = (staged.unsqueeze(-1) == real.unsqueeze(-2)) & real_cold.unsqueeze(-2) & keep.unsqueeze(-1)
                a = acc[(name, M)]
                a[0] += match.any(-1).sum().item(); a[1] += keep.sum().item(); a[2] += real_cold.sum().item()
    rows_total += 1
print('layer pairs', rows_total, '| decode rows only (last %d per prompt), leave-one-prompt-out ridge' % DECODE)
print('%-12s %4s %8s %10s %10s' % ('method', 'M', 'recall', 'precision', 'staged/row'))
rows = len(dumps) * DECODE * rows_total
for (name, M), (cov, st, rc) in acc.items():
    print('%-12s %4d %7.1f%% %9.1f%% %10.2f' % (name, M, 100 * cov / rc, 100 * cov / max(st, 1), st / rows))
print('real cold routes per row-layer: %.2f' % (acc[('pilot', 1)][2] / rows))
