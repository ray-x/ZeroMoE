#!/usr/bin/env python3
"""Does adding (a) last token's routed set and (b) the usage prior to the pilot+ridge score improve cold recall?
Same setup as qwen_cold_recall.py; grid over weights a (prev-token bonus) and b (log-usage prior)."""
import sys, glob, os, json, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*')); model, hot_json, usage_path = sys.argv[2:5]
D, E, TOPK, LAM, DECODE, dev = 2560, 512, 10, 100.0, 64, 'cuda'
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
hot, prior = {}, {}
for l, H in Hn.items():
    c = torch.zeros(E, device=dev)
    for e, v in cnt[l].items(): c[e] = v
    order = c.topk(H).indices
    m = torch.zeros(E, dtype=torch.bool, device=dev); m[order] = True; hot[l] = m
    prior[l] = torch.log1p(c) / torch.log1p(c).max()
def load(d, l):
    return (torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev),
            torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev))
def ridge(X, Y, mu):
    Xc = X - mu
    return torch.linalg.solve(Xc.T @ Xc + LAM * torch.eye(D, device=dev) * Xc.shape[0] / D, Xc.T @ (Y - Y.mean(0, keepdim=True)))
grid = list(itertools.product((0.0, 0.25, 0.5, 1.0), (0.0, 0.1, 0.25, 0.5)))
Ms = (1, 3, 5)
acc = {(g, M): [0, 0, 0] for g in grid for M in Ms}
for l in sorted(Hn):
    t = l + 1
    if t not in W or t not in hot: continue
    xs, ss = zip(*[load(d, l) for d in dumps]); nxt = [load(d, t)[1] for d in dumps]
    ps = [torch.softmax(x @ W[t].T, -1) for x in xs]
    for v in range(len(dumps)):
        tr = [i for i in range(len(dumps)) if i != v]
        Xtr = torch.cat([xs[i] for i in tr]); T = torch.cat([nxt[i] - ps[i] for i in tr]); mu = Xtr.mean(0, keepdim=True)
        R = ridge(Xtr, T, mu); c = T.mean(0)
        x, p = xs[v][-DECODE:], ps[v][-DECODE:]
        y = nxt[v][-DECODE:]
        yprev = nxt[v][-DECODE - 1:-1]
        prev = torch.zeros_like(y).scatter_(1, yprev.topk(TOPK, -1).indices, 1.0)
        real = y.topk(TOPK, -1).indices; real_cold = ~hot[t][real]
        base = p + (x - mu) @ R + c
        for (a, b) in grid:
            top = (torch.log(base.clamp_min(1e-5)) + a * prev + b * prior[t]).topk(TOPK, -1).indices
            cc = ~hot[t][top]
            for M in Ms:
                keep = cc & (cc.long().cumsum(-1) <= M)
                st = top.masked_fill(~keep, -1)
                match = (st.unsqueeze(-1) == real.unsqueeze(-2)) & real_cold.unsqueeze(-2) & keep.unsqueeze(-1)
                a_ = acc[((a, b), M)]; a_[0] += match.any(-1).sum().item(); a_[1] += keep.sum().item(); a_[2] += real_cold.sum().item()
print('%-14s' % '(prev, prior)' + ''.join('  M=%d recall/prec' % M for M in Ms))
for g in grid:
    print('%-14s' % str(g) + ''.join('   %5.1f%% / %5.1f%%' % (100 * acc[(g, M)][0] / acc[(g, M)][2], 100 * acc[(g, M)][0] / max(acc[(g, M)][1], 1)) for M in Ms))
