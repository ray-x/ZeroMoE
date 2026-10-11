#!/usr/bin/env python3
"""Does the GLM pilot ridge transfer to new prompts? Fit on survey dumps (+ some live rounds), test on a held-out live round.
live dump = one session of 5 prompts x 95 decode tokens (rows 0-94, 95-189, ...).
usage: glm_ridge_transfer.py <survey_probe_dir> <live_dump_dir> <model> <usage>"""
import sys, glob, numpy as np, torch, gguf
sv, live, model, usage = sys.argv[1:5]
D, E, TOPK, N_HOT = 4096, 288, 8, 2398
dev = 'cuda'
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W, NORM, BIAS = {}, {}, {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for t in gguf.GGUFReader(sh).tensors:
        for l in range(3, 45):
            if t.name == f'blk.{l}.ffn_gate_inp.weight': W[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E, D).to(dev)
            elif t.name == f'blk.{l}.ffn_norm.weight': NORM[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(D).to(dev)
            elif t.name == f'blk.{l}.exp_probs_b.bias': BIAS[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E).to(dev)
cnt = {}
for line in open(usage):
    p = line.split()
    if len(p) == 3 and 3 <= int(p[0]) <= 44: cnt[(int(p[0]), int(p[1]))] = float(p[2])
hot = {l: torch.zeros(E, dtype=torch.bool, device=dev) for l in range(3, 45)}
for l, e in sorted(cnt, key=lambda k: -cnt[k])[:N_HOT]: hot[l][e] = True
def rms(x, eps=1e-5): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
def pilot(x, l1): return torch.sigmoid((rms(x) * NORM[l1]) @ W[l1].T) + BIAS[l1]
def fit(X, T, mu, lam):
    sd = X.std(0, keepdim=True).clamp_min(1e-3)
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * torch.eye(D, device=dev) * Z.shape[0] / D
    return torch.linalg.solve(A, Z.T @ (T - T.mean(0, keepdim=True))) / sd.T
sdirs = sorted(glob.glob(sv + '/d[0-9]*'))
res = {k: np.zeros(8) for k in ('pilot', 'survey-only', 'survey+4live')}
cnt_rows = 0
ROUND = 95
for l in range(3, 44):
    xs = [torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev) for d in sdirs]
    ys = [torch.tensor(np.fromfile(f'{d}/L{l+1}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev) for d in sdirs]
    Xs = torch.cat([rms(x) for x in xs]); Ts = torch.cat([y - pilot(x, l + 1) for x, y in zip(xs, ys)])
    xl = torch.tensor(np.fromfile(f'{live}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev)
    yl = torch.tensor(np.fromfile(f'{live}/L{l+1}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev)
    Pl = pilot(xl, l + 1); Rl = rms(xl)
    mu_s = Xs.mean(0, keepdim=True); R_s = fit(Xs, Ts, mu_s, 1e3); c_s = Ts.mean(0)
    for r in range(5):
        te = slice(r * ROUND, (r + 1) * ROUND)
        tr = torch.ones(xl.shape[0], dtype=torch.bool, device=dev); tr[te] = False
        Xa = torch.cat([Xs, Rl[tr]]); Ta = torch.cat([Ts, (yl - Pl)[tr]])
        mu_a = Xa.mean(0, keepdim=True); R_a = fit(Xa, Ta, mu_a, 1e3); c_a = Ta.mean(0)
        real = yl[te].topk(TOPK, -1).indices
        for name, s in (('pilot', Pl[te]), ('survey-only', Pl[te] + (Rl[te] - mu_s) @ R_s + c_s),
                        ('survey+4live', Pl[te] + (Rl[te] - mu_a) @ R_a + c_a)):
            top = s.masked_fill(hot[l + 1], -1e9).topk(8, -1).indices
            res[name] += (top.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).sum(0).cpu().numpy()
        if l == 3: cnt_rows += ROUND
print('held-out live round (leave-one-round-out), cold-only precision by rank, %d layers x %d rows' % (41, cnt_rows))
for k, v in res.items():
    pr = 100 * v / (cnt_rows * 41)
    print('%-13s' % k + ''.join('  r%d %5.1f%%' % (j + 1, pr[j]) for j in range(8)) + '   first-4 mean %.1f%%' % pr[:4].mean())
