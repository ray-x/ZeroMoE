#!/usr/bin/env python3
"""Learned next-layer expert predictor ("probe"): predict layer l+1's router selection scores from the post-attention residual of
layer l, using dumps written by ZEROMOE_PROBE_DUMP (L<l>.x.bin = x_l rows f32 [N,D]; L<l>.s.bin = layer-l selection scores [N,E]).
Compares (a) the pilot (layer l+1's own norm+router applied to x_l), (b) a ridge probe z->s_{l+1}, (c) pilot + ridge correction.
Precision@k = mean over tokens of |predicted top-k intersect real top-8| / k, leave-one-prompt-out."""
import sys, os, glob, math
import numpy as np, torch, gguf
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model = sys.argv[2]
D, E, TOPK = 4096, 288, 8
layers = list(range(3, 44))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

shards = [model]
if '-00001-of-' in model:
    head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
    shards = [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
W, NORM, BIAS = {}, {}, {}
for sh in shards:
    for t in gguf.GGUFReader(sh).tensors:
        for l in range(3, 45):
            if t.name == f'blk.{l}.ffn_gate_inp.weight': W[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E, D)
            elif t.name == f'blk.{l}.ffn_norm.weight': NORM[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(D)
            elif t.name == f'blk.{l}.exp_probs_b.bias': BIAS[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E)
print('router weights loaded for layers', len(W), flush=True)

def load(d, l):
    x = np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)
    s = np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)
    return torch.tensor(x), torch.tensor(s)

def rms(x, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)

def pilot(x, l1):  # layer l1's real router applied to x
    z = rms(x) * NORM[l1]
    return torch.sigmoid(z @ W[l1].T) + BIAS[l1]

def prec(score, real_s, ks):
    real = real_s.topk(TOPK, dim=-1).indices
    out = {}
    for k in ks:
        pred = score.topk(k, dim=-1).indices
        hit = (pred.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).float().sum(-1)
        out[k] = (hit.sum().item(), pred.numel())
    return out

# validate the router reimplementation on layer 4: s_4 computed from x_4
x4, s4 = load(dumps[0], 4)
chk = pilot(x4, 4)
print('pilot reimplementation vs dumped scores on layer 4: max|diff| = %.3e' % (chk - s4).abs().max().item(), flush=True)

ks = (1, 2, 3, 4, 5, 8)
lambdas = (1e-1, 1.0, 10.0, 100.0)
agg = {m: {k: [0, 0] for k in ks} for m in ('pilot', 'ridge', 'pilot+ridge')}
per_layer = {}
for l in layers:
    data = [load(d, l) for d in dumps]            # x_l, s_l per prompt
    nxt = [load(d, l + 1)[1] for d in dumps]       # s_{l+1} per prompt
    for t in range(len(dumps)):
        v = (t + 1) % len(dumps)
        tr = [i for i in range(len(dumps)) if i not in (t, v)]
        Xtr = torch.cat([rms(data[i][0]) for i in tr]).to(dev); Ytr = torch.cat([nxt[i] for i in tr]).to(dev)
        Ptr = torch.cat([pilot(data[i][0], l + 1) for i in tr]).to(dev)
        Xv, Yv = rms(data[v][0]).to(dev), nxt[v].to(dev); Pv = pilot(data[v][0], l + 1).to(dev)
        Xte, Yte = rms(data[t][0]).to(dev), nxt[t].to(dev); Pte = pilot(data[t][0], l + 1).to(dev)
        mu = Xtr.mean(0, keepdim=True); ymu = Ytr.mean(0, keepdim=True); rmu = (Ytr - Ptr).mean(0, keepdim=True)
        def ridge(X, Y, lam):
            Xc = X - mu
            A = Xc.T @ Xc + lam * torch.eye(D, device=dev) * Xc.shape[0] / D
            return torch.linalg.solve(A, Xc.T @ (Y - Y.mean(0, keepdim=True)))
        best = {}
        for name, tgt_tr, base_v, base_te in (('ridge', Ytr, 0, 0), ('pilot+ridge', Ytr - Ptr, Pv, Pte)):
            bl, bs = None, -1
            for lam in lambdas:
                Wt = ridge(Xtr, tgt_tr, lam)
                pv = (Xv - mu) @ Wt + (tgt_tr.mean(0, keepdim=True)) + (base_v if name != 'ridge' else 0)
                sc = prec(pv, Yv, (3,))[3][0]
                if sc > bs: bs, bl, bW = sc, lam, Wt
            pte = (Xte - mu) @ bW + tgt_tr.mean(0, keepdim=True) + (base_te if name != 'ridge' else 0)
            best[name] = pte
        for name, sc in (('pilot', Pte), ('ridge', best['ridge']), ('pilot+ridge', best['pilot+ridge'])):
            r = prec(sc, Yte, ks)
            for k in ks:
                agg[name][k][0] += r[k][0]; agg[name][k][1] += r[k][1]
                pl = per_layer.setdefault((name, l), {}).setdefault(k, [0, 0]); pl[0] += r[k][0]; pl[1] += r[k][1]
print('\nprecision@k vs real top-8 of layer l+1 (leave-one-prompt-out over %d prompts, %d layer pairs, tokens=%d)' % (
    len(dumps), len(layers), agg['pilot'][3][1] // 3))
print('%-12s' % 'method' + ''.join('  k=%-5d' % k for k in ks))
for m in agg:
    print('%-12s' % m + ''.join('  %6.1f%%' % (100.0 * agg[m][k][0] / max(agg[m][k][1], 1)) for k in ks))
print('\nper-layer precision@3 (pilot / ridge / pilot+ridge); worst 5 layers by pilot+ridge:')
rows = []
for l in layers:
    r = [100.0 * per_layer[(m, l)][3][0] / max(per_layer[(m, l)][3][1], 1) for m in ('pilot', 'ridge', 'pilot+ridge')]
    rows.append((r[2], l, r))
for _, l, r in sorted(rows)[:5]:
    print('  layer %2d: %5.1f / %5.1f / %5.1f' % (l, *r))
