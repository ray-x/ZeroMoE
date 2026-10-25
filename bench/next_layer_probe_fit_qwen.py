#!/usr/bin/env python3
"""Qwen4Exp (Qwen3.8-Flash-Next) version of next_layer_probe_fit.py. Dumps from ZEROMOE_QWEN4EXP_PROBE_DUMP: L<l>.x.bin = FFN input of layer l
(HC-mixed activation the router consumes, [N,D]), L<l>.s.bin = router softmax probs of layer l ([N,E]). Qwen has no selection bias in this GGUF,
so the real selection is top-k of the probs. Pilot = softmax(W_{l+1} x_l) (what the engine's early-ready predictor computes)."""
import sys, glob, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model = sys.argv[2]
D, E, TOPK = 2560, 512, 10
shards = [model]
if '-00001-of-' in model:
    head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0]); shards = [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
W = {}
for sh in shards:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D)
layers = sorted(l for l in W if l + 1 in W)
print('router weights for layers', len(W), flush=True)

def load(d, l):
    x = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D))
    s = torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E))
    return x, s

def pilot(x, l1): return torch.softmax(x @ W[l1].T, dim=-1)

def prec(score, real_s, ks):
    real = real_s.topk(TOPK, dim=-1).indices
    out = {}
    for k in ks:
        pred = score.topk(k, dim=-1).indices
        hit = (pred.unsqueeze(-1) == real.unsqueeze(-2)).any(-1).float().sum(-1)
        out[k] = (hit.sum().item(), pred.numel())
    return out

x0, s0 = load(dumps[0], layers[0])
print('router reimplementation vs dumped probs, layer %d: max|diff| = %.3e' % (layers[0], (pilot(x0, layers[0]) - s0).abs().max().item()), flush=True)
ks = (1, 2, 3, 4, 5, 10)
lambdas = (1e-1, 1.0, 10.0, 100.0)
agg = {m: {k: [0, 0] for k in ks} for m in ('pilot', 'ridge', 'pilot+ridge')}
per_layer = {}
for l in layers:
    data = [load(d, l) for d in dumps]
    nxt = [load(d, l + 1)[1] for d in dumps]
    for t in range(len(dumps)):
        v = (t + 1) % len(dumps)
        tr = [i for i in range(len(dumps)) if i not in (t, v)]
        Xtr = torch.cat([data[i][0] for i in tr]).to(dev); Ytr = torch.cat([nxt[i] for i in tr]).to(dev)
        Ptr = torch.cat([pilot(data[i][0], l + 1) for i in tr]).to(dev)
        Xv, Yv, Pv = data[v][0].to(dev), nxt[v].to(dev), pilot(data[v][0], l + 1).to(dev)
        Xte, Yte, Pte = data[t][0].to(dev), nxt[t].to(dev), pilot(data[t][0], l + 1).to(dev)
        mu = Xtr.mean(0, keepdim=True)
        def ridge(X, Y, lam):
            Xc = X - mu
            A = Xc.T @ Xc + lam * torch.eye(D, device=dev) * Xc.shape[0] / D
            return torch.linalg.solve(A, Xc.T @ (Y - Y.mean(0, keepdim=True)))
        best = {}
        for name, tgt, bv, bt in (('ridge', Ytr, 0, 0), ('pilot+ridge', Ytr - Ptr, Pv, Pte)):
            bs, bW = -1, None
            for lam in lambdas:
                Wt = ridge(Xtr, tgt, lam)
                sc = prec((Xv - mu) @ Wt + tgt.mean(0, keepdim=True) + bv, Yv, (3,))[3][0]
                if sc > bs: bs, bW = sc, Wt
            best[name] = (Xte - mu) @ bW + tgt.mean(0, keepdim=True) + bt
        for name, sc in (('pilot', Pte), ('ridge', best['ridge']), ('pilot+ridge', best['pilot+ridge'])):
            r = prec(sc, Yte, ks)
            for k in ks:
                agg[name][k][0] += r[k][0]; agg[name][k][1] += r[k][1]
                pl = per_layer.setdefault((name, l), {}).setdefault(k, [0, 0]); pl[0] += r[k][0]; pl[1] += r[k][1]
print('\nprecision@k vs real top-%d of layer l+1 (leave-one-prompt-out over %d prompts, %d layer pairs)' % (TOPK, len(dumps), len(layers)))
print('%-12s' % 'method' + ''.join('  k=%-5d' % k for k in ks))
for m in agg:
    print('%-12s' % m + ''.join('  %6.1f%%' % (100.0 * agg[m][k][0] / max(agg[m][k][1], 1)) for k in ks))
rows = []
for l in layers:
    r = [100.0 * per_layer[(m, l)][3][0] / max(per_layer[(m, l)][3][1], 1) for m in ('pilot', 'ridge', 'pilot+ridge')]
    rows.append((r[2], l, r))
print('\nworst 5 layers at k=3 (pilot / ridge / pilot+ridge):')
for _, l, r in sorted(rows)[:5]: print('  layer %2d: %5.1f / %5.1f / %5.1f' % (l, *r))
