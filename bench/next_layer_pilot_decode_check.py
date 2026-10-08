#!/usr/bin/env python3
"""Pilot (layer l+1's router applied to the post-attention residual of layer l) precision@k, split into prefill rows and decode rows of one dump."""
import sys, numpy as np, torch, gguf
d, model, n_prefill = sys.argv[1], sys.argv[2], int(sys.argv[3])
D, E = 4096, 288
shards = [model]
if '-00001-of-' in model:
    head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0]); shards = [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]
W, NORM, BIAS = {}, {}, {}
for sh in shards:
    for t in gguf.GGUFReader(sh).tensors:
        for l in range(3, 45):
            if t.name == f'blk.{l}.ffn_gate_inp.weight': W[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E, D)
            elif t.name == f'blk.{l}.ffn_norm.weight': NORM[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(D)
            elif t.name == f'blk.{l}.exp_probs_b.bias': BIAS[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E)
rms = lambda x: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
res = {'prefill': {}, 'decode': {}}
for l in range(3, 44):
    x = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D))
    s1 = torch.tensor(np.fromfile(f'{d}/L{l+1}.s.bin', dtype=np.float32).reshape(-1, E))
    pil = torch.sigmoid((rms(x) * NORM[l + 1]) @ W[l + 1].T) + BIAS[l + 1]
    real = s1.topk(8, -1).indices
    for part, sl in (('prefill', slice(0, n_prefill)), ('decode', slice(n_prefill, None))):
        r, p = real[sl], pil[sl]
        for k in (1, 2, 3, 4, 5, 8):
            hit = (p.topk(k, -1).indices.unsqueeze(-1) == r.unsqueeze(-2)).any(-1).float().sum(-1)
            a = res[part].setdefault(k, [0.0, 0]); a[0] += hit.sum().item(); a[1] += hit.numel() * k
for part in res:
    n = res[part][1][1] // 1
    print('%-8s rows/layer=%d ' % (part, (len(x) - n_prefill) if part == 'decode' else n_prefill) + ' '.join('k=%d: %5.1f%%' % (k, 100 * res[part][k][0] / res[part][k][1]) for k in (1, 2, 3, 4, 5, 8)))
