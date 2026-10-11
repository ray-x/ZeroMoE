#!/usr/bin/env python3
"""Write L<l>.s.bin (real selection scores: sigmoid(router(rms(x) * norm)) + bias) for engine probe taps that only hold L<l>.x.bin.
usage: glm_probe_scores.py <dump_dir> <model first shard>"""
import sys, numpy as np, torch, gguf
d, model = sys.argv[1], sys.argv[2]
D, E = 4096, 288
head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W, NORM, BIAS = {}, {}, {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for t in gguf.GGUFReader(sh).tensors:
        for l in range(3, 45):
            if t.name == f'blk.{l}.ffn_gate_inp.weight': W[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E, D).cuda()
            elif t.name == f'blk.{l}.ffn_norm.weight': NORM[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(D).cuda()
            elif t.name == f'blk.{l}.exp_probs_b.bias': BIAS[l] = torch.tensor(np.array(t.data, dtype=np.float32)).reshape(E).cuda()
for l in range(3, 45):
    x = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).cuda()
    rn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)
    s = torch.sigmoid((rn * NORM[l]) @ W[l].T) + BIAS[l]
    s.cpu().numpy().astype('<f4').tofile(f'{d}/L{l}.s.bin')
print('wrote scores for layers 3..44,', x.shape[0], 'rows')
