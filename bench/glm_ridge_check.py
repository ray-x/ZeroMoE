#!/usr/bin/env python3
"""Apply an exported GLM ridge file (RDG1) exactly as the engine does, to chosen probe dumps, and print cold-only precision by rank.
usage: glm_ridge_check.py <ridge.bin> <model first shard> <usage_file> <dump_dir>... [n_hot=2398 via env NHOT]"""
import sys, os, struct, numpy as np, torch, gguf
rb, model, usage = sys.argv[1], sys.argv[2], sys.argv[3]; dirs = sys.argv[4:]
D, E, TOPK, N_HOT = 4096, 288, 8, int(os.environ.get('NHOT', 2398))
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
rec = {}
with open(rb, 'rb') as f:
    magic, d, e, cnt_ = struct.unpack('<4i', f.read(16))
    for _ in range(cnt_):
        (t,) = struct.unpack('<i', f.read(4))
        mu = torch.tensor(np.frombuffer(f.read(4 * D), '<f4').copy()).to(dev)
        Rt = torch.tensor(np.frombuffer(f.read(2 * E * D), '<f2').copy().reshape(E, D).astype(np.float32)).to(dev)
        c = torch.tensor(np.frombuffer(f.read(4 * E), '<f4').copy()).to(dev)
        rec[t] = (mu, Rt, c)
def rms(x, eps=1e-5): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
acc = {m: [np.zeros(8), 0] for m in ('pilot', 'ridge', 'ridge16')}
for d in dirs:
    for l in range(3, 44):
        x = torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev)
        y = torch.tensor(np.fromfile(f'{d}/L{l+1}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev)
        rn = rms(x)
        sc = torch.sigmoid((rn * NORM[l + 1]) @ W[l + 1].T) + BIAS[l + 1]
        mu, Rt, c = rec[l + 1]
        sr = sc + (rn - mu) @ Rt.T + c
        corr32 = (rn - mu) @ Rt.T + c
        corr16 = ((rn - mu).half() @ Rt.half().T).float() + c          # f16 activations, as a ggml F16 mat-mul would use
        sr16 = sc + corr16
        gaps = (sc.topk(2, -1).values[:, 0] - sc.topk(2, -1).values[:, 1]).mean().item()
        if l == 20 and not hasattr(sys, '_p'):
            sys._p = 1
            print('layer 21 stats: |corr32| mean %.4f max %.3f | |corr16-corr32| mean %.5f | top1-top2 gap %.3f | |Rt| max %.1f | rn-mu absmax %.2f'
                  % (corr32.abs().mean().item(), corr32.abs().max().item(), (corr16 - corr32).abs().mean().item(), gaps, Rt.abs().max().item(), (rn - mu).abs().max().item()))
        real = y.topk(TOPK, -1).indices
        for name, s in (('pilot', sc), ('ridge', sr), ('ridge16', sr16)):
            top = s.masked_fill(hot[l + 1], -1e9).topk(8, -1).indices
            hit = (top.unsqueeze(-1) == real.unsqueeze(-2)).any(-1)
            acc[name][0] += hit.sum(0).cpu().numpy(); acc[name][1] += hit.shape[0]
print('dumps:', ' '.join(os.path.basename(d) for d in dirs))
for m, (h, r) in acc.items(): print('%-6s' % m + ''.join('  r%d %5.1f%%' % (j + 1, 100 * h[j] / r) for j in range(8)))
