#!/usr/bin/env python3
"""Offline: prefetch accuracy of the pilot (+ridge) for SSD-resident experts under a DDR+VRAM capacity budget.
Tiers (global usage ranking over the split layers): VRAM = top N_v, DDR static pool = next N_d, SSD = the rest.
Layers outside hot_counts (plain CPU layers) are always DDR and consume DDR budget.
usage: qwen_ssd_miss.py <dump_dir> <model.gguf shard1> <hot_counts.json> <usage_file>"""
import sys, glob, os, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
dumps = sorted(glob.glob(sys.argv[1] + '/d[0-9]*'))
model, hot_json, usage_path = sys.argv[2], sys.argv[3], sys.argv[4]
D, E, TOPK, LAM, DECODE, LAYERS = 2560, 512, 10, 100.0, 64, 48
dev = 'cuda'
GIB = 1 << 30
DDR_BUFFER_GIB, SWAP_GIB = 28.0, 2.0
# (label, bytes/expert, vram expert GiB)
CONFIGS = [('Q2_0  vram 9', 1.39e6, 9.0), ('IQ3_S vram 6', 2.66e6, 6.0), ('IQ3_S vram 9', 2.66e6, 9.0),
           ('IQ3_S vram 12', 2.66e6, 12.0)]
KS, MS = (10, 16, 24), (1, 2, 3, 5)
BUCKETS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.01)

head, tail = model.split('-00001-of-'); n = int(tail.split('.')[0])
W = {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
Hn = {int(k): v for k, v in json.load(open(hot_json)).items()}
split_layers = sorted(Hn)
cnt = {}
for line in open(usage_path):
    p = line.split()
    if len(p) == 3: cnt.setdefault(int(p[0]), {})[int(p[1])] = float(p[2])
ranked = sorted(((c, l, e) for l in split_layers for e, c in cnt[l].items()), reverse=True)
# experts never seen in the usage file are appended (count 0)
seen = {(l, e) for _, l, e in ranked}
ranked += [(0.0, l, e) for l in split_layers for e in range(E) if (l, e) not in seen]
plain_experts = (LAYERS - len(split_layers)) * E
masks = []
for label, B, vram_gib in CONFIGS:
    n_v = int(vram_gib * GIB // B)
    n_d = max(int((DDR_BUFFER_GIB - SWAP_GIB) * GIB // B) - plain_experts, 0)
    vram = {l: torch.zeros(E, dtype=torch.bool, device=dev) for l in split_layers}
    ddr = {l: torch.zeros(E, dtype=torch.bool, device=dev) for l in split_layers}
    for i, (_, l, e) in enumerate(ranked):
        if i < n_v: vram[l][e] = True
        elif i < n_v + n_d: ddr[l][e] = True
    tot = len(split_layers) * E
    print('%-14s expert %.2f MB | VRAM %5d (%4.1f%%) DDR-static %5d (%4.1f%%) SSD %5d (%4.1f%%)' % (
        label, B / 1e6, min(n_v, tot), 100 * min(n_v, tot) / tot, min(n_d, tot - min(n_v, tot)),
        100 * min(n_d, tot - min(n_v, tot)) / tot, max(tot - n_v - n_d, 0), 100 * max(tot - n_v - n_d, 0) / tot))
    masks.append((vram, ddr))

def load(d, l):
    return (torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev),
            torch.tensor(np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E)).to(dev))
def ridge(X, Y, mu):
    Xc = X - mu
    A = Xc.T @ Xc + LAM * torch.eye(D, device=dev) * Xc.shape[0] / D
    return torch.linalg.solve(A, Xc.T @ (Y - Y.mean(0, keepdim=True)))

class Acc:
    def __init__(s):
        s.rows = 0; s.real = {'vram': 0, 'ddr': 0, 'ssd': 0}; s.layers_miss = 0
        s.m = {(K, M): [0, 0, 0] for K in KS for M in MS}  # covered, staged, fully-covered layers
        s.pos = [[0, 0] for _ in range(6)]                 # K=24, M=5: precision by cold rank position
        s.bk = [[0, 0] for _ in BUCKETS[:-1]]              # K=24, M=10: precision by candidate score
accs = [Acc() for _ in CONFIGS]
for l in split_layers:
    t = l + 1
    if t not in W or t not in Hn: continue
    xs, _ = zip(*[load(d, l) for d in dumps]); nxt = [load(d, t)[1] for d in dumps]
    ps = [torch.softmax(x @ W[t].T, -1) for x in xs]
    for v in range(len(dumps)):
        tr = [i for i in range(len(dumps)) if i != v]
        Xtr = torch.cat([xs[i] for i in tr]); T = torch.cat([nxt[i] - ps[i] for i in tr]); mu = Xtr.mean(0, keepdim=True)
        R = ridge(Xtr, T, mu); c = T.mean(0)
        sl = slice(-DECODE, None)
        x, y, p = xs[v][sl], nxt[v][sl], ps[v][sl]
        real = y.topk(TOPK, -1).indices
        score = p + (x - mu) @ R + c
        sv, top = score.topk(max(KS), -1)
        for (vram, ddr), a in zip(masks, accs):
            ssd_t = ~(vram[t] | ddr[t])
            real_ssd = ssd_t[real]
            n_real = real_ssd.sum(-1)
            a.rows += real.shape[0]
            a.real['vram'] += vram[t][real].sum().item()
            a.real['ddr'] += (ddr[t][real]).sum().item()
            a.real['ssd'] += n_real.sum().item()
            a.layers_miss += (n_real > 0).sum().item()
            cold_all = ssd_t[top]
            for K in KS:
                cand = cold_all[:, :K]; rank = cand.long().cumsum(-1)
                for M in MS:
                    keep = cand & (rank <= M)
                    st = top[:, :K].masked_fill(~keep, -1)
                    hit = (st.unsqueeze(-1) == real.unsqueeze(-2)) & real_ssd.unsqueeze(-2) & keep.unsqueeze(-1)
                    cov = hit.any(-2)
                    r = a.m[(K, M)]
                    r[0] += cov.sum().item(); r[1] += keep.sum().item()
                    r[2] += ((cov.sum(-1) == n_real) & (n_real > 0)).sum().item()
                    if K == 24 and M == 5:
                        ph = hit.any(-1)
                        for j in range(1, 6):
                            sel = keep & (rank == j)
                            a.pos[j][0] += (ph & sel).sum().item(); a.pos[j][1] += sel.sum().item()
            cand = cold_all[:, :24]
            hit = ((top[:, :24].unsqueeze(-1) == real.unsqueeze(-2)) & real_ssd.unsqueeze(-2)).any(-1) & cand
            for b in range(len(BUCKETS) - 1):
                sel = cand & (sv >= BUCKETS[b]) & (sv < BUCKETS[b + 1])
                a.bk[b][0] += (hit & sel).sum().item(); a.bk[b][1] += sel.sum().item()

print('\nrows = %d decode row-layers (6 prompts, last %d tokens, leave-one-prompt-out ridge), %d split layers' % (
    accs[0].rows, DECODE, len(split_layers)))
for (label, _, _), a in zip(CONFIGS, accs):
    r = a.rows
    print('\n=== %s ===' % label)
    print('routed experts per row-layer: in VRAM %.2f | DDR (not VRAM) %.2f | SSD %.2f ; layers with >=1 SSD miss: %.0f%%' % (
        a.real['vram'] / r, a.real['ddr'] / r, a.real['ssd'] / r, 100 * a.layers_miss / r))
    if a.real['ssd'] == 0: print('  no SSD misses'); continue
    print('  %-5s %-3s %9s %10s %12s %20s' % ('pool', 'M', 'recall', 'precision', 'fetch/row', 'miss-layers fully covered'))
    for K in KS:
        for M in MS:
            cov, st, full = a.m[(K, M)]
            print('  %-5d %-3d %8.1f%% %9.1f%% %12.2f %19.1f%%' % (K, M, 100 * cov / a.real['ssd'], 100 * cov / max(st, 1),
                                                                    st / r, 100 * full / max(a.layers_miss, 1)))
    print('  precision by cold-rank position (pool 24, M=5): ' + ' '.join(
        '#%d %.0f%% (n=%d)' % (j, 100 * a.pos[j][0] / max(a.pos[j][1], 1), a.pos[j][1]) for j in range(1, 6)))
    print('  precision by candidate score (pool 24): ' + ' '.join(
        '[%.2f,%.2f) %.0f%% (n=%d)' % (BUCKETS[b], min(BUCKETS[b + 1], 1.0), 100 * a.bk[b][0] / max(a.bk[b][1], 1), a.bk[b][1])
        for b in range(len(BUCKETS) - 1)))
