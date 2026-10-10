#!/usr/bin/env python3
"""Combined simulation on long decode streams: static placement + LRU swap area + pilot prefetch into the swap.
Ridge files are the engine's exact format (RDG1). Prefetch is assumed to arrive before the target layer (optimistic).
usage: qwen_prefetch_swap_sim.py <router_model Q2_0 shard1> <size_model IQ3_S shard1> <usage_tap> <decode_rows> <vram_gib> <ddr_total_gib> <ridge.bin|-> ... -- <dump_dir>..."""
import sys, os, glob, struct, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min
from qwen_expert_repack import TYPES
a = sys.argv[1:]; sep = a.index('--')
router_model, size_model, usage_path, DEC, VRAM, DDR_TOTAL = a[0], a[1], a[2], int(a[3]), float(a[4]), float(a[5])
ridge_files, dirs = a[6:sep], a[sep + 1:]
LAYERS, E, D, TOPK, GIB = 48, 512, 2560, 10, 1 << 30
PLAIN = [0, 1, 2, 22]; dev = 'cuda'
head, tail = router_model.split('-00001-of-'); n = int(tail.split('.')[0])
W = {}
for sh in [f'{head}-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]:
    for name, ent in gguf_min.parse(sh).items():
        if name.startswith('blk.') and name.endswith('.ffn_gate_inp.weight'):
            W[int(name.split('.')[1])] = torch.tensor(gguf_min.read_f32(sh, ent)).reshape(E, D).to(dev)
sz = np.zeros(LAYERS)
for sh in sorted(glob.glob(size_model.replace('00001-of', '0000*-of'))):
    for name, (dims, ty, off) in gguf_min.parse(sh).items():
        if name.startswith('blk.') and '_exps.weight' in name and any(p in name for p in ('gate', 'up', 'down')):
            bsz, bel = TYPES[ty]; sz[int(name.split('.')[1])] += dims[0] * dims[1] // bel * bsz
tap = np.zeros((LAYERS, E))
for line in open(usage_path):
    p = line.split()
    if len(p) == 3 and int(p[0]) < LAYERS: tap[int(p[0]), int(p[1])] = float(p[2])

def load_ridge(path):
    out = {}
    with open(path, 'rb') as f:
        magic, d, e, cnt = struct.unpack('<4i', f.read(16))
        for _ in range(cnt):
            (t,) = struct.unpack('<i', f.read(4))
            mu = torch.tensor(np.frombuffer(f.read(4 * D), dtype='<f4').copy()).to(dev)
            Rt = torch.tensor(np.frombuffer(f.read(2 * E * D), dtype='<f2').copy().reshape(E, D).astype(np.float32)).to(dev)
            c = torch.tensor(np.frombuffer(f.read(4 * E), dtype='<f4').copy()).to(dev)
            out[t] = (mu, Rt, c)
    return out
variants = {'pilot': None}
for p in ridge_files:
    if p != '-': variants[os.path.basename(p).replace('.bin', '')] = load_ridge(p)

def place(ddr_gib):
    ssd = np.ones((LAYERS, E), dtype=bool); ddr_left = ddr_gib * GIB; vram_left = VRAM * GIB
    for l in PLAIN: ssd[l] = False; ddr_left -= sz[l] * E
    cand = sorted(((tap[l, e], l, e) for l in range(LAYERS) if l not in PLAIN for e in range(E)), key=lambda t: -t[0])
    for _, l, e in cand:
        if vram_left >= sz[l]: ssd[l, e] = False; vram_left -= sz[l]
        elif ddr_left >= sz[l]: ssd[l, e] = False; ddr_left -= sz[l]
    return ssd

streams = []
for d in dirs:
    X = [torch.tensor(np.fromfile(f'{d}/L{l}.x.bin', dtype=np.float32).reshape(-1, D)).to(dev) for l in range(LAYERS)]
    S = [np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E) for l in range(LAYERS)]
    R = np.stack([np.argpartition(-s, TOPK, axis=1)[:, :TOPK] for s in S], 1)   # [tokens, layers, TOPK]
    streams.append((X, R))
CAND_K = 24
def candidates(X, vname, k):
    """cand[t] = [tokens, CAND_K] predicted expert ids for layer t from layer t-k's FFN input."""
    out = {}
    for t in range(LAYERS):
        l = t - k
        if l < 0 or t in PLAIN: continue
        x = X[l]; sc = torch.softmax(x @ W[t].T, -1)
        if variants[vname] is not None and t in variants[vname]:
            mu, Rt, c = variants[vname][t]; sc = sc + (x - mu) @ Rt.T + c
        out[t] = sc.topk(CAND_K, -1).indices.cpu().numpy()
    return out

def simulate(R, cand, ssd, swap_gib, K, M, k):
    cap = swap_gib * GIB; used = 0.0; lru = collections.OrderedDict()
    T = R.shape[0]; start = T - DEC
    demand = fetched = hits = used_pf = layers_miss = rows = 0
    pf_in_swap = set()
    def insert(key, l):
        nonlocal used
        while used + sz[l] > cap and lru:
            v, _ = lru.popitem(last=False); used -= sz[v[0]]; pf_in_swap.discard(v)
        if used + sz[l] <= cap: lru[key] = True; used += sz[l]; return True
        return False
    for t in range(T):
        for l in range(LAYERS):
            if cand is not None and t >= 0:
                tgt = l + k
                if tgt < LAYERS and tgt in cand:
                    taken = 0
                    for e in cand[tgt][t][:K]:
                        if not ssd[tgt, e] or (tgt, int(e)) in lru: continue
                        if insert((tgt, int(e)), tgt):
                            pf_in_swap.add((tgt, int(e)))
                            if t >= start: fetched += 1
                        taken += 1
                        if taken >= M: break
            m = 0
            for e in R[t, l]:
                if not ssd[l, e]: continue
                key = (l, int(e))
                if key in lru:
                    lru.move_to_end(key)
                    if t >= start:
                        hits += 1
                        if key in pf_in_swap: used_pf += 1
                    pf_in_swap.discard(key)
                    continue
                if t >= start: demand += 1; m += 1
                if cap > 0: insert(key, l)
            if t >= start: layers_miss += m > 0; rows += 1
    return demand / rows, layers_miss / rows, hits / rows, fetched / rows, used_pf / rows

print('decode rows per stream: last %d | VRAM %.1f GiB | DDR total %.1f GiB (static + swap) | streams: %s' % (
    DEC, VRAM, DDR_TOTAL, ' '.join(os.path.basename(d) for d in dirs)))
hdr = '%-44s %9s %10s %9s %10s %9s' % ('config', 'demand/L', 'miss-layers', 'swap hit', 'prefetch/L', 'pf used/L')
print(hdr)
def run(label, swap, vname=None, K=0, M=0, k=1):
    ssd = place(DDR_TOTAL - swap); rs = []
    for X, R in streams:
        cand = candidates(X, vname, k) if vname else None
        rs.append(simulate(R, cand, ssd, swap, K, M, k))
    r = np.array(rs)
    print('%-44s %9.3f %9.1f%% %9.3f %10.3f %9.3f   per-stream demand/L: %s' % (
        label, r[:, 0].mean(), 100 * r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), r[:, 4].mean(),
        ' '.join('%.3f' % x for x in r[:, 0])))
run('static only', 0.0)
for g in (1.0, 2.0, 4.0): run('LRU swap %.0f GiB, demand only' % g, g)
for vname in variants:
    for k in (1, 3):
        for K, M in ((10, 3), (16, 3)):
            run('swap 2 GiB + prefetch %s k=%d pool%d M%d' % (vname[:18], k, K, M), 2.0, vname, K, M, k)
