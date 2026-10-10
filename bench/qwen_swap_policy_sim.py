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

class Swap:
    """Byte-capped swap pool shared across layers. policy: lru | lru-cold | seg<q> | 2q."""
    def __init__(self, cap, policy):
        self.cap, self.policy = cap, policy
        self.prob, self.prot = collections.OrderedDict(), collections.OrderedDict()   # LRU -> MRU order
        self.pb = self.tb = 0.0
        self.ghost = collections.OrderedDict(); self.q = 0.25
        if policy.startswith('seg'): self.q = float(policy[3:]) if len(policy) > 3 else 0.25
        if policy == '2q': self.q = 0.25
    def has(self, k): return k in self.prob or k in self.prot
    def _evict_one(self):
        two = self.policy.startswith('seg') or self.policy == '2q'
        if two and self.prob and (self.pb > self.q * self.cap or not self.prot):
            v, _ = self.prob.popitem(last=False); self.pb -= sz[v[0]]
            if self.policy == '2q':
                self.ghost[v] = True
                while len(self.ghost) > 4000: self.ghost.popitem(last=False)
            return v
        d = self.prot if self.prot else self.prob
        v, _ = d.popitem(last=False)
        if d is self.prot: self.tb -= sz[v[0]]
        else: self.pb -= sz[v[0]]
        return v
    def touch(self, k):
        """a routed hit."""
        if k in self.prot: self.prot.move_to_end(k); return
        if self.policy.startswith('seg'):                     # promote probationary -> protected
            del self.prob[k]; self.pb -= sz[k[0]]; self.prot[k] = True; self.tb += sz[k[0]]
        # classic 2Q: A1in hits are not promoted; lru policies never use prob
    def insert(self, k, kind):
        l = k[0]; evicted = []
        while self.pb + self.tb + sz[l] > self.cap and (self.prob or self.prot): evicted.append(self._evict_one())
        if self.pb + self.tb + sz[l] > self.cap: return False, evicted
        p = self.policy
        if p == 'lru': self.prot[k] = True; self.tb += sz[l]
        elif p == 'lru-cold':
            self.prot[k] = True; self.tb += sz[l]
            if kind == 'prefetch': self.prot.move_to_end(k, last=False)
        elif p.startswith('seg'):
            if kind == 'prefetch': self.prob[k] = True; self.pb += sz[l]
            else: self.prot[k] = True; self.tb += sz[l]
        else:                                                   # 2q
            if kind == 'demand' and k in self.ghost:
                del self.ghost[k]; self.prot[k] = True; self.tb += sz[l]
            else: self.prob[k] = True; self.pb += sz[l]
        return True, evicted

def simulate(R, cand, ssd, swap_gib, policy, K, M, k):
    sw = Swap(swap_gib * GIB, policy)
    T = R.shape[0]; start = T - DEC
    demand = fetched = hits = used_pf = layers_miss = rows = 0
    pf = set()
    for t in range(T):
        for l in range(LAYERS):
            if cand is not None:
                tgt = l + k
                if tgt < LAYERS and tgt in cand:
                    taken = 0
                    for e in cand[tgt][t][:K]:
                        key = (tgt, int(e))
                        if not ssd[tgt, e] or sw.has(key): continue
                        ok, ev = sw.insert(key, 'prefetch')
                        for v in ev: pf.discard(v)
                        if ok:
                            pf.add(key)
                            if t >= start: fetched += 1
                        taken += 1
                        if taken >= M: break
            m = 0
            for e in R[t, l]:
                if not ssd[l, e]: continue
                key = (l, int(e))
                if sw.has(key):
                    sw.touch(key)
                    if t >= start:
                        hits += 1
                        if key in pf: used_pf += 1
                    pf.discard(key)
                    continue
                if t >= start: demand += 1; m += 1
                if sw.cap > 0:
                    ok, ev = sw.insert(key, 'demand')
                    for v in ev: pf.discard(v)
            if t >= start: layers_miss += m > 0; rows += 1
    return demand / rows, layers_miss / rows, hits / rows, fetched / rows, used_pf / rows

print('decode rows per stream: last %d | VRAM %.1f GiB | DDR total %.1f GiB (static + swap) | streams: %s' % (
    DEC, VRAM, DDR_TOTAL, ' '.join(os.path.basename(d) for d in dirs)))
print('%-52s %9s %10s %10s %9s' % ('config', 'demand/L', 'miss-layers', 'prefetch/L', 'pf used/L'))
def run(label, swap, policy, vname=None, K=0, M=0, k=1):
    ssd = place(DDR_TOTAL - swap); rs = []
    for X, R in streams:
        cand = candidates(X, vname, k) if vname else None
        rs.append(simulate(R, cand, ssd, swap, policy, K, M, k))
    r = np.array(rs)
    print('%-52s %9.3f %9.1f%% %10.3f %9.3f   per-stream: %s' % (
        label, r[:, 0].mean(), 100 * r[:, 1].mean(), r[:, 3].mean(), r[:, 4].mean(), ' '.join('%.3f' % x for x in r[:, 0])))
POL = ['lru', 'lru-cold', 'seg0.25', 'seg0.5', '2q']
new = [v for v in variants if v.endswith('_std')][0]; old = 'pilot'
for g in (1.0, 2.0):
    print('\n--- swap %.0f GiB ---' % g)
    for pol in POL: run('%-8s demand only' % pol, g, pol)
    for vname, K, M in ((new, 10, 3), (new, 16, 3), (new, 24, 5), (old, 16, 3)):
        for pol in POL: run('%-8s + %s k=1 pool%d M%d' % (pol, vname[:12], K, M), g, pol, vname, K, M, 1)
