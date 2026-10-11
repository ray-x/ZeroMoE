#!/usr/bin/env python3
"""Offline: dynamic VRAM hot-set policies on real GLM decode routes (engine probe tap -> router scores -> top-8 per layer per token).
Static VRAM set = most used experts of the usage file minus a swappable region; the swappable region D (slots) is managed by the policy.
hit = a cold route (not in the pinned set) served from D; uploads/token = promotions issued (per-token budget U).
usage: glm_dynamic_vram_sim.py <dump_dir with L<l>.x.bin/.s.bin> <usage_file> [n_vram=2398] [swap_frac=0.25]"""
import sys, numpy as np, collections
d, usage = sys.argv[1], sys.argv[2]
N_VRAM = int(sys.argv[3]) if len(sys.argv) > 3 else 2398
SWAP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
E, TOPK, L0, L1 = 288, 8, 3, 44
cnt = {}
for line in open(usage):
    p = line.split()
    if len(p) == 3 and L0 <= int(p[0]) <= L1: cnt[(int(p[0]), int(p[1]))] = float(p[2])
ranked = sorted(cnt, key=lambda k: -cnt[k])
n_dyn = int(N_VRAM * SWAP)
pinned = set(ranked[:N_VRAM - n_dyn])
static_extra = set(ranked[N_VRAM - n_dyn:N_VRAM])
S = {l: np.fromfile(f'{d}/L{l}.s.bin', dtype=np.float32).reshape(-1, E) for l in range(L0, L1 + 1)}
T = S[L0].shape[0]
routes = []   # per token: list of (layer, expert)
for t in range(T):
    r = []
    for l in range(L0, L1 + 1):
        for e in np.argpartition(-S[l][t], TOPK)[:TOPK]: r.append((l, int(e)))
    routes.append(r)

def run(policy, thr=2, budget=4, half=30.0):
    dyn = collections.OrderedDict()           # LRU order of the swappable region
    if policy == 'static':
        dyn = collections.OrderedDict((k, 1) for k in static_extra)
    miss = collections.Counter(); score = {}; last = {}
    hits = cold = uploads = 0
    for t, rt in enumerate(routes):
        pend = []
        for k in rt:
            if k in pinned: continue
            cold += 1
            if k in dyn:
                hits += 1; dyn.move_to_end(k); continue
            if policy == 'static': continue
            if policy == 'misscount':
                miss[k] += 1
                if miss[k] >= thr: pend.append(k)
            else:   # 'score': decayed re-reference score tracked for every routed expert (the host LRU "heat"), promote when hot
                s = score.get(k, 0.0) * 0.5 ** ((t - last.get(k, t)) / half) + 1.0
                score[k] = s; last[k] = t
                if s >= thr: pend.append(k)
        n = 0
        for k in pend:
            if n >= budget: break
            if k in dyn: continue
            if len(dyn) >= n_dyn: dyn.popitem(last=False)
            dyn[k] = 1; uploads += 1; n += 1
            miss[k] = 0
    return hits, cold, uploads
print('tokens %d, cold-candidate routes %d/token, pinned %d, swappable slots %d' % (T, len(routes[0]), len(pinned), n_dyn))
print('%-30s %8s %9s %9s %11s %11s' % ('policy', 'hit rate', 'hits/tok', 'upl/tok', 'saved ms/tok (pinned)', 'saved ms/tok (bounce)'))
base = None
for name, kw in [('static', {}), ('misscount>=2 (engine today)', dict(policy='misscount', thr=2)), ('misscount>=3', dict(policy='misscount', thr=3)),
                 ('heat>=2 hl30 (host-LRU score)', dict(policy='score', thr=2)), ('heat>=3 hl30', dict(policy='score', thr=3)),
                 ('heat>=2 hl10', dict(policy='score', thr=2, half=10.0)), ('heat>=4 hl30', dict(policy='score', thr=4))]:
    kw = dict(kw); pol = kw.pop('policy', 'static')
    h, c, u = run(pol, **kw)
    if base is None: base = h
    saved = lambda up_ms: ((h - base) * 0.39 - u * up_ms) / T
    print('%-30s %7.1f%% %9.2f %9.2f %21.2f %21.2f' % (name, 100 * h / c, h / T, u / T, saved(0.014), saved(0.12)))
print('cold routes per token (not in the pinned set): %.0f; CPU expert compute 0.39 ms; pinned upload costs ~0.014 ms CPU, bounce ~0.12 ms' % (run('static')[1] / T))
