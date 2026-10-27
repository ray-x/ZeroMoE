#!/usr/bin/env python3
"""Eviction-score study for expert promotion under real evictions (slack=0), MTP-style steps.
Usage: expert_promotion_eviction_sim.py [TRACE_DIR]  (recorded traces: ~/Projects/ZeroMoE-results/promotion-study/traces)
"""
import sys, glob, math, collections, itertools
from expert_promotion_sim import load, slots_per_layer, L, summarize  # same directory

def run(trace, hot, K, evict='blend', F=64, half_life=20000, window=1, cap=6, burst=20, burst_tokens=40, T=2,
        slack=1.0, decay_every=4096):
    """Promotion with ranked top-N per step; step = `window` tokens (MTP commits ~3 tokens per step).
    slack = share of never-used placeholder slots kept (1.0 = real config, 0.0 = every slot holds a used expert)."""
    lam = math.log(2) / half_life
    res = [dict() for _ in range(L)]          # expert -> [last_used, hits, crf]
    fill = [int(max(0, K[l] - len(hot[l])) * slack) for l in range(L)]
    cap_slots = [len(hot[l]) + fill[l] for l in range(L)]
    for l in range(L):
        for e in hot[l]: res[l][e] = [0, 0, 0.0]
    cand = [collections.Counter() for _ in range(L)]
    clock = miss = tot = prom = 0; hits_since = 0
    def score(v, now):
        last, hits, crf = v
        if evict == 'lru': return last
        if evict == 'lfu': return hits * 1e9 + last          # frequency first, recency breaks ties
        if evict == 'lrfu': return crf * math.exp(-lam * (now - last))
        return last + F * hits                                # blend (ZeroMoE tiered cache)
    for i in range(0, len(trace), window):
        steps = trace[i:i + window]
        for tok in steps:
            for l in range(L):
                for e in tok[l]:
                    tot += 1; clock += 1
                    r = res[l].get(e)
                    if r is not None:
                        r[2] = r[2] * math.exp(-lam * (clock - r[0])) + 1.0
                        r[0] = clock; r[1] += 1; hits_since += 1
                    else:
                        miss += 1; cand[l][e] += 1
        stepno = i // window
        budget = cap * window + (burst if stepno * window < burst_tokens else 0)
        pool = sorted(((c, l, e) for l in range(L) for e, c in cand[l].items() if c >= T), reverse=True)
        for c, l, e in pool[:budget]:
            if fill[l] > 0 and len(res[l]) < cap_slots[l]:
                fill[l] -= 1                                   # evict a never-used placeholder
            else:
                v = min(res[l], key=lambda x: score(res[l][x], clock)); del res[l][v]
            res[l][e] = [clock, c, float(c)]; cand[l].pop(e); prom += 1
        if hits_since >= decay_every:
            hits_since = 0
            for ll in range(L):
                for v in res[ll].values(): v[1] >>= 1
    return miss, tot, prom

def build(order, d='tr'):
    # d: directory with <name>.ids and trace_<name>.engine.log (see docs/design/expert-promotion.md)
    single = {}
    for n in set(order):
        t, h = load(f'{d}/{n}.ids'); single[n] = (t, h, slots_per_layer(f'{d}/trace_{n}.engine.log'))
    trace = []; hot = [set() for _ in range(L)]
    for n in order:
        t, h, K = single[n]; trace += t
        for l in range(L): hot[l] |= h[l]
    return trace, hot, K

if __name__ == '__main__':
    import os
    TRACES = sys.argv[1] if len(sys.argv) > 1 else 'tr'
    order = ['story', 'math', 'review', 'story', 'btree', 'math', 'palin', 'leet', 'story', 'review']
    trace, hot, K = build(order, TRACES)
    print("session:", len(trace), "tokens, topic order:", order)
    st = 0
    for tok in trace:
        for l in range(L):
            for e in tok[l]: st += e not in hot[l]
    tot_ = len(trace) * L * 10
    print(f"static miss {100*st/tot_:.1f}%")
    def line(tag, **kw):
        m, t, p = run(trace, hot, K, **kw)
        print(f"  {tag:46s} miss {100*m/t:5.2f}%  CPU evals -{100*(1-m/st):3.0f}%  promos/token {p/len(trace):4.1f}", flush=True)
    for slack in (1.0, 0.0):
        print(f"== slack={slack} (share of never-used slots kept), MTP-style step = 3 tokens, cap 6/token")
        line("LRU", evict='lru', slack=slack, window=3)
        line("LFU", evict='lfu', slack=slack, window=3)
        for F in (16, 64, 256, 1024):
            line(f"blend F={F} (ZeroMoE)", evict='blend', F=F, slack=slack, window=3)
        for hl in (2000, 10000, 50000, 250000):
            line(f"LRFU half-life={hl} accesses", evict='lrfu', half_life=hl, slack=slack, window=3)
    print("== step size, slack=0, blend F=64")
    for w in (1, 3, 4, 8):
        line(f"step = {w} token(s), cap 6/token", evict='blend', slack=0.0, window=w)
