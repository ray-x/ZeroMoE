#!/usr/bin/env python3
"""Replay recorded Qwen4Exp routing under expert-promotion policies (design study, see
docs/design/expert-promotion.md).
trace line: "<token> <layer> e:hot e:hot ..." (10 routed experts, hot = in the static VRAM set).
Per layer l the VRAM set has K_l slots (from the engine log). Policies may swap experts
inside a layer. Metric: share of routed experts computed on the CPU (misses) and promotions.
"""
import re, sys, glob, collections, itertools, json

L = 48

def slots_per_layer(log):
    k = [0] * L
    for m in re.finditer(r"Layer (\d+) static hot/cold split: hot=(\d+)", open(log, errors='replace').read()):
        k[int(m.group(1))] = int(m.group(2))
    return k

def load(ids_file):
    toks = collections.defaultdict(lambda: [None] * L)
    hot = [set() for _ in range(L)]
    for line in open(ids_file):
        f = line.split(); t, l = int(f[0]), int(f[1])
        r = []
        for x in f[2:]:
            e, h = x.split(':'); e = int(e); r.append(e)
            if h == '1': hot[l].add(e)
        toks[t][l] = r
    T = sorted(toks)
    return [toks[t] for t in T], hot

def run_static(trace, hot, K):
    miss = tot = 0
    for tok in trace:
        for l in range(L):
            for e in tok[l]:
                tot += 1; miss += e not in hot[l]
    return miss, tot, 0

def run_oracle(trace, hot, K):
    # best static set for this very trace, same slot count
    miss = tot = 0
    for l in range(L):
        cnt = collections.Counter(e for tok in trace for e in tok[l])
        keep = {e for e, _ in cnt.most_common(K[l])}
        for e, c in cnt.items():
            tot += c; miss += c * (e not in keep)
    return miss, tot, 0

def run_tiered(trace, hot, K, T=2, F=64, decay_every=4096, cap=10**9, rotate=False, per_layer=10**9):
    """ZeroMoE-style: score = last_used + F*hit_count; promote after T misses (victim = lowest score in layer)."""
    res = [dict() for _ in range(L)]      # expert -> [last_used, hits]
    fill = [K[l] - len(hot[l]) for l in range(L)]   # unobserved resident slots: never used, lowest score
    for l in range(L):
        for e in hot[l]: res[l][e] = [0, 0]
    cand = [collections.Counter() for _ in range(L)]
    clock = 0; hits_since = 0; miss = tot = prom = 0
    for ti, tok in enumerate(trace):
        left = cap
        used_l = [0] * L
        for l in (range(L) if not rotate else [(ti * 7 + j) % L for j in range(L)]):
            for e in tok[l]:
                tot += 1; clock += 1
                r = res[l].get(e)
                if r is not None:
                    r[0] = clock; r[1] += 1; hits_since += 1
                else:
                    miss += 1
                    cand[l][e] += 1
                    if cand[l][e] >= T and left > 0 and used_l[l] < per_layer:
                        left -= 1; used_l[l] += 1
                        if fill[l] > 0:
                            fill[l] -= 1                       # evict a never-used placeholder
                        else:
                            v = min(res[l], key=lambda x: res[l][x][0] + F * res[l][x][1])
                            del res[l][v]
                        res[l][e] = [clock, cand[l].pop(e)]
                        prom += 1
                if hits_since >= decay_every:
                    hits_since = 0
                    for ll in range(L):
                        for v in res[ll].values(): v[1] >>= 1
    return miss, tot, prom

def run_batch(trace, hot, K, period=13, swaps=96, minc=2.0, margin=1.5, decay=0.9):
    """Strata-style: decayed per-(layer,expert) usage; every `period` tokens swap the best candidates in."""
    res = [set(hot[l]) for l in range(L)]
    fill = [K[l] - len(hot[l]) for l in range(L)]
    use = [collections.defaultdict(float) for _ in range(L)]
    miss = tot = prom = 0
    for i, tok in enumerate(trace):
        for l in range(L):
            for e in tok[l]:
                tot += 1; use[l][e] += 1.0
                miss += e not in res[l]
        if (i + 1) % period == 0:
            sw = []
            for l in range(L):
                c = sorted(((u, e) for e, u in use[l].items() if e not in res[l] and u >= minc), reverse=True)
                if not c: continue
                v = sorted((use[l].get(e, 0.0), e) for e in res[l])   # unused residents have 0
                # placeholders (never used) behave as zero-usage victims
                vict = [(0.0, None)] * fill[l] + v
                vict.sort(key=lambda x: x[0])
                for (cu, ce), (vu, ve) in zip(c, vict):
                    if cu < vu + margin: break
                    sw.append((cu - vu, l, ce, ve))
            sw.sort(reverse=True)
            for _, l, ce, ve in sw[:swaps]:
                if ve is None: fill[l] -= 1
                else: res[l].discard(ve)
                res[l].add(ce); prom += 1
            for l in range(L):
                for e in list(use[l]):
                    use[l][e] *= decay
                    if use[l][e] < 0.05: del use[l][e]
    return miss, tot, prom

def evaluate(policy, traces, **kw):
    out = []
    for name, (trace, hot, K) in traces.items():
        out.append((name, len(trace), *policy(trace, hot, K, **kw)))
    return out

def summarize(rows):
    miss = sum(r[2] for r in rows); tot = sum(r[3] for r in rows); prom = sum(r[4] for r in rows); tk = sum(r[1] for r in rows)
    return miss / tot, prom / tk

if __name__ == '__main__':
    d = sys.argv[1]
    traces = {}
    for f in sorted(glob.glob(d + '/*.ids')):
        n = f.split('/')[-1][:-4]
        trace, hot = load(f)
        K = slots_per_layer(f'{d}/trace_{n}.engine.log')
        traces[n] = (trace, hot, K)
    print("traces:", {n: len(v[0]) for n, v in traces.items()}, " total slots:", sum(next(iter(traces.values()))[2]))
    base = summarize(evaluate(run_static, traces))
    orc = summarize(evaluate(run_oracle, traces))
    print(f"static (today):          miss {100*base[0]:.2f}%   promotions/token {base[1]:.1f}")
    print(f"oracle static (ceiling): miss {100*orc[0]:.2f}%   ({100*(1-orc[0]/base[0]):.0f}% fewer CPU evals)")
