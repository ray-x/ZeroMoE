#!/usr/bin/env python3
"""Offline N+1 expert prediction from the router trace: predict layer l+1's routed set from layer l's routed set
(learned co-occurrence), leave-one-prompt-out, precision@k against the real top-8 of l+1."""
import sys, collections
sys.path.insert(0, 'bench')
import expert_cache_sim as S
D = 'docs/ssd-offload/data/'
prompts = S.parse(D + 'glm53flash_survey5_router_trace.txt', 3, 44)
usage = S.load_usage(D + 'glm53flash_usage_old_survey.txt')
layers = list(range(3, 44))
NE = 288
def sweeps(p):
    return p['tokens']
# static per-layer popularity prior from the OLD survey (independent of the traces)
pop = {L: [0] * NE for L in range(3, 45)}
for (L, e), c in usage.items():
    if L in pop and 0 <= e < NE: pop[L][e] += c
hot_cut = 0.20   # treat the top 20% of experts per layer (by prior) as VRAM-hot, evaluate the cold-only prediction too
hot = {L: set(sorted(range(NE), key=lambda e: -pop[L][e])[:int(NE * hot_cut)]) for L in pop}
def evaluate(train, test, ks=(3, 4, 5, 8)):
    # co[l][e][e'] : times e' routed at l+1 given e routed at l
    co = {l: [collections.Counter() for _ in range(NE)] for l in layers}
    for p in train:
        for tok in sweeps(p):
            for l in layers:
                a, b = tok.get(l), tok.get(l + 1)
                if not a or not b: continue
                for e in a:
                    for e2 in b: co[l][e][e2] += 1
    res = {k: dict(all=[0, 0], cold=[0, 0], prior=[0, 0], prior_cold=[0, 0]) for k in ks}
    for p in test:
        for tok in sweeps(p):
            for l in layers:
                a, b = tok.get(l), tok.get(l + 1)
                if not a or not b: continue
                score = collections.Counter()
                for e in a:
                    for e2, c in co[l][e].items(): score[e2] += c
                ranked = [e for e, _ in score.most_common()]
                ranked_cold = [e for e in ranked if e not in hot[l + 1]]
                prior = sorted(range(NE), key=lambda e: -pop[l + 1][e])
                prior_cold = [e for e in prior if e not in hot[l + 1]]
                real = set(b); real_cold = {e for e in b if e not in hot[l + 1]}
                for k in ks:
                    for name, cand, tgt in (('all', ranked, real), ('cold', ranked_cold, real_cold),
                                            ('prior', prior, real), ('prior_cold', prior_cold, real_cold)):
                        pick = cand[:k]
                        res[k][name][0] += len(set(pick) & tgt); res[k][name][1] += len(pick)
    return res
tot = {}
for i in range(len(prompts)):
    train = [p for j, p in enumerate(prompts) if j != i]
    r = evaluate(train, [prompts[i]])
    for k, d in r.items():
        for n, (h, c) in d.items():
            t = tot.setdefault((k, n), [0, 0]); t[0] += h; t[1] += c
print("precision@k of predicting layer l+1's routed experts (leave-one-prompt-out, %d prompts)" % len(prompts))
print("%-4s %-22s %-22s %-22s %-22s" % ("k", "co-occurrence (all)", "co-occurrence (cold)", "static prior (all)", "static prior (cold)"))
for k in (3, 4, 5, 8):
    cells = []
    for n in ('all', 'cold', 'prior', 'prior_cold'):
        h, c = tot[(k, n)]; cells.append("%5.1f%% (%d/%d)" % (100.0 * h / max(c, 1), h, c))
    print("%-4d %-22s %-22s %-22s %-22s" % (k, *cells))
