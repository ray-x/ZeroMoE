#!/usr/bin/env python3
"""How many cold routes would a dynamic GPU-resident set remove?  Replays the survey router trace.
Initial resident set = top-N by an independent usage file (old survey); per-layer slots stay per-layer (a slot is only
reusable by another expert of the same layer). After each decode token, up to U of the experts that just missed (most
recently seen first) replace the least-recently-used resident of their layer. U=0 is the static baseline."""
import re, sys, collections
sys.path.insert(0, 'bench')
from expert_cache_sim import parse, load_usage
D = 'docs/ssd-offload/data/'
trace = D + 'glm53flash_survey5_router_trace.txt'
usage = load_usage(D + 'glm53flash_usage_old_survey.txt')
prompts = parse(trace, 3, 44)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4700
ranked = sorted(usage.items(), key=lambda kv: -kv[1])
base = [k for k, _ in ranked if 3 <= k[0] <= 44][:N]
def run(U, session=True):
    miss_tot = routes_tot = tokens = 0
    res = {}
    def reset():
        r = collections.defaultdict(dict)  # layer -> {expert: last_used}
        for L, e in base: r[L][e] = 0
        return r
    res = reset(); clock = 0
    for p in prompts:
        if not session: res = reset()
        for sweep in p['tokens']:  # decode tokens only
            clock += 1; tokens += 1; missed = []
            for L, ids in sweep.items():
                for e in ids:
                    routes_tot += 1
                    if e in res[L]: res[L][e] = clock
                    else: miss_tot += 1; missed.append((L, e))
            for L, e in missed[:U]:
                if e in res[L] or not res[L]: continue
                victim = min(res[L], key=res[L].get)
                del res[L][victim]; res[L][e] = clock
    return miss_tot / tokens, routes_tot / tokens
for U in (0, 2, 4, 8, 16, 32):
    for sess in (True, False):
        m, r = run(U, sess)
        print('U=%2d %-10s cold routes/token %.1f of %.0f (%.1f%%)' % (U, 'session' if sess else 'per-prompt', m, r, 100 * m / r))
