#!/usr/bin/env python3
"""Replay a router trace (engine.log with `[TRACE l=L] router picks: eN(...)` lines) against DDR-cache policies.

Universe: experts of the CPU layers (default 11-44); the rest live in VRAM. Prompts are split by the engine's
[REQUEST] markers; inside a prompt the first sweep is the last prefill token, later sweeps are decode tokens.
Initial resident set = coldest-excluded top-C by an INDEPENDENT usage file (the old survey), so results are out of
sample. Policies: static, static+LRU (cache, persists over the session), LRU from empty, pinned half + LRU.
"""
import argparse, collections, json, re, sys

PICK = re.compile(r'\[TRACE l=(\d+)\] router picks:(.*)$')
ID = re.compile(r'e(\d+)\(')


def parse(path, lo, hi):
    prompts = []; cur = []; sweep = {}; last_layer = -1
    def flush():
        nonlocal sweep
        if sweep: cur.append(sweep); sweep = {}
    for line in open(path, errors='replace'):
        m = PICK.search(line)
        if m:
            L = int(m.group(1)); ids = [int(x) for x in ID.findall(m.group(2))]
            if L <= last_layer: flush()
            sweep[L] = ids; last_layer = L
        elif '[REQUEST] ' in line and ' tokens in ' in line:
            flush(); last_layer = -1
            if cur: prompts.append(cur); cur = []
    flush()
    if cur: prompts.append(cur)
    # first sweep of each prompt = last prefill token; the rest = decode tokens
    out = []
    for p in prompts:
        pre, dec = p[0], p[1:]
        out.append({'prefill_last': {L: v for L, v in pre.items() if lo <= L <= hi},
                    'tokens': [{L: v for L, v in t.items() if lo <= L <= hi} for t in dec]})
    return out


def load_usage(path):
    c = collections.Counter()
    for l in open(path):
        a = l.split()
        if len(a) == 3: c[(int(a[0]), int(a[1]))] += int(a[2])
    return c


class LRU:
    def __init__(self, cap, init_keys=(), pinned=()):
        self.cap = cap; self.d = collections.OrderedDict(); self.pinned = set(pinned)
        for k in init_keys: self.d[k] = 1
    def access(self, k):
        if k in self.d:
            self.d.move_to_end(k); return True
        if len(self.d) >= self.cap:
            for old in self.d:                       # oldest non-pinned
                if old not in self.pinned: del self.d[old]; break
        self.d[k] = 1; return False


def simulate(prompts, policy, C, init_rank, session, pin_frac=0.5):
    """Returns per-prompt list of misses per decode token."""
    init = init_rank[:C]                              # hottest first
    pinned = set(init[:int(C * pin_frac)]) if policy == 'pinned+LRU' else set()
    def fresh():
        if policy == 'static': return None
        if policy == 'LRU-empty': return LRU(C)
        return LRU(C, init_keys=reversed(init), pinned=pinned)   # hottest = most recent
    static_set = set(init)
    cache = fresh(); res = []
    for p in prompts:
        if not session: cache = fresh()
        m = []
        for tok in p['tokens']:
            miss = 0
            for L, ids in tok.items():
                for e in ids:
                    k = (L, e)
                    if policy == 'static': miss += k not in static_set
                    else: miss += not cache.access(k)
            m.append(miss)
        res.append(m)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace'); ap.add_argument('--rank-usage', required=True, help='independent usage file (old survey)')
    ap.add_argument('--check-usage', help='usage file from the same run (counts prefill+decode tokens), to validate the trace parse')
    ap.add_argument('--check-prefill', help='prefill-only usage file of the same run; decode-only usage = --check-usage minus this')
    ap.add_argument('--layers', default='11-44'); ap.add_argument('--budgets', default='32,40,48,56')
    ap.add_argument('--expert-bytes', type=float, default=6.49e6); ap.add_argument('--json')
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split('-'))
    prompts = parse(a.trace, lo, hi)
    print(f'prompts parsed: {len(prompts)}; decode tokens per prompt: {[len(p["tokens"]) for p in prompts]}; layers {lo}-{hi}')
    if a.check_usage:
        want = load_usage(a.check_usage)
        if a.check_prefill:
            pf = load_usage(a.check_prefill)
            want = collections.Counter({k: want[k] - pf.get(k, 0) for k in want})
        got = collections.Counter()
        for p in prompts:
            for t in p['tokens']:
                for L, ids in t.items():
                    for e in ids: got[(L, e)] += 1
        # the usage file counts every decode token; compare on the CPU layers
        w = {k: v for k, v in want.items() if lo <= k[0] <= hi}
        diff = sum(abs(w.get(k, 0) - got.get(k, 0)) for k in set(w) | set(got)); tot = sum(w.values())
        print(f'trace vs usage_decode (same run): trace hits {sum(got.values())}, usage hits {tot}, sum|diff| {diff} ({100*diff/max(tot,1):.2f}% of usage)')
    rank_c = load_usage(a.rank_usage)
    universe = [(L, e) for L in range(lo, hi + 1) for e in range(288)]
    init_rank = sorted(universe, key=lambda k: (-rank_c[k], k))
    # working sets
    sess = set(); print('\nworking set (distinct experts used in decode):')
    for i, p in enumerate(prompts):
        ws = {(L, e) for t in p['tokens'] for L, ids in t.items() for e in ids}; sess |= ws
        print(f'  prompt {i}: {len(ws)} experts ({100*len(ws)/len(universe):.1f}% of {len(universe)}), session so far {len(sess)}')
    ntok = sum(len(p['tokens']) for p in prompts); out = {}
    print(f'\n{"DDR GiB":>7s} {"C":>6s} {"policy":>12s} {"mode":>8s} {"misses/tok":>11s} {"first20":>8s} {"after20":>8s} {"SSD MB/tok":>11s} {"ms/tok@2.2":>11s}')
    for B in [float(x) for x in a.budgets.split(',')]:
        C = int(B * 2 ** 30 / a.expert_bytes)
        for policy in ('static', 'LRU-empty', 'static+LRU', 'pinned+LRU'):
            for session in ((True,) if policy == 'static' else (False, True)):
                r = simulate(prompts, policy, C, init_rank, session)
                allm = [x for p in r for x in p]; f20 = [x for p in r for x in p[:20]]; a20 = [x for p in r for x in p[20:]]
                mean = lambda v: sum(v) / max(len(v), 1)
                mode = 'session' if session else 'per-prompt'
                print(f'{B:7.0f} {C:6d} {policy:>12s} {mode:>8s} {mean(allm):11.1f} {mean(f20):8.1f} {mean(a20):8.1f} {mean(allm)*a.expert_bytes/1e6:11.0f} {mean(allm)*2.2:11.1f}')
                out[f'{B}:{policy}:{mode}'] = dict(C=C, misses_per_token=mean(allm), first20=mean(f20), after20=mean(a20))
    if a.json: json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
