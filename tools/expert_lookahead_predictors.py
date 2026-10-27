#!/usr/bin/env python3
"""Offline study: how well can the cold experts of token i be predicted from earlier tokens?

Input: a ZEROMOE_PROMO_TRACE file. Each line: "<token> <t_in_window> <layer> e:r e:r ..." where r = 1 if the
expert was resident (GPU) when the token ran, 0 if it was computed on the CPU (a cold evaluation).
Predictors propose, per layer, a set of experts; they are scored on the cold evaluations of the next token:
  recall    = cold evaluations whose expert was in the proposed set
  precision = proposed experts that the next token routed to (cold or not), a lower bound on usefulness
  size      = proposed experts per layer
Usage: expert_lookahead_predictors.py TRACE
"""
import sys, collections

def load(path):
    toks = collections.defaultdict(dict)   # token -> layer -> list of (expert, resident)
    win_pos = {}
    for line in open(path):
        f = line.split()
        t, w, l = int(f[0]), int(f[1]), int(f[2])
        toks[t][l] = [(int(x.split(':')[0]), int(x.split(':')[1])) for x in f[3:]]
        win_pos[t] = w
    return toks, win_pos

def main(path):
    toks, win_pos = load(path)
    order = sorted(toks)
    layers = sorted(toks[order[0]])
    print(f"{len(order)} tokens, {len(layers)} layers with a hot set")
    cold_total = sum(1 for t in order for l in layers for e, r in toks[t][l] if not r)
    routed_total = sum(len(toks[t][l]) for t in order for l in layers)
    print(f"cold evaluations: {cold_total} of {routed_total} routed ({100*cold_total/routed_total:.1f}%)\n")

    def evaluate(name, propose):
        covered = cold = proposed = used = 0
        for i in range(8, len(order)):
            t = order[i]
            for l in layers:
                actual = {e for e, r in toks[t][l]}
                cold_now = {e for e, r in toks[t][l] if not r}
                P = propose(order, i, l)
                proposed += len(P); used += len(P & actual)
                cold += len(cold_now); covered += len(P & cold_now)
        n = (len(order) - 8) * len(layers)
        print(f"{name:52s} recall {100*covered/max(cold,1):5.1f}%  precision {100*used/max(proposed,1):5.1f}%  size/layer {proposed/n:5.1f}")

    def cold_of(order, i, l, back):
        out = set()
        for j in range(max(0, i - back), i):
            out |= {e for e, r in toks[order[j]][l] if not r}
        return out
    def routed_of(order, i, l, back):
        out = set()
        for j in range(max(0, i - back), i):
            out |= {e for e, r in toks[order[j]][l]}
        return out
    evaluate("cold experts of the previous token", lambda o, i, l: cold_of(o, i, l, 1))
    evaluate("cold experts of the previous 4 tokens (a window)", lambda o, i, l: cold_of(o, i, l, 4))
    evaluate("cold experts of the previous 16 tokens", lambda o, i, l: cold_of(o, i, l, 16))
    evaluate("cold experts of the previous 64 tokens", lambda o, i, l: cold_of(o, i, l, 64))
    evaluate("all experts routed by the previous token", lambda o, i, l: routed_of(o, i, l, 1))
    def twice(order, i, l, back=32):
        c = collections.Counter()
        for j in range(max(0, i - back), i):
            for e, r in toks[order[j]][l]:
                if not r: c[e] += 1
        return {e for e, n in c.items() if n >= 2}
    evaluate("cold experts missed >=2 times in the last 32 tokens", lambda o, i, l: twice(o, i, l))
    # within a window: tokens after the first in the same window, predicted from the window's first token
    covered = cold = proposed = used = 0
    for i, t in enumerate(order):
        if win_pos[t] == 0:
            continue
        first = order[i - win_pos[t]] if i - win_pos[t] >= 0 else None
        if first is None or win_pos.get(first) != 0: continue
        for l in layers:
            actual = {e for e, r in toks[t][l]}; cold_now = {e for e, r in toks[t][l] if not r}
            P = {e for e, r in toks[first][l] if not r}
            proposed += len(P); used += len(P & actual); cold += len(cold_now); covered += len(P & cold_now)
    print(f"{'in-window: cold experts of the first window token':52s} recall {100*covered/max(cold,1):5.1f}%  precision {100*used/max(proposed,1):5.1f}%")

if __name__ == '__main__':
    main(sys.argv[1])
