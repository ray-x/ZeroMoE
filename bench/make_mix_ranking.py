#!/usr/bin/env python3
"""Combine two usage files ("layer expert count" lines): base + weight * extra, where extra is scaled to the base's total count first.
  bench/make_mix_ranking.py <base> <extra> <out> [weight=1.0]
Used for docs/models/glm53-flash.md: the shipped AJ-derived ranking + the GSQ-RCO pilot counts."""
import collections, sys
def load(p):
    c = collections.Counter()
    for l in open(p):
        t = l.split()
        if len(t) == 3: c[(int(t[0]), int(t[1]))] += int(t[2])
    return c
base, extra = load(sys.argv[1]), load(sys.argv[2])
w = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
scale = sum(base.values()) / sum(extra.values())
mix = collections.Counter(base)
for k, v in extra.items(): mix[k] += int(round(v * scale * w))
with open(sys.argv[3], 'w') as f:
    for (l, e), v in sorted(mix.items()): f.write(f'{l} {e} {v}\n')
print(len(mix), 'entries, total', sum(mix.values()))
