#!/usr/bin/env python3
"""Per-round decode miss report for run_pinned_ab.sh logs. usage: pinab_report.py NAME..."""
import re, sys, datetime
def ts(s):
    h, m, x = s.split(':'); return int(h) * 3600 + int(m) * 60 + float(x)
for name in sys.argv[1:]:
    rounds = []; last_pos = None; cur = None; prog = []
    for line in open(f'pinab/{name}.engine.log', errors='replace'):
        m = re.match(r'\[(\d+:\d+:[\d.]+)\].*DECODE-TIME\] pos=(\d+) .*total=([\d.]+) ms', line)
        if m:
            t, pos, ms = ts(m.group(1)), int(m.group(2)), float(m.group(3))
            if last_pos is None or pos < last_pos: cur = {'t0': t, 'ms': [], 'pos0': pos}; rounds.append(cur)
            cur['ms'].append(ms); cur['t1'] = t; last_pos = pos
            continue
        m = re.match(r'\[(\d+:\d+:[\d.]+)\].*OFFLOAD\] progress: calls=(\d+) routes=(\d+) misses=(\d+) loads=(\d+) transient=(\d+) evictions=(\d+) .*cold_routes=(\d+)', line)
        if m: prog.append((ts(m.group(1)),) + tuple(int(x) for x in m.groups()[1:]))
    print(f'== {name}: {len(rounds)} decode rounds, {len(prog)} progress lines')
    print('round  tokens  ms/token(mean of last 60)   decode loads   loads/token   evictions   (progress deltas between round start/end)')
    def at(t):  # last progress sample at or before t
        best = None
        for p in prog:
            if p[0] <= t: best = p
        return best
    for i, r in enumerate(rounds):
        a, b = at(r['t0']), at(r['t1'])
        if a is None and b is not None: a = (0,0,0,0,0,0,0,0)
        n = len(r['ms']); tail = r['ms'][-60:]
        if a and b and b is not a:
            dl, de = b[4] - a[4], b[6] - a[6]
            print(f'{i:5d} {n:7d} {sum(tail)/len(tail):12.1f} {dl:20d} {dl/max(n,1):13.2f} {de:11d}')
        else:
            print(f'{i:5d} {n:7d} {sum(tail)/len(tail):12.1f}   (no progress sample inside the round)')
    txt = open(f'pinab/{name}.engine.log', errors='replace').read()
    pl = re.findall(r'prefetch_loads=(\d+) prefetch=([\d.]+) ms', txt)
    if pl: print('prefetch worker: loads=%s time=%.1f s' % (pl[-1][0], float(pl[-1][1]) / 1000))
    sg = re.findall(r'segmented: prefetch_inserts=(\d+) promotions=(\d+) prefetch_evicted_unused=(\d+)', txt)
    if sg: print('segmented cache: prefetch_inserts=%s promotions(used prefetches)=%s evicted_unused=%s' % sg[-1])
    mn = re.findall(r'loads=(\d+) transient=\d+ evictions=\d+ errors=\d+ load=([\d.]+) ms callback=([\d.]+) ms', txt)
    if mn: print('miss node: demand loads=%s load time=%.1f s callback total=%.1f s' % (mn[-1][0], float(mn[-1][1]) / 1000, float(mn[-1][2]) / 1000))
    if prog: print('cumulative at end: loads=%d transient=%d evictions=%d cold_routes=%d' % (prog[-1][4], prog[-1][5], prog[-1][6], prog[-1][7]))
