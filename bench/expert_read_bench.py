#!/usr/bin/env python3
"""Read-cost microbenchmark for routed experts stored in a (sharded) GGUF.

Reads whole experts (gate+up+down planes) at random (layer, expert) positions and reports per-expert
latency and throughput for: buffered cold (posix_fadvise DONTNEED first), buffered warm (page-cache hit),
and O_DIRECT (aligned superset read, as the engine's staging does), at 1..N concurrent readers.
Pure stdlib. Read-only on the model. Drops only the benchmarked file's clean pages (fadvise), needs no root.
"""
import argparse, json, mmap, os, random, statistics as st, struct, subprocess, sys, threading, time

TYPE_BYTES = {  # ggml type -> (block elems, bytes per block)
    0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 42: (32, 10)}


class R:
    def __init__(s, f): s.f = f
    def u(s, fmt):
        n = struct.calcsize(fmt); return struct.unpack('<' + fmt, s.f.read(n))[0]
    def string(s):
        n = s.u('Q'); return s.f.read(n).decode('utf-8', 'replace')


def skip_val(r, t):
    fixed = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
    if t in fixed: return r.u(fixed[t])
    if t == 8: return r.string()
    if t == 9:
        et = r.u('I'); n = r.u('Q')
        return [skip_val(r, et) for _ in range(n)] if n < 100000 else [skip_val(r, et) for _ in range(n)] and None
    raise ValueError(f'kv type {t}')


def parse(path):
    with open(path, 'rb') as f:
        r = R(f)
        assert f.read(4) == b'GGUF'
        ver = r.u('I'); nt = r.u('Q'); nk = r.u('Q'); align = 32
        for _ in range(nk):
            k = r.string(); t = r.u('I'); v = skip_val(r, t)
            if k == 'general.alignment': align = v
        ts = []
        for _ in range(nt):
            name = r.string(); nd = r.u('I'); dims = [r.u('Q') for _ in range(nd)]; ty = r.u('I'); off = r.u('Q')
            ts.append((name, dims, ty, off))
        data = (f.tell() + align - 1) // align * align
    return ts, data, align


def experts(paths, layers):
    out = {}  # layer -> list of (path, abs_off, per_expert_bytes, n_expert) for gate/up/down
    for p in paths:
        ts, data, align = parse(p)
        for name, dims, ty, off in ts:
            if '_exps.weight' not in name or not name.startswith('blk.'): continue
            L = int(name.split('.')[1])
            if L not in layers or len(dims) != 3: continue
            be, bb = TYPE_BYTES[ty]
            per = (dims[0] // be) * bb * dims[1]
            out.setdefault(L, {})[name.split('.')[2]] = (p, data + off, per, dims[2], ty)
    return out, align


def fadv_dontneed(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)


def resident_pages(path):
    o = subprocess.run(['fincore', '-b', '-n', '--output', 'RES', path], capture_output=True, text=True).stdout.split()
    return int(o[0]) if o else -1


def plan(ex, layers, n, seed):
    rnd = random.Random(seed); L = sorted(layers); res = []
    for _ in range(n):
        l = rnd.choice(L); e = rnd.randrange(288)
        planes = []
        for k in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
            p, off, per, ne, ty = ex[l][k]
            planes.append((p, off + e * per, per))
        res.append(planes)
    return res


def run(mode, work, nthreads):
    lat = []; lock = threading.Lock(); bytes_ = [0]; fds = {}
    for planes in work:
        for p, _, _ in planes:
            if p not in fds:
                fds[p] = (os.open(p, os.O_RDONLY), os.open(p, os.O_RDONLY | os.O_DIRECT))
    idx = [0]; PG = 4096

    def worker():
        buf = mmap.mmap(-1, 16 << 20) if mode == 'odirect' else None
        while True:
            with lock:
                i = idx[0]; idx[0] += 1
            if i >= len(work): return
            t0 = time.perf_counter(); nb = 0
            for p, off, ln in work[i]:
                fd, dfd = fds[p]
                if mode == 'cold':
                    os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED)
                if mode in ('cold', 'warm'):
                    d = os.pread(fd, ln, off); nb += len(d)
                else:
                    a = off & ~(PG - 1); al = ((off + ln + PG - 1) & ~(PG - 1)) - a
                    n = os.preadv(dfd, [memoryview(buf)[:al]], a); nb += n
            dt = time.perf_counter() - t0
            with lock: lat.append(dt * 1e3); bytes_[0] += nb
    t0 = time.perf_counter()
    th = [threading.Thread(target=worker) for _ in range(nthreads)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.perf_counter() - t0
    for a, b in fds.values(): os.close(a); os.close(b)
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    return dict(mode=mode, threads=nthreads, experts=len(work), MB_per_expert=round(bytes_[0] / len(work) / 1e6, 2),
                GBps=round(bytes_[0] / wall / 1e9, 2), p50_ms=round(q(.5), 2), p95_ms=round(q(.95), 2),
                p99_ms=round(q(.99), 2), mean_ms=round(st.mean(lat), 2), wall_s=round(wall, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shards', nargs='+'); ap.add_argument('--layers', default='11-44'); ap.add_argument('--experts', type=int, default=120)
    ap.add_argument('--threads', default='1,2,4,8'); ap.add_argument('--seed', type=int, default=7); ap.add_argument('--json')
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split('-')); layers = set(range(lo, hi + 1))
    ex, align = experts(a.shards, layers)
    per = {k: v for k, v in list(ex[sorted(ex)[0]].items())}
    print(f'general.alignment={align}; layers found={len(ex)}; per-expert bytes (gate/up/down) in layer {sorted(ex)[0]}: '
          + '/'.join(str(per[k][2]) for k in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')))
    print('plane offsets mod 4096 (first 6 experts of layer %d): ' % sorted(ex)[0]
          + str([[(per[k][1] + e * per[k][2]) % 4096 for k in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')] for e in range(2)]))
    rows = []
    for nth in [int(x) for x in a.threads.split(',')]:
        work = plan(ex, layers, a.experts * (1 if nth == 1 else 2), a.seed)
        for p in a.shards: fadv_dontneed(p)
        rows.append(run('cold', work, nth))
        rows.append(run('warm', work, nth))          # same experts again: now cached by the cold pass
        for p in a.shards: fadv_dontneed(p)
        before = sum(resident_pages(p) for p in a.shards)
        rows.append(run('odirect', work, nth))
        after = sum(resident_pages(p) for p in a.shards)
        rows[-1]['odirect_cache_growth_MB'] = round((after - before) / 1e6, 1)  # fincore -b reports bytes
        for p in a.shards: fadv_dontneed(p)
        for r in rows[-3:]: print(json.dumps(r))
    if a.json: json.dump(rows, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
