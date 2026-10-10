#!/usr/bin/env python3
"""Repack ONLY the routed-expert tensors of a Qwen GGUF into a 4 KiB-aligned sidecar (gate|up|down per expert, padded),
and benchmark O_DIRECT expert reads: original GGUF layout (3 reads/expert) vs the repacked layout (1 read/expert).
No per-layer token embedding / dense tensors are copied.
  repack: qwen_expert_repack.py repack <src.gguf> <out.bin> <layers e.g. 3-10 | 3,5,7>
  bench : qwen_expert_repack.py bench  <src.gguf> <out.bin> [samples=2000] [threads=1,4,8,16]"""
import sys, os, json, mmap, time, random, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import gguf_min

ALIGN = 4096
# ggml type -> (bytes per block, elements per block)
TYPES = {0: (4, 1), 1: (2, 1), 2: (18, 32), 8: (34, 32), 10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256),
         14: (210, 256), 16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32), 21: (110, 256),
         22: (82, 256), 23: (136, 256), 29: (56, 256), 42: (9, 32), 30: (2, 1)}

def up(n, a=ALIGN): return (n + a - 1) // a * a

def layer_plan(src, layers):
    t = gguf_min.parse(src); plan = {}
    for l in layers:
        parts = []
        for proj in ('gate', 'up', 'down'):
            dims, ty, off = t[f'blk.{l}.ffn_{proj}_exps.weight']
            E = dims[2]; elems = dims[0] * dims[1]; bsz, bel = TYPES[ty]
            assert elems % bel == 0
            parts.append({'off': off, 'bytes': elems // bel * bsz, 'type': ty})
        plan[l] = {'E': E, 'parts': parts, 'raw': sum(p['bytes'] for p in parts)}
        plan[l]['stride'] = up(plan[l]['raw'])
    return plan

def parse_layers(s):
    out = []
    for p in s.split(','):
        a, _, b = p.partition('-'); out += list(range(int(a), int(b or a) + 1))
    return out

def cmd_repack(src, out, layers):
    plan = layer_plan(src, layers); idx = {'src': src, 'align': ALIGN, 'layers': {}}
    pos = 0
    with open(src, 'rb') as sf, open(out, 'wb') as of:
        for l, pl in plan.items():
            idx['layers'][l] = {'base': pos, 'stride': pl['stride'], 'E': pl['E'], 'raw': pl['raw'],
                                'parts': pl['parts']}
            for e in range(pl['E']):
                buf = bytearray(pl['stride'])
                o = 0
                for p in pl['parts']:
                    n = p['bytes']
                    sf.seek(p['off'] + e * n); buf[o:o + n] = sf.read(n); o += n
                of.write(buf)
            pos += pl['E'] * pl['stride']
    json.dump(idx, open(out + '.json', 'w'))
    raw = sum(pl['E'] * pl['raw'] for pl in plan.values())
    print('wrote %s: %.2f GiB (raw expert bytes %.2f GiB, padding +%.2f%%), layers %s' % (
        out, pos / 2**30, raw / 2**30, 100 * (pos - raw) / raw, list(plan)))

def cmd_bench(src, out, samples, thread_list):
    idx = json.load(open(out + '.json')); L = {int(k): v for k, v in idx['layers'].items()}
    rnd = random.Random(1)
    picks = [(rnd.choice(list(L)), rnd.randrange(next(iter(L.values()))['E'])) for _ in range(samples)]
    fd_src = os.open(src, os.O_RDONLY | os.O_DIRECT); fd_rp = os.open(out, os.O_RDONLY | os.O_DIRECT)
    maxb = max(up(v['stride']) + 2 * ALIGN for v in L.values())
    # correctness: one expert from each layout, plain buffered reads
    l0, e0 = picks[0]; v = L[l0]
    with open(src, 'rb') as f, open(out, 'rb') as g:
        orig = b''
        for p in v['parts']:
            n = p['bytes']; f.seek(p['off'] + e0 * n); orig += f.read(n)
        g.seek(v['base'] + e0 * v['stride']); assert g.read(v['raw']) == orig, 'repack mismatch'
    print('layers %s, %d samples, expert raw bytes %s MB; repack verified' % (
        list(L), samples, sorted({round(x['raw'] / 1e6, 2) for x in L.values()})))

    def worker_orig(chunk):
        buf = mmap.mmap(-1, maxb * 3); lat = []
        for l, e in chunk:
            v = L[l]; t0 = time.perf_counter()
            for i, p in enumerate(v['parts']):
                n = p['bytes']; a = p['off'] + e * n
                lo = a // ALIGN * ALIGN; ln = up(a + n - lo)
                os.preadv(fd_src, [memoryview(buf)[i * maxb:i * maxb + ln]], lo)
            lat.append(time.perf_counter() - t0)
        return lat

    def worker_rp(chunk):
        buf = mmap.mmap(-1, maxb); lat = []
        for l, e in chunk:
            v = L[l]; t0 = time.perf_counter()
            os.preadv(fd_rp, [memoryview(buf)[:v['stride']]], v['base'] + e * v['stride'])
            lat.append(time.perf_counter() - t0)
        return lat

    useful = sum(L[l]['raw'] for l, _ in picks)
    print('%-8s %-9s %10s %10s %12s %12s' % ('threads', 'layout', 'experts/s', 'MB/s', 'mean ms/exp', 'p95 ms/exp'))
    for th in thread_list:
        for name, fn in (('orig', worker_orig), ('repack', worker_rp), ('orig', worker_orig), ('repack', worker_rp)):
            chunks = [picks[i::th] for i in range(th)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(th) as ex: lats = [x for r in ex.map(fn, chunks) for x in r]
            dt = time.perf_counter() - t0
            print('%-8d %-9s %10.0f %10.0f %12.2f %12.2f' % (th, name, samples / dt, useful / dt / 1e6,
                                                          1e3 * np.mean(lats), 1e3 * np.percentile(lats, 95)))

if __name__ == '__main__':
    if sys.argv[1] == 'repack': cmd_repack(sys.argv[2], sys.argv[3], parse_layers(sys.argv[4]))
    else: cmd_bench(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 2000,
                    [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else '1,4,8,16').split(',')])
