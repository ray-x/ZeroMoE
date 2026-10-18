#!/usr/bin/env python3
"""IO benchmark of mirror layouts for DeepSeek V4.1 expert reads.

Compares, for k concurrent cold experts of ONE layer (what a layer's miss batch
looks like), the wall time to read them with O_DIRECT from real shard offsets:

  P  primary only: gate+up+down of every expert from the primary drive
  W  whole-expert striping: each expert lives on ONE drive (half per drive);
     needs the mirror to hold full experts
  S  intra-expert split: gate+up from the primary, down from the mirror
     (the mirror holds only `down`)
  H  hybrid: split when k == 1, whole-expert striping when k >= 2 (needs full
     experts on the mirror)

A sparse bench file on the mirror drive holds the full gate/up/down slices of
the sampled experts at the SAME offsets as the primary shard (that is how the
engine's mirror works), so the access pattern matches the real one. `down`
slices are not 4 KiB aligned (stride 5,068,800), so each read is widened to
aligned boundaries exactly as the engine's staged path must.

  dsv41_mirror_io_bench.py --shard FILE --mirror-dir DIR [--trials N]
"""
import argparse
import mmap
import os
import random
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_min  # noqa: E402

ALIGN = 4096


def aligned(off, n):
    a0 = off // ALIGN * ALIGN
    a1 = (off + n + ALIGN - 1) // ALIGN * ALIGN
    return a0, a1 - a0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--mirror-dir", required=True)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--pool-experts", type=int, default=480)
    ap.add_argument("--keep", action="store_true", help="keep the bench file")
    args = ap.parse_args()

    tens = gguf_min.parse(args.shard)
    layers = {}
    for name, (dims, ty, off) in tens.items():
        m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if m:
            layers.setdefault(int(m.group(1)), {})[m.group(2)] = (dims, ty, off)
    layers = {l: t for l, t in layers.items() if len(t) == 3}
    if not layers:
        sys.exit("shard has no complete packed expert layers")

    def per_expert(dims, ty):
        ne0, ne1, _ = dims
        bb = {10: 84, 11: 110}.get(ty)
        if bb is None:
            sys.exit(f"unsupported ggml type {ty}")
        return (ne0 // 256) * bb * ne1

    geo = {}
    for l, t in layers.items():
        geo[l] = {k: (t[k][2], per_expert(t[k][0], t[k][1]), t[k][0][2]) for k in t}
    l0 = sorted(layers)[0]
    gb, ub, db = (geo[l0][k][1] for k in ("gate", "up", "down"))
    print(f"shard layers {min(layers)}..{max(layers)} ({len(layers)}), expert slices: "
          f"gate {gb} up {ub} down {db} = {gb+ub+db} bytes; "
          f"down is {100*db/(gb+ub+db):.0f}% of an expert, gate+up {100*(gb+ub)/(gb+ub+db):.0f}%")

    rng = random.Random(7)
    # Pool: experts grouped by layer so a trial can draw k from the SAME layer.
    pool = {}
    per_layer = max(8, args.pool_experts // len(layers))
    for l in sorted(layers):
        n_exp = geo[l]["gate"][2]
        pool[l] = rng.sample(range(n_exp), min(per_layer, n_exp))
    total_experts = sum(len(v) for v in pool.values())

    os.makedirs(args.mirror_dir, exist_ok=True)
    bpath = os.path.join(args.mirror_dir, "iobench_full.bin")
    sz = os.stat(args.shard).st_size
    bfd = os.open(bpath, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(bfd, sz)
    pfd_buf = os.open(args.shard, os.O_RDONLY)
    print(f"building bench mirror: {total_experts} full experts "
          f"({total_experts*(gb+ub+db)/1e9:.1f} GB) at primary offsets in {bpath}")
    for l, es in pool.items():
        for e in es:
            for k in ("gate", "up", "down"):
                off, n, _ = geo[l][k]
                buf = os.pread(pfd_buf, n, off + e * n)
                os.pwrite(bfd, buf, off + e * n)
    os.fsync(bfd)
    os.posix_fadvise(bfd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(bfd)
    os.close(pfd_buf)

    fdA = os.open(args.shard, os.O_RDONLY | os.O_DIRECT)
    fdB = os.open(bpath, os.O_RDONLY | os.O_DIRECT)
    bufs = [mmap.mmap(-1, 12 * 1024 * 1024 + 2 * ALIGN) for _ in range(64)]

    def read(fd, off, n, slot):
        a0, an = aligned(off, n)
        return os.preadv(fd, [memoryview(bufs[slot])[:an]], a0)

    pool_exec = ThreadPoolExecutor(max_workers=24)

    def batch(reqs):
        t0 = time.perf_counter()
        futs = [pool_exec.submit(read, fd, off, n, i) for i, (fd, off, n) in enumerate(reqs)]
        for f in futs:
            f.result()
        return (time.perf_counter() - t0) * 1e3

    def expert_reqs(l, e, drive_gu, drive_down):
        r = []
        for k, d in (("gate", drive_gu), ("up", drive_gu), ("down", drive_down)):
            off, n, _ = geo[l][k]
            r.append((fdA if d == 0 else fdB, off + e * n, n))
        return r

    def build(mode, l, es):
        reqs = []
        for i, e in enumerate(es):
            if mode == "P":
                reqs += expert_reqs(l, e, 0, 0)
            elif mode == "W":
                reqs += expert_reqs(l, e, i % 2, i % 2)
            elif mode == "S":
                reqs += expert_reqs(l, e, 0, 1)
            elif mode == "H":
                reqs += expert_reqs(l, e, 0, 1) if len(es) == 1 else expert_reqs(l, e, i % 2, i % 2)
        return reqs

    # Warm threads / drives once, discard.
    for _ in range(6):
        l = rng.choice(list(pool))
        batch(build("P", l, rng.sample(pool[l], 2)))

    print(f"\n{'k':>2} {'mode':>4} {'mean ms':>8} {'p50':>7} {'p95':>7} {'MB/s':>7}   (trials={args.trials}, same experts for every mode)")
    results = {}
    for k in (1, 2, 3, 4, 6):
        # Identical draws for every mode so only the layout differs.
        draws = []
        for _ in range(args.trials):
            l = rng.choice(list(pool))
            draws.append((l, rng.sample(pool[l], k)))
        for mode in ("P", "W", "S", "H"):
            ts = [batch(build(mode, l, es)) for l, es in draws]
            mb = k * (gb + ub + db) / 1e6
            results[(k, mode)] = st.mean(ts)
            print(f"{k:>2} {mode:>4} {st.mean(ts):8.2f} {st.median(ts):7.2f} "
                  f"{sorted(ts)[int(.95*len(ts))-1]:7.2f} {mb/st.mean(ts)*1000:7.0f}")
    print("\nspeedup vs P (higher is better):")
    for k in (1, 2, 3, 4, 6):
        print(f"  k={k}: " + "  ".join(f"{m}={results[(k,'P')]/results[(k,m)]:.2f}x"
                                       for m in ("W", "S", "H")))
    if not args.keep:
        os.unlink(bpath)
        print("bench file removed")


if __name__ == "__main__":
    main()
