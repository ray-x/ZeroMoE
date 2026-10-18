#!/usr/bin/env python3
"""Build a SPARSE partial mirror of the DeepSeek V4.1 shards on a second drive.

The mirror holds only the `ffn_down_exps` slices of the least-used experts
(ranked from a usage survey). Each mirror shard has the same basename and the
same apparent size as its primary (the engine's open_mirror() checks both),
but is sparse: only the selected slices occupy disk. A coverage list
("layer expert" per line) tells the engine which experts may use the
intra-expert split (gate+up from the primary, down from the mirror); every
other expert reads from the primary alone, so a hole is never read.

Layers 0-1 store experts as per-expert tensors (not packed) and are skipped.

  dsv41_pack_cold_mirror.py --model-dir DIR --mirror-dir DIR \
      --usage survey_usage [--usage old_usage ...] --budget-gb 48 \
      --coverage-out FILE [--dry-run]

Usage files are "layer expert count" lines; the first file is the primary
ranking, later ones only break ties.
"""
import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_min  # noqa: E402

GGML_Q3_K = 11
Q3_K_BLOCK_ELEMS, Q3_K_BLOCK_BYTES = 256, 110


def load_usage(path):
    d = {}
    for line in open(path):
        a = line.split()
        if len(a) == 3:
            d[(int(a[0]), int(a[1]))] = int(a[2])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mirror-dir", required=True)
    ap.add_argument("--usage", action="append", required=True)
    ap.add_argument("--budget-gb", type=float, default=48.0,
                    help="decimal GB of down slices to place on the mirror")
    ap.add_argument("--coverage-out", required=True)
    ap.add_argument("--verify-fraction", type=float, default=0.03)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    usage = [load_usage(p) for p in args.usage]
    shards = sorted(f for f in os.listdir(args.model_dir) if f.endswith(".gguf"))
    # (layer) -> (shard file, abs offset, per-expert bytes, n_experts)
    layers = {}
    for name in shards:
        path = os.path.join(args.model_dir, name)
        for tname, (dims, ty, off) in gguf_min.parse(path).items():
            m = re.fullmatch(r"blk\.(\d+)\.ffn_down_exps\.weight", tname)
            if not m:
                continue
            if ty != GGML_Q3_K:
                sys.exit(f"{tname}: expected Q3_K down (type {GGML_Q3_K}), got {ty}")
            ne0, ne1, n_exp = dims
            if ne0 % Q3_K_BLOCK_ELEMS:
                sys.exit(f"{tname}: row length {ne0} is not a multiple of 256")
            per_expert = (ne0 // Q3_K_BLOCK_ELEMS) * Q3_K_BLOCK_BYTES * ne1
            layers[int(m.group(1))] = (name, off, per_expert, n_exp)
    if not layers:
        sys.exit("no packed ffn_down_exps tensors found")
    per_expert = {v[2] for v in layers.values()}
    print(f"packed layers: {len(layers)} ({min(layers)}..{max(layers)}), "
          f"down slice bytes per expert: {sorted(per_expert)}")

    cand = [(l, e) for l, (_, _, _, ne) in layers.items() for e in range(ne)]
    cand.sort(key=lambda k: tuple(u.get(k, 0) for u in usage) + (k[0], k[1]))
    chosen, total = [], 0
    for k in cand:
        sz = layers[k[0]][2]
        if total + sz > args.budget_gb * 1e9:
            break
        chosen.append(k)
        total += sz
    print(f"selected {len(chosen)} of {len(cand)} experts = {total/1e9:.1f} GB "
          f"of down slices (budget {args.budget_gb} GB)")
    first_use = usage[0]
    print("survey picks covered by the selection: "
          f"{sum(first_use.get(k, 0) for k in chosen)} of {sum(first_use.values())} "
          f"({100*sum(first_use.get(k,0) for k in chosen)/max(1,sum(first_use.values())):.2f}%)")
    if args.dry_run:
        return

    os.makedirs(args.mirror_dir, exist_ok=True)
    out_fds = {}
    for name in shards:
        src = os.path.join(args.model_dir, name)
        dst = os.path.join(args.mirror_dir, name)
        fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, os.stat(src).st_size)   # apparent size == primary
        out_fds[name] = fd
    src_fds = {n: os.open(os.path.join(args.model_dir, n), os.O_RDONLY) for n in shards}

    by_layer = {}
    for l, e in chosen:
        by_layer.setdefault(l, []).append(e)
    done = 0
    for l in sorted(by_layer):
        name, off, sz, _ = layers[l]
        for e in sorted(by_layer[l]):
            buf = os.pread(src_fds[name], sz, off + e * sz)
            if len(buf) != sz:
                sys.exit(f"short read L{l} E{e}")
            os.pwrite(out_fds[name], buf, off + e * sz)
            done += 1
        for n in src_fds:
            os.posix_fadvise(src_fds[n], 0, 0, os.POSIX_FADV_DONTNEED)
        for n in out_fds:
            os.fdatasync(out_fds[n])
            os.posix_fadvise(out_fds[n], 0, 0, os.POSIX_FADV_DONTNEED)
        print(f"  layer {l}: {len(by_layer[l])} experts ({done}/{len(chosen)})", flush=True)

    # Verify a random sample byte-for-byte (fresh reads from both drives).
    rng = random.Random(1)
    sample = rng.sample(chosen, max(1, int(len(chosen) * args.verify_fraction)))
    bad = 0
    for l, e in sample:
        name, off, sz, _ = layers[l]
        a = os.pread(src_fds[name], sz, off + e * sz)
        b = os.pread(out_fds[name], sz, off + e * sz)
        bad += a != b
    print(f"verify: {len(sample)} sampled slices, {bad} mismatches")
    if bad:
        sys.exit("VERIFY FAILED")
    with open(args.coverage_out, "w") as f:
        for l, e in sorted(chosen):
            f.write(f"{l} {e}\n")
    print(f"coverage written: {args.coverage_out} ({len(chosen)} experts)")


if __name__ == "__main__":
    main()
