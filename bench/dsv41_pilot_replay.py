#!/usr/bin/env python3
"""Offline accuracy of a DeepSeek V4.1 expert 'pilot' (next-layer router on the
current layer's hidden state), from MOESTREAM_ROUTER_IN_LOG + MOESTREAM_ROUTE_LOG.

For each decode token and layer L:
  actual(L)      = the experts the engine really picked at layer L
  self-check     = router_L applied to layer L's own router input must reproduce
                   actual(L) (validates the gating assumption; hash-routed layers
                   fail it and are skipped)
  pilot(L+1)     = top-k of  sqrtsoftplus(W_{L+1} . x_L) + bias_{L+1}
                   where x_L is layer L's router input (available about one layer
                   before layer L+1's own router runs)
Reported per guess-size k: precision (used / k) and the share of COLD picks the
pilot names, where cold = not in (hottest-120 of the other prompts | prefill set |
last 16 tokens), the picks that actually stall decode.

  dsv41_pilot_replay.py --model-dir DIR --run NAME[,NAME...] --dir LOGDIR
Logs are LOGDIR/routein_NAME.bin and LOGDIR/route_NAME.log.
"""
import argparse
import collections
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_min  # noqa: E402

D, NE, TOPK = 5120, 384, 6


def bf16(path, off, n):
    u = np.fromfile(path, dtype=np.uint16, count=n, offset=off).astype(np.uint32) << 16
    return u.view(np.float32)


def load_router(model_dir):
    W, B = {}, {}
    for name in sorted(f for f in os.listdir(model_dir) if f.endswith(".gguf")):
        path = os.path.join(model_dir, name)
        for t, (dims, ty, off) in gguf_min.parse(path).items():
            m = re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", t)
            if m:
                assert ty == 30 and tuple(dims) == (D, NE), (t, dims, ty)
                W[int(m.group(1))] = bf16(path, off, D * NE).reshape(NE, D)
            m = re.fullmatch(r"blk\.(\d+)\.exp_probs_b\.bias", t)
            if m:
                assert ty == 0, (t, ty)
                B[int(m.group(1))] = np.fromfile(path, dtype=np.float32, count=NE, offset=off)
    return W, B


def score(W, B, L, x):
    z = W[L] @ x
    sp = np.where(z > 0, z + np.log1p(np.exp(-np.abs(z))), np.log1p(np.exp(-np.abs(z)) * (z <= 0)) + 0.0)
    sp = np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
    return np.sqrt(sp) + B.get(L, 0.0)


def read_runs(dirp, name):
    ids = collections.defaultdict(list)
    pre = collections.defaultdict(list)
    for line in open(os.path.join(dirp, f"route_{name}.log")):
        p = line.split()
        L, S = int(p[0][1:]), int(p[1][1:])
        (pre if S > 1 else ids)[L].append([int(v) for v in p[3:]])
    xs = collections.defaultdict(list)  # decode only, in call order per layer
    with open(os.path.join(dirp, f"routein_{name}.bin"), "rb") as f:
        rec = 12 + 4 * D
        while True:
            b = f.read(rec)
            if len(b) < rec:
                break
            L, S, s = struct.unpack("<3i", b[:12])
            if S == 1:
                xs[L].append(np.frombuffer(b[12:], dtype=np.float32))
    return ids, pre, xs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    names = args.run.split(",")
    W, B = load_router(args.model_dir)
    layers = sorted(W)
    print(f"router weights for layers {layers[0]}..{layers[-1]} ({len(layers)}), bias layers {len(B)}")
    data = {n: read_runs(args.dir, n) for n in names}
    # hash-routed / unreliable layers: self-check
    ok_layers = []
    for L in layers:
        hit = tot = 0
        for n in names:
            ids, _, xs = data[n]
            for t in range(min(len(ids[L]), len(xs[L]), 20)):
                top = set(np.argsort(-score(W, B, L, xs[L][t]))[:TOPK])
                hit += len(top & set(ids[L][t])); tot += TOPK
        if tot and hit / tot > 0.95:
            ok_layers.append(L)
        elif tot:
            print(f"  layer {L}: self-check {100*hit/tot:.0f}% -> skipped (hash-routed or gating mismatch)")
    print(f"layers passing the self-check (router reproduces the engine's own picks): {len(ok_layers)}")
    ok = set(ok_layers)
    Ks = (1, 2, 3, 4, 5, 6, 8, 12)
    res = {k: [0, 0] for k in Ks}      # [hits over all picks, hits over cold picks]
    npred = ncold = 0
    for test in names:
        ids, pre, xs = data[test]
        others = [n for n in names if n != test]
        hot = collections.defaultdict(collections.Counter)
        for n in others:
            for L, rows in data[n][0].items():
                for r in rows:
                    hot[L].update(r)
        for L in layers:
            if L + 1 not in ok or L not in xs:
                continue
            static = set(e for e, _ in hot[L + 1].most_common(120))
            pset = set(e for r in pre[L + 1] for e in r)
            T = min(len(ids[L + 1]), len(xs[L]))
            for t in range(16, T):
                actual = set(ids[L + 1][t])
                recent = set(e for k in range(t - 16, t) for e in ids[L + 1][k])
                cold = set(e for e in actual if e not in static and e not in pset and e not in recent)
                order = np.argsort(-score(W, B, L + 1, xs[L][t]))
                npred += 1; ncold += len(cold)
                for k in Ks:
                    g = set(int(e) for e in order[:k])
                    res[k][0] += len(g & actual); res[k][1] += len(g & cold)
    print(f"\nscored {npred} (layer, token) predictions; cold picks {ncold} "
          f"({100*ncold/max(1,npred*TOPK):.1f}% of picks)")
    print("guess-set k | used/k (precision) | recall of the 6 | share of COLD picks named")
    for k in Ks:
        print(f"  top-{k:<2d}      | {100*res[k][0]/(npred*k):5.1f}%             | "
              f"{100*res[k][0]/(npred*TOPK):5.1f}%          | {100*res[k][1]/max(1,ncold):5.1f}%")
    print(f"random guessing: precision {100*TOPK/NE:.1f}%")


if __name__ == "__main__":
    main()
