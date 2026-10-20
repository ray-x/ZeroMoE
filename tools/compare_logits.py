#!/usr/bin/env python3
"""Compare MoEStream and llama.cpp raw decode-logit dumps."""

import argparse
import math
import struct
from pathlib import Path


def load(path):
    data = memoryview(Path(path).read_bytes())
    result = []
    offset = 0
    while offset < len(data):
        step, width = struct.unpack_from("<II", data, offset)
        offset += 8
        row = struct.unpack_from(f"<{width}f", data, offset)
        offset += width * 4
        if offset > len(data):
            raise ValueError(f"truncated logit row in {path}")
        result.append((step, row))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moestream")
    parser.add_argument("llama")
    args = parser.parse_args()
    a, b = load(args.moestream), load(args.llama)
    if len(a) != len(b):
        raise SystemExit(f"row count differs: moestream={len(a)} llama={len(b)}")
    for (sa, x), (sb, y) in zip(a, b):
        if sa != sb or len(x) != len(y):
            raise SystemExit(f"row layout differs at {sa}/{sb}: {len(x)}/{len(y)}")
        diffs = [abs(i - j) for i, j in zip(x, y)]
        top_x = max(range(len(x)), key=x.__getitem__)
        top_y = max(range(len(y)), key=y.__getitem__)
        dot = sum(i * j for i, j in zip(x, y))
        nx = math.sqrt(sum(i * i for i in x))
        ny = math.sqrt(sum(j * j for j in y))
        print(f"step={sa} vocab={len(x)} max_abs={max(diffs):.7g} "
              f"rmse={math.sqrt(sum(d*d for d in diffs)/len(diffs)):.7g} "
              f"cosine={dot/(nx*ny):.9g} top1={top_x}/{top_y} "
              f"top1_match={top_x == top_y}")


if __name__ == "__main__":
    main()
