#!/usr/bin/env python3
"""Compare two ZEROMOE_DECODE_LOGITS_DUMP files (records: uint32 step, uint32 n, n float32), standard library only.
Per step: bitwise equality, max_abs, rmse, cosine, top1 and whether all values are finite."""
import sys, struct, math, array
def load(p):
    b = open(p, 'rb').read(); out = []; o = 0
    while o + 8 <= len(b):
        step, n = struct.unpack_from('<II', b, o); o += 8
        a = array.array('f'); a.frombytes(b[o:o + 4 * n]); o += 4 * n
        out.append((step, a, b[o - 4 * n:o]))
    return out
A, B = load(sys.argv[1]), load(sys.argv[2])
print(f'records: {len(A)} vs {len(B)}')
allbit = len(A) == len(B)
for (sa, x, rx), (sb, y, ry) in zip(A, B):
    eq = rx == ry; allbit &= eq
    mx = rm = dot = nx = ny = 0.0; fin = True
    for u, v in zip(x, y):
        d = u - v; mx = max(mx, abs(d)); rm += d * d; dot += u * v; nx += u * u; ny += v * v
        if not (math.isfinite(u) and math.isfinite(v)): fin = False
    cos = dot / math.sqrt(nx * ny) if nx and ny else float('nan')
    t1 = max(range(len(x)), key=x.__getitem__); t2 = max(range(len(y)), key=y.__getitem__)
    print(f'step={sa}/{sb} bitwise_equal={eq} max_abs={mx:.6g} rmse={math.sqrt(rm/len(x)):.6g} cosine={cos:.9f} top1={t1}/{t2} finite={fin}')
print('ALL STEPS BITWISE EQUAL' if allbit else 'NOT bitwise equal')
