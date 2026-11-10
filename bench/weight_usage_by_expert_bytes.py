#!/usr/bin/env python3
"""Weight a `layer expert count` usage file by per-expert bytes from a GGUF.

The engine's `pin_strategy=global_topn` ranks raw route counts, so a miss in a
Q4_K layer (14.2 MB/expert) costs the same as a miss in a Q2_K layer (8.3 MB).
Scaling each layer's counts by its expert bytes makes the global top-N prefer
the expensive early layers.

  bench/weight_usage_by_expert_bytes.py <usage_in> <usage_out> --gguf <model.gguf>

The output remains in the pin-file format: "layer expert weighted_count".
`--reference min` normalizes to the smallest MoE layer (default); `--reference L`
normalizes to layer L's bytes.
"""
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from gguf_min import parse  # noqa: E402

# ggml type -> (block elements, block bytes) for the quant types this model uses.
BLOCK = {
    10: (256, 84),    # Q2_K
    11: (256, 110),   # Q3_K
    12: (256, 144),   # Q4_K
    13: (256, 176),   # Q5_K
    14: (256, 210),   # Q6_K
    8:  (32, 34),     # Q8_0
    0:  (1, 4),       # F32
    1:  (1, 2),       # F16
    30: (1, 2),       # BF16
}


def per_expert_bytes(gguf_path):
    """layer -> (gate+up+down bytes per expert, type names)."""
    tensors = parse(gguf_path)
    layer_bytes = defaultdict(int)
    layer_types = defaultdict(set)
    for name, (dims, ty, _off) in tensors.items():
        if not name.endswith(('ffn_gate_exps.weight', 'ffn_up_exps.weight',
                              'ffn_down_exps.weight')):
            continue
        # blk.<L>.ffn_*_exps.weight
        parts = name.split('.')
        if len(parts) < 4 or parts[0] != 'blk':
            continue
        layer = int(parts[1])
        if dims[2] <= 0:
            continue
        elems = 1
        for d in dims:
            elems *= d
        if ty not in BLOCK:
            raise SystemExit(f'{name}: unsupported type {ty}')
        block_elems, block_bytes = BLOCK[ty]
        if elems % block_elems:
            raise SystemExit(f'{name}: {elems} not a multiple of {block_elems}')
        total = (elems // block_elems) * block_bytes
        layer_bytes[layer] += total // dims[2]
        layer_types[layer].add(ty)
    return dict(layer_bytes), {k: sorted(v) for k, v in layer_types.items()}


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    usage_in, usage_out = sys.argv[1], sys.argv[2]
    gguf = None
    reference = 'min'
    for i, a in enumerate(sys.argv):
        if a == '--gguf' and i + 1 < len(sys.argv):
            gguf = sys.argv[i + 1]
        if a == '--reference' and i + 1 < len(sys.argv):
            reference = sys.argv[i + 1]
    if not gguf:
        raise SystemExit('--gguf is required')

    layer_bytes, layer_types = per_expert_bytes(gguf)
    if not layer_bytes:
        raise SystemExit('no *_exps tensors found')
    ref = min(layer_bytes.values()) if reference == 'min' else layer_bytes[int(reference)]
    weights = {l: b / ref for l, b in layer_bytes.items()}

    counts = defaultdict(int)
    with open(usage_in) as f:
        for line in f:
            t = line.split()
            if len(t) == 3:
                counts[(int(t[0]), int(t[1]))] += int(t[2])

    out = []
    for (layer, expert), count in counts.items():
        w = weights.get(layer, 1.0)
        out.append((layer, expert, max(1, int(round(count * w)))))
    out.sort()
    with open(usage_out, 'w') as f:
        for layer, expert, count in out:
            f.write(f'{layer} {expert} {count}\n')

    print(f'wrote {usage_out}: {len(out)} entries, total {sum(c for _, _, c in out)}')
    for l in sorted(layer_bytes):
        if l in layer_types:
            types = ','.join({10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K',
                              14: 'Q6_K', 8: 'Q8_0'}.get(t, str(t))
                             for t in layer_types[l])
            print(f'  L{l:2d} {layer_bytes[l]/1e6:6.2f} MB/expert  x{weights[l]:.3f}  {types}')


if __name__ == '__main__':
    main()
