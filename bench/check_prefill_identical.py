#!/usr/bin/env python3
"""Exact prefill equality check between two MOESTREAM_QWEN4EXP_ALLLAYERS_DUMP directories.

Compares the batched prefill snapshot (state.step0.f32, layer-major, `sequence` positions per layer) element for
element. Exit 0 if every position of every layer is bit-identical, 1 otherwise, printing the first differing layer
per position. Guards against graph changes (e.g. marking router tensors ggml_set_output) that perturb prefill
numerics without changing what is computed.
"""
import array, json, sys
from pathlib import Path


def load(d):
    meta = json.loads(Path(f'{d}/state.step0.f32.json').read_text())
    a = array.array('f')
    a.frombytes(Path(f'{d}/state.step0.f32').read_bytes())
    return meta, a


def main():
    if len(sys.argv) != 3:
        sys.exit('usage: check_prefill_identical.py REF_DIR CAND_DIR')
    (mr, ref), (mc, cand) = load(sys.argv[1]), load(sys.argv[2])
    if (mr['layers'], mr['sequence'], mr['layer_elements']) != (mc['layers'], mc['sequence'], mc['layer_elements']):
        sys.exit(f'shape mismatch: {mr} vs {mc}')
    layers, seq = mr['layers'], mr['sequence']
    row = mr['layer_elements'] // seq
    bad = {}
    for p in range(seq):
        for l in range(layers):
            o = (l * seq + p) * row
            if ref[o:o + row] != cand[o:o + row]:
                bad[p] = l
                break
    if bad:
        print(f'FAIL: {len(bad)}/{seq} positions differ; first differing layer per position: {bad}')
        sys.exit(1)
    print(f'OK: prefill bit-identical ({seq} positions x {layers} layers)')


if __name__ == '__main__':
    main()
