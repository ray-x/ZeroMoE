#!/usr/bin/env python3
"""Compare MOESTREAM_DECODE_LOGITS_DUMP files on identical token histories.

Each record is native-endian uint32 step, uint32 vocabulary size, then float32
logits. Dumps include step 0 (the prefill prediction), before sampler penalties.
No third-party packages are required. KL and TV use softmax at temperature 1.
"""

import argparse
import array
import json
import math
import struct
from pathlib import Path


def records(path):
    with Path(path).open("rb") as stream:
        expected = 0
        vocab = None
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError(f"{path}: truncated record header")
            step, n = struct.unpack("=II", header)
            if step != expected or n == 0 or (vocab is not None and n != vocab):
                raise ValueError(f"{path}: invalid step/vocabulary: {step}/{n}")
            payload = stream.read(4 * n)
            if len(payload) != 4 * n:
                raise ValueError(f"{path}: truncated logits at step {step}")
            logits = array.array("f")
            logits.frombytes(payload)
            if not all(math.isfinite(x) for x in logits):
                raise ValueError(f"{path}: nonfinite logits at step {step}")
            yield step, logits
            expected += 1
            vocab = n


def metrics(a, b, token=None):
    if len(a) != len(b):
        raise ValueError("vocabulary sizes differ")
    n = len(a)
    if n < 2:
        raise ValueError("need at least two vocabulary entries")
    ia = max(range(n), key=a.__getitem__)
    ib = max(range(n), key=b.__getitem__)
    second = max(x for i, x in enumerate(a) if i != ia)
    ma, mb = a[ia], b[ib]
    ea = [math.exp(x - ma) for x in a]
    eb = [math.exp(x - mb) for x in b]
    za, zb = math.fsum(ea), math.fsum(eb)
    logza, logzb = ma + math.log(za), mb + math.log(zb)
    delta = [y - x for x, y in zip(a, b)]
    mean_delta = math.fsum(delta) / n
    # KL uses log probabilities directly, avoiding underflow in log(p/q).
    kl = math.fsum(p / za * (-d + logzb - logza)
                   for p, d in zip(ea, delta))
    result = {
        "reference_top1": ia,
        "candidate_top1": ib,
        "top1_agrees": ia == ib,
        "reference_margin": ma - second,
        "reference_top1_candidate_rank": 1 + sum(x > b[ia] for x in b),
        "kl_ref_candidate": max(0.0, kl),
        "total_variation": math.fsum(abs(p / za - q / zb)
                                      for p, q in zip(ea, eb)) / 2,
        "max_abs_delta": max(abs(d) for d in delta),
        "rms_delta": math.sqrt(math.fsum(d * d for d in delta) / n),
        "centered_rms_delta": math.sqrt(math.fsum((d - mean_delta) ** 2
                                                  for d in delta) / n),
        "mean_delta": mean_delta,
    }
    if token is not None:
        if not 0 <= token < n:
            raise ValueError(f"invalid teacher token: {token}")
        result["teacher_token"] = token
        result["teacher_nll_delta"] = -delta[token] + logzb - logza
    return result


def compare(reference, candidate, tokens=None, expected_steps=None):
    ra, rb = records(reference), records(candidate)
    rows = []
    while True:
        a, b = next(ra, None), next(rb, None)
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("record counts differ")
            break
        step, logits = a
        if step != b[0]:
            raise ValueError("steps differ")
        if tokens is not None and step >= len(tokens):
            raise ValueError("teacher token sequence is shorter than the logits dump")
        token = None if tokens is None else tokens[step]
        rows.append({"step": step, **metrics(logits, b[1], token)})
    if not rows or (expected_steps is not None and len(rows) != expected_steps):
        raise ValueError(f"expected {expected_steps} records, found {len(rows)}")
    summary = {"steps": len(rows), "top1_agreement":
               sum(r["top1_agrees"] for r in rows) / len(rows)}
    for key in ("kl_ref_candidate", "total_variation", "max_abs_delta",
                "rms_delta", "centered_rms_delta"):
        summary[f"mean_{key}"] = math.fsum(r[key] for r in rows) / len(rows)
        summary[f"max_{key}"] = max(r[key] for r in rows)
    if tokens is not None:
        summary["mean_teacher_nll_delta"] = math.fsum(
            r["teacher_nll_delta"] for r in rows) / len(rows)
    summary["first_top1_disagreement"] = next(
        (r for r in rows if not r["top1_agrees"]), None)
    return {"reference": str(reference), "candidate": str(candidate),
            "summary": summary, "per_step": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    tokens = None if args.tokens is None else [int(x) for x in args.tokens.read_text().split()]
    result = compare(args.reference, args.candidate, tokens, args.steps)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
