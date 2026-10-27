#!/usr/bin/env python3
"""Compare teacher-forced MTP draft argmaxes from ZeroMoE/Bee trace dumps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


FIXTURE_MAGIC = "ZEROMOE_MTP_FIXTURE_V1"
DUMP_MAGIC = "ZEROMOE_MTP_DUMP_V1"
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fixture(path: Path):
    raw = path.read_bytes()
    fingerprint = FNV_OFFSET
    for byte in raw:
        fingerprint = ((fingerprint ^ byte) * FNV_PRIME) & MASK64

    lines = raw.decode("ascii").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty fixture")
    header = lines[0].split()
    if len(header) != 5 or header[0] != FIXTURE_MAGIC:
        raise ValueError(f"{path}: invalid fixture header")
    d, w, vocab, rows = map(int, header[1:])
    if d <= 0 or w < d or vocab <= 0 or rows <= 0 or len(lines) != rows + 1:
        raise ValueError(f"{path}: invalid fixture dimensions or row count")

    tokens = []
    for row, line in enumerate(lines[1:]):
        fields = line.split()
        if len(fields) != 2 + d + w:
            raise ValueError(f"{path}: row {row} has the wrong number of values")
        token, position = map(int, fields[:2])
        if not 0 <= token < vocab or position != row:
            raise ValueError(f"{path}: invalid token or position at row {row}")
        tokens.append(token)
    return (fingerprint, d, w, vocab, rows), tokens


def top1_logits(path: Path, expected_header, phase: int):
    result = {}
    with path.open(encoding="ascii") as source:
        header = source.readline().split()
        if len(header) != 6 or header[0] != DUMP_MAGIC:
            raise ValueError(f"{path}: invalid dump header")
        fingerprint, d, w, vocab, rows = map(int, header[1:])
        if (fingerprint, d, w, vocab, rows) != expected_header:
            raise ValueError(f"{path}: fixture fingerprint or dimensions differ")

        for line_number, line in enumerate(source, 2):
            fields = line.split(maxsplit=3)
            if len(fields) != 4:
                raise ValueError(f"{path}:{line_number}: malformed dump record")
            record_phase, row = int(fields[0]), int(fields[1])
            stage = fields[2]
            if record_phase != phase or stage != "logits":
                continue
            if not 0 <= row < rows or row in result:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate logits row")
            count = int(fields[3].split(maxsplit=1)[0])
            value_text = fields[3][len(str(count)):].strip()
            if count != vocab:
                raise ValueError(f"{path}:{line_number}: logits size is {count}, expected {vocab}")

            best_id = -1
            best = second = -math.inf
            n_values = 0
            for n_values, text in enumerate(value_text.split(), 1):
                value = float(text)
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: non-finite logit")
                if value > best:
                    second, best, best_id = best, value, n_values - 1
                elif value > second:
                    second = value
            if n_values != vocab:
                raise ValueError(f"{path}:{line_number}: read {n_values} logits, expected {vocab}")
            result[row] = (best_id, best - second)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("candidate_dump", type=Path)
    parser.add_argument("reference_dump", type=Path)
    parser.add_argument("--phase", type=int, default=0)
    args = parser.parse_args()

    try:
        header, tokens = fixture(args.fixture)
        candidate = top1_logits(args.candidate_dump, header, args.phase)
        reference = top1_logits(args.reference_dump, header, args.phase)
        common = sorted(candidate.keys() & reference.keys())
        if not common:
            raise ValueError("the dumps contain no common logits rows for this phase")
        transitions = [row for row in common if row + 1 < len(tokens)]
        agreements = sum(candidate[row][0] == reference[row][0] for row in common)
        candidate_correct = sum(candidate[row][0] == tokens[row + 1] for row in transitions)
        reference_correct = sum(reference[row][0] == tokens[row + 1] for row in transitions)

        print(f"phase={args.phase}; candidate rows={len(candidate)}; reference rows={len(reference)}; common={len(common)}")
        print(f"draft top-1 agreement: {agreements}/{len(common)} ({100 * agreements / len(common):.2f}%)")
        print(f"fixture next-token top-1: candidate={candidate_correct}/{len(transitions)}; reference={reference_correct}/{len(transitions)}")
        for row in common:
            candidate_id = candidate[row][0]
            reference_id = reference[row][0]
            if candidate_id != reference_id:
                print(f"mismatch row={row}: candidate={candidate_id} margin={candidate[row][1]:.6g}; "
                      f"reference={reference_id} margin={reference[row][1]:.6g}")
        return 0
    except (OSError, UnicodeError, ValueError, OverflowError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
