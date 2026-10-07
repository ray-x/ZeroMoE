#!/usr/bin/env python3
"""Audit lossless byte-compression headroom in GGUF quantized MoE tensors.

This is an offline feasibility tool. It never writes into a GGUF file and
does not load tensor data into VRAM. The tensor table and mmap-backed data
views come from the `gguf` Python package.
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

try:
    import numpy as np
    import gguf
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "Missing Python dependency. Run with an environment containing "
        "llama.cpp's gguf-py package and NumPy."
    ) from exc


EXPERT_NAME = re.compile(
    r"^blk\.(\d+)\.ffn_(gate|up|down)(?:_exps|\.\d+)\.weight$"
)
DEFAULT_TILE_BYTES = 256
IO_CHUNK_BYTES = 4 * 1024 * 1024
RANS_SCALE_BITS = 12
RANS_TOTAL = 1 << RANS_SCALE_BITS
RANS_LOWER_BOUND = 1 << 23


@dataclass
class TensorRef:
    path: str
    reader: object
    tensor: object
    layer: int
    projection: str


def model_files(path: str) -> list[str]:
    candidate = Path(path)
    if candidate.is_dir():
        files = sorted(glob.glob(str(candidate / "*.gguf")))
    else:
        files = [str(candidate)]
    if not files:
        raise ValueError(f"No GGUF files found at {path}")
    return files


def discover_tensors(paths: Iterable[str]) -> list[TensorRef]:
    found: list[TensorRef] = []
    # Keep readers alive: each tensor's data array is a view into its mmap.
    for path in paths:
        reader = gguf.GGUFReader(path)
        for tensor in reader.tensors:
            match = EXPERT_NAME.fullmatch(tensor.name)
            if not match:
                continue
            # Quantized GGUF MoE variants can mix IQ types with the K-family
            # formats (for example Q4_K gate/up and Q5_K down tensors).
            if not tensor.tensor_type.name.startswith(("Q", "IQ")):
                continue
            found.append(TensorRef(
                path=path,
                reader=reader,
                tensor=tensor,
                layer=int(match.group(1)),
                projection=match.group(2),
            ))
    return found


def choose_tensors(
    found: list[TensorRef], layers: set[int] | None, max_tensors: int
) -> list[TensorRef]:
    candidates = [t for t in found if layers is None or t.layer in layers]
    if not candidates:
        layer_note = f" in layers {sorted(layers)}" if layers else ""
        raise ValueError(f"No quantized expert FFN tensors found{layer_note}")

    candidates.sort(key=lambda t: (t.layer, ("gate", "up", "down").index(t.projection), t.tensor.name))
    # Choose gate/up/down from the earliest available MoE layer first. This
    # gives the default audit projection coverage without requiring a guessed
    # layer index or reading every tensor in the model.
    chosen: list[TensorRef] = []
    by_layer: dict[int, list[TensorRef]] = collections.defaultdict(list)
    for item in candidates:
        by_layer[item.layer].append(item)
    for layer in sorted(by_layer):
        layer_items = by_layer[layer]
        for projection in ("gate", "up", "down"):
            item = next((x for x in layer_items if x.projection == projection), None)
            if item is not None and all(id(item) != id(existing) for existing in chosen):
                chosen.append(item)
        if len(chosen) >= max_tensors:
            break
    if len(chosen) < max_tensors:
        for item in candidates:
            if all(id(item) != id(existing) for existing in chosen):
                chosen.append(item)
            if len(chosen) >= max_tensors:
                break
    return chosen[:max_tensors]


def sample_ranges(
    size: int, budget: int, alignment: int, window_count: int = 8
) -> list[tuple[int, int]]:
    if size <= budget:
        return [(0, size)]
    window = max(1, budget // window_count)
    window -= window % alignment
    if window == 0:
        window = alignment
    count = min(window_count, max(1, budget // window))
    result = []
    for i in range(count):
        start = ((2 * i + 1) * size) // (2 * count) - window // 2
        start = max(0, min(start, size - window))
        start -= start % alignment
        result.append((start, window))
    return result


def byte_views(tensor_data: object, ranges: list[tuple[int, int]]) -> Iterator[memoryview]:
    byte_array = tensor_data.view(np.uint8).reshape(-1)
    for start, length in ranges:
        yield memoryview(byte_array[start : start + length])


def zstd_compress_size(parts: Iterable[memoryview], level: int) -> int:
    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("The `zstd` command is required for the compression baseline")
    with tempfile.TemporaryFile() as encoded:
        process = subprocess.Popen(
            [zstd, "-q", "-c", f"-{level}"],
            stdin=subprocess.PIPE,
            stdout=encoded,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            for part in parts:
                for offset in range(0, len(part), IO_CHUNK_BYTES):
                    process.stdin.write(part[offset : offset + IO_CHUNK_BYTES])
            process.stdin.close()
            error = process.stderr.read() if process.stderr else b""
            status = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        if status != 0:
            raise RuntimeError(f"zstd failed: {error.decode(errors='replace').strip()}")
        encoded.seek(0, os.SEEK_END)
        return encoded.tell()


def palette_estimate(parts: list[memoryview], tile_bytes: int) -> tuple[int, int, int]:
    """Estimate a simple fixed-width local palette with raw-tile fallback.

    Per tile the estimate includes a one-byte mode/count header, the palette,
    and tightly packed fixed-width indices. It omits only GGUF-level framing.
    """
    total = 0
    palette_tiles = 0
    raw_tiles = 0
    for part in parts:
        usable = len(part) - (len(part) % tile_bytes)
        tiles = np.frombuffer(part[:usable], dtype=np.uint8).reshape(-1, tile_bytes)
        if not len(tiles):
            continue
        sorted_tiles = np.sort(tiles, axis=1)
        unique_counts = 1 + np.count_nonzero(sorted_tiles[:, 1:] != sorted_tiles[:, :-1], axis=1)
        for unique_count in unique_counts.tolist():
            bits = max(1, (int(unique_count) - 1).bit_length())
            coded_size = 1 + int(unique_count) + (tile_bytes * bits + 7) // 8
            if coded_size < tile_bytes:
                total += coded_size
                palette_tiles += 1
            else:
                total += tile_bytes + 1
                raw_tiles += 1
        # The final partial tile is stored verbatim; its length follows from
        # the original tensor size, so it needs no per-tile mode byte.
        total += len(part) - usable
    return total, palette_tiles, raw_tiles


def bitshuffle(parts: list[memoryview]) -> Iterator[memoryview]:
    """Transpose groups of eight bytes into their eight bit planes."""
    for part in parts:
        data = np.frombuffer(part, dtype=np.uint8)
        usable = len(data) - (len(data) % 8)
        groups = data[:usable].reshape(-1, 8)
        planes = [
            np.packbits(((groups >> bit) & 1), axis=1, bitorder="little").reshape(-1)
            for bit in range(8)
        ]
        shuffled = np.concatenate(planes) if planes else np.empty(0, dtype=np.uint8)
        if usable:
            yield memoryview(shuffled)
        if usable < len(data):
            yield memoryview(data[usable:])


def bitunshuffle(shuffled: memoryview, original_size: int) -> np.ndarray:
    """Inverse of ``bitshuffle`` for one sample window."""
    usable = original_size - (original_size % 8)
    group_count = usable // 8
    encoded = np.frombuffer(shuffled, dtype=np.uint8)
    planes = encoded[:usable].reshape(8, group_count)
    restored = np.zeros((group_count, 8), dtype=np.uint8)
    for bit in range(8):
        unpacked = np.unpackbits(planes[bit], bitorder="little").reshape(-1, 8)
        restored |= unpacked << bit
    result = np.empty(original_size, dtype=np.uint8)
    result[:usable] = restored.reshape(-1)
    if usable < original_size:
        result[usable:] = encoded[usable : usable + original_size - usable]
    return result


def rans_roundtrip(raw: np.ndarray, position_counts: np.ndarray) -> tuple[int, float, float]:
    """Encode/decode bytes with static rANS tables selected by IQ block offset."""
    block_bytes = position_counts.shape[0]
    frequencies = np.zeros_like(position_counts, dtype=np.uint16)
    cumulative = np.zeros_like(position_counts, dtype=np.uint16)
    inverse = np.zeros((block_bytes, RANS_TOTAL), dtype=np.uint8)

    for position in range(block_bytes):
        row = position_counts[position]
        active = np.flatnonzero(row)
        available = RANS_TOTAL - len(active)
        shares = row[active].astype(np.float64) * (available / int(row.sum()))
        allocated = np.floor(shares).astype(np.int64) + 1
        remainder = RANS_TOTAL - int(allocated.sum())
        if remainder:
            order = np.argsort(shares - np.floor(shares))[::-1]
            allocated[order[:remainder]] += 1
        frequencies[position, active] = allocated.astype(np.uint16)
        cumulative[position] = (np.cumsum(frequencies[position], dtype=np.uint32) -
                                frequencies[position]).astype(np.uint16)
        for symbol in active:
            lo = int(cumulative[position, symbol])
            hi = lo + int(frequencies[position, symbol])
            inverse[position, lo:hi] = int(symbol)

    source = np.ascontiguousarray(raw, dtype=np.uint8)
    emitted = bytearray()
    state = RANS_LOWER_BOUND
    encode_start = time.perf_counter()
    for index in range(len(source) - 1, -1, -1):
        position = index % block_bytes
        symbol = int(source[index])
        frequency = int(frequencies[position, symbol])
        slot_base = int(cumulative[position, symbol])
        maximum = ((RANS_LOWER_BOUND >> RANS_SCALE_BITS) << 8) * frequency
        while state >= maximum:
            emitted.append(state & 0xFF)
            state >>= 8
        state = ((state // frequency) << RANS_SCALE_BITS) + (state % frequency) + slot_base
    encoded = state.to_bytes(4, "little") + bytes(reversed(emitted))
    encode_seconds = time.perf_counter() - encode_start

    decoded = np.empty_like(source)
    state = int.from_bytes(encoded[:4], "little")
    cursor = 4
    mask = RANS_TOTAL - 1
    decode_start = time.perf_counter()
    for index in range(len(source)):
        position = index % block_bytes
        slot = state & mask
        symbol = int(inverse[position, slot])
        decoded[index] = symbol
        frequency = int(frequencies[position, symbol])
        slot_base = int(cumulative[position, symbol])
        state = frequency * (state >> RANS_SCALE_BITS) + (slot - slot_base)
        while state < RANS_LOWER_BOUND:
            if cursor >= len(encoded):
                raise RuntimeError("rANS decoder ran out of renormalization bytes")
            state = (state << 8) | encoded[cursor]
            cursor += 1
    decode_seconds = time.perf_counter() - decode_start
    if not np.array_equal(decoded, source) or cursor != len(encoded) or state != RANS_LOWER_BOUND:
        raise RuntimeError("block-position rANS round-trip failed")
    return len(encoded), encode_seconds, decode_seconds


def analyze(
    ref: TensorRef, sample_budget: int, zstd_level: int, tile_bytes: int, run_rans: bool
) -> dict[str, object]:
    tensor = ref.tensor
    size = int(tensor.n_bytes)
    _block_elements, block_bytes = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
    # Align samples both to IQ blocks and palette tiles so estimates don't
    # mistake sampling-window tails for saved bytes.
    ranges = sample_ranges(size, sample_budget, math.lcm(block_bytes, tile_bytes))
    parts = list(byte_views(tensor.data, ranges))
    sampled = sum(len(part) for part in parts)

    counts = np.zeros(256, dtype=np.uint64)
    position_counts = np.zeros((block_bytes, 256), dtype=np.uint64)
    zero_count = 0
    for part in parts:
        hist = np.bincount(np.frombuffer(part, dtype=np.uint8), minlength=256)
        counts += hist.astype(np.uint64, copy=False)
        zero_count += int(hist[0])
        block_rows = np.frombuffer(part, dtype=np.uint8).reshape(-1, block_bytes)
        for position in range(block_bytes):
            pos_hist = np.bincount(block_rows[:, position], minlength=256)
            position_counts[position] += pos_hist.astype(np.uint64, copy=False)
    probabilities = counts[counts != 0] / sampled
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    pos_totals = position_counts.sum(axis=1, keepdims=True)
    pos_probabilities = np.divide(position_counts, pos_totals, out=np.zeros_like(position_counts, dtype=np.float64), where=pos_totals != 0)
    position_entropies = -np.where(pos_probabilities > 0, pos_probabilities * np.log2(np.maximum(pos_probabilities, 1e-300)), 0).sum(axis=1)
    block_position_entropy = float(position_entropies.mean())
    rans_bytes = None
    rans_model_bytes = block_bytes * 256 * 2
    rans_encode_seconds = None
    rans_decode_seconds = None
    if run_rans:
        sample_data = np.concatenate([np.frombuffer(part, dtype=np.uint8) for part in parts])
        rans_bytes, rans_encode_seconds, rans_decode_seconds = rans_roundtrip(
            sample_data, position_counts
        )

    zstd_bytes = zstd_compress_size(parts, zstd_level)
    shuffled_parts = list(bitshuffle(parts))
    for original, shuffled in zip(parts, shuffled_parts, strict=True):
        restored = bitunshuffle(shuffled, len(original))
        if not np.array_equal(restored, np.frombuffer(original, dtype=np.uint8)):
            raise RuntimeError(f"bitshuffle round-trip failed for {tensor.name}")
    bitshuffle_bytes = zstd_compress_size(shuffled_parts, zstd_level)
    palette_bytes, palette_tiles, raw_tiles = palette_estimate(parts, tile_bytes)
    sparse_bytes = (sampled + 7) // 8 + sampled - zero_count

    return {
        "tensor": tensor.name,
        "path": ref.path,
        "type": tensor.tensor_type.name,
        "shape": [int(x) for x in tensor.shape],
        "file_bytes": size,
        "sample_bytes": sampled,
        "h0_bits_per_byte": entropy,
        "h0_reduction_percent": max(0.0, (1.0 - entropy / 8.0) * 100.0),
        "block_bytes": block_bytes,
        "block_position_h0_bits_per_byte": block_position_entropy,
        "block_position_h0_reduction_percent": max(0.0, (1.0 - block_position_entropy / 8.0) * 100.0),
        "rans_payload_bytes": rans_bytes,
        "rans_model_bytes": rans_model_bytes if run_rans else None,
        "rans_total_reduction_percent": (
            (1.0 - (int(rans_bytes) + rans_model_bytes) / sampled) * 100.0
            if rans_bytes is not None else None
        ),
        "rans_encode_seconds": rans_encode_seconds,
        "rans_decode_seconds": rans_decode_seconds,
        "zero_percent": zero_count / sampled * 100.0,
        "sparse_mask_estimated_bytes": sparse_bytes,
        "sparse_mask_reduction_percent": (1.0 - sparse_bytes / sampled) * 100.0,
        "zstd_bytes": zstd_bytes,
        "zstd_reduction_percent": (1.0 - zstd_bytes / sampled) * 100.0,
        "bitshuffle_zstd_bytes": bitshuffle_bytes,
        "bitshuffle_zstd_reduction_percent": (1.0 - bitshuffle_bytes / sampled) * 100.0,
        "palette_estimated_bytes": palette_bytes,
        "palette_reduction_percent": (1.0 - palette_bytes / sampled) * 100.0,
        "palette_tiles": palette_tiles,
        "raw_fallback_tiles": raw_tiles,
        "palette_tile_bytes": tile_bytes,
    }


def percent(value: float) -> str:
    # The zstd frame header adds only a few bytes to a multi-megabyte sample;
    # avoid printing that as a misleading negative zero.
    return "0.00%" if abs(value) < 0.005 else f"{value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", help="GGUF shard path or directory of shards")
    parser.add_argument(
        "--layer", type=int, action="append", dest="layers",
        help="audit only this layer (repeatable; default chooses earliest MoE layer)",
    )
    parser.add_argument("--max-tensors", type=int, default=3)
    parser.add_argument("--sample-mib", type=int, default=32,
                        help="bytes sampled per tensor, spread across the tensor (default: 32)")
    parser.add_argument("--full-tensor", action="store_true",
                        help="scan each selected tensor completely instead of sampling")
    parser.add_argument("--rans", action="store_true",
                        help="run a CPU block-position rANS encode/decode round-trip")
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--tile-bytes", type=int, default=DEFAULT_TILE_BYTES,
                        help="tile width for the simple palette estimate (default: 256)")
    parser.add_argument("--json", dest="json_path", help="also write results as JSON")
    args = parser.parse_args()
    if args.max_tensors < 1 or (args.sample_mib < 1 and not args.full_tensor) or args.tile_bytes < 2:
        parser.error("--max-tensors, --sample-mib, and --tile-bytes must be positive")

    try:
        refs = discover_tensors(model_files(args.gguf))
        selected = choose_tensors(refs, set(args.layers) if args.layers else None, args.max_tensors)
        rows = [analyze(
            ref,
            int(ref.tensor.n_bytes) if args.full_tensor else args.sample_mib * 1024 * 1024,
            args.zstd_level,
            args.tile_bytes,
            args.rans,
        ) for ref in selected]
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Lossless quantized MoE byte audit (offline sample; no GPU work)")
    print("Tensor | Type | file MiB | sample MiB | H0 reduction | position-H0 | Zstd | bitshuffle+Zstd | sparse mask | tile palette")
    for row in rows:
        print(
            f"{row['tensor']} | {row['type']} | {row['file_bytes'] / 2**20:.1f} | "
            f"{row['sample_bytes'] / 2**20:.1f} | {percent(float(row['h0_reduction_percent']))} | "
            f"{percent(float(row['block_position_h0_reduction_percent']))} | "
            f"{percent(float(row['zstd_reduction_percent']))} | "
            f"{percent(float(row['bitshuffle_zstd_reduction_percent']))} | "
            f"{percent(float(row['sparse_mask_reduction_percent']))} | "
            f"{percent(float(row['palette_reduction_percent']))}"
        )
    total_sample = sum(int(row["sample_bytes"]) for row in rows)
    weighted_h0 = sum(float(row["h0_bits_per_byte"]) * int(row["sample_bytes"]) for row in rows) / total_sample
    weighted_position_h0 = sum(
        float(row["block_position_h0_bits_per_byte"]) * int(row["sample_bytes"])
        for row in rows
    ) / total_sample
    total_zstd = sum(int(row["zstd_bytes"]) for row in rows)
    h0_reduction = max(0.0, (1.0 - weighted_h0 / 8.0) * 100.0)
    position_h0_reduction = max(0.0, (1.0 - weighted_position_h0 / 8.0) * 100.0)
    zstd_reduction = (1.0 - total_zstd / total_sample) * 100.0
    verdict = "PASS" if h0_reduction > 15.0 else "FAIL" if h0_reduction < 5.0 else "INCONCLUSIVE"
    print(f"\nWeighted H0 byte-marginal estimate: {percent(h0_reduction)} reduction ({verdict} vs supplied H0 criterion)")
    print(f"Weighted block-position conditioned H0: {percent(position_h0_reduction)} estimated reduction")
    print(f"Measured Zstd-3 sample reduction: {percent(zstd_reduction)}")
    print("H0 is an IID byte estimate, not a guaranteed compression ratio. Palette and sparse sizes are codec estimates;")
    print("their decode/index overhead is not included. Bitshuffle+Zstd is an offline diagnostic, not a GPU codec.")
    if args.rans:
        total_rans = sum(int(row["rans_payload_bytes"]) + int(row["rans_model_bytes"]) for row in rows)
        rans_reduction = (1.0 - total_rans / total_sample) * 100.0
        total_rans_decode_s = sum(float(row["rans_decode_seconds"]) for row in rows)
        print(f"Block-position rANS payload + frequency tables: {percent(rans_reduction)} reduction (round-trip verified)")
        print(f"CPU rANS decode rate: {total_sample / max(total_rans_decode_s, 1e-12) / 2**20:.2f} MiB/s")
    if args.json_path:
        import json
        with open(args.json_path, "w", encoding="utf-8") as out:
            json.dump({"summary": {"h0_reduction_percent": h0_reduction,
                                   "block_position_h0_reduction_percent": position_h0_reduction,
                                   "zstd_reduction_percent": zstd_reduction,
                                   "verdict": verdict,
                                   "rans_requested": args.rans}, "tensors": rows}, out, indent=2)
            out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
