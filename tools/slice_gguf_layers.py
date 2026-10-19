#!/usr/bin/env python3
"""Copy the first N layers of a GGUF shard into a standalone diagnostic GGUF.

The input must contain every tensor needed by the selected layers. Tensor data
is copied byte for byte; this tool does not decode or requantize weights.
"""

import argparse
import re
import struct
from dataclasses import dataclass
from pathlib import Path


SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                7: 1, 10: 8, 11: 8, 12: 8}
CHUNK = 8 * 1024 * 1024


def read_u(file, width):
    data = file.read(width)
    if len(data) != width:
        raise ValueError("truncated GGUF")
    return int.from_bytes(data, "little")


def read_string(file):
    return file.read(read_u(file, 8)).decode("utf-8")


def pack_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def skip_value(file, kind):
    if kind == 8:
        file.seek(read_u(file, 8), 1)
    elif kind == 9:
        child, count = read_u(file, 4), read_u(file, 8)
        if child in SCALAR_SIZES:
            file.seek(count * SCALAR_SIZES[child], 1)
        else:
            for _ in range(count):
                skip_value(file, child)
    elif kind in SCALAR_SIZES:
        file.seek(SCALAR_SIZES[kind], 1)
    else:
        raise ValueError(f"unsupported GGUF metadata type {kind}")


@dataclass
class Tensor:
    name: str
    dims: tuple[int, ...]
    kind: int
    offset: int
    size: int = 0


def read_header(file):
    if file.read(4) != b"GGUF":
        raise ValueError("not a GGUF")
    version = read_u(file, 4)
    if version != 3:
        raise ValueError(f"expected GGUF v3, got {version}")
    tensor_count, kv_count = read_u(file, 8), read_u(file, 8)
    fields = []
    for _ in range(kv_count):
        name = read_string(file)
        kind = read_u(file, 4)
        start = file.tell()
        skip_value(file, kind)
        end = file.tell()
        file.seek(start)
        raw = file.read(end - start)
        fields.append((name, kind, raw))
    tensors = []
    for _ in range(tensor_count):
        name = read_string(file)
        dims = tuple(read_u(file, 8) for _ in range(read_u(file, 4)))
        kind, offset = read_u(file, 4), read_u(file, 8)
        tensors.append(Tensor(name, dims, kind, offset))
    alignment = 32
    for name, kind, raw in fields:
        if name == "general.alignment":
            alignment = int.from_bytes(raw, "little")
    data_start = (file.tell() + alignment - 1) // alignment * alignment
    by_offset = sorted(tensors, key=lambda tensor: tensor.offset)
    file.seek(0, 2)
    data_size = file.tell() - data_start
    for index, tensor in enumerate(by_offset):
        end = by_offset[index + 1].offset if index + 1 < len(by_offset) else data_size
        tensor.size = end - tensor.offset
        if tensor.size <= 0:
            raise ValueError(f"invalid tensor extent: {tensor.name}")
    return fields, tensors, data_start, alignment


def sliced_field(name, kind, raw, layers):
    if name == "bailingmoe3.block_count":
        return layers.to_bytes(len(raw), "little")
    if name in {"bailingmoe3.attention.head_count_kv",
                "bailingmoe3.swiglu_clamp_exp",
                "bailingmoe3.swiglu_clamp_shexp"}:
        if kind != 9:
            raise ValueError(f"expected array metadata: {name}")
        child, count = struct.unpack_from("<IQ", raw)
        if count < layers or child not in SCALAR_SIZES:
            raise ValueError(f"cannot slice metadata: {name}")
        width = SCALAR_SIZES[child]
        return struct.pack("<IQ", child, layers) + raw[12:12 + layers * width]
    return raw


def write_slice(source, destination, layers):
    with source.open("rb") as inp:
        fields, tensors, data_start, alignment = read_header(inp)
        selected = []
        for tensor in tensors:
            match = re.match(r"blk\.(\d+)\.", tensor.name)
            if match is None or int(match.group(1)) < layers:
                selected.append(tensor)
        found = {int(match.group(1)) for tensor in selected
                 if (match := re.match(r"blk\.(\d+)\.", tensor.name))}
        if found != set(range(layers)):
            raise ValueError("the input shard does not contain all requested layers")
        fields = [(name, kind, sliced_field(name, kind, raw, layers))
                  for name, kind, raw in fields if not name.startswith("split.")]
        with destination.open("wb") as out:
            out.write(b"GGUF" + struct.pack("<IQQ", 3, len(selected), len(fields)))
            for name, kind, raw in fields:
                out.write(pack_string(name) + struct.pack("<I", kind) + raw)
            offset = 0
            for tensor in selected:
                offset = (offset + alignment - 1) // alignment * alignment
                out.write(pack_string(tensor.name))
                out.write(struct.pack("<I", len(tensor.dims)))
                for dim in tensor.dims:
                    out.write(struct.pack("<Q", dim))
                out.write(struct.pack("<IQ", tensor.kind, offset))
                offset += tensor.size
            out.write(bytes((-out.tell()) % alignment))
            for index, tensor in enumerate(selected):
                out.write(bytes((-out.tell()) % alignment))
                inp.seek(data_start + tensor.offset)
                remaining = tensor.size
                while remaining:
                    chunk = inp.read(min(CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"truncated tensor: {tensor.name}")
                    out.write(chunk)
                    remaining -= len(chunk)
                if index % 20 == 0:
                    print(f"{index + 1}/{len(selected)} tensors copied", flush=True)
    print(f"wrote {destination} ({destination.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--layers", type=int, default=8)
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("--layers must be positive")
    write_slice(args.source, args.destination, args.layers)


if __name__ == "__main__":
    main()
