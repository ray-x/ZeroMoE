#!/usr/bin/env python3
"""Build a pruned GGUF shard with selected high-precision boundary layers.

The tool copies GGUF bytes without decoding or requantizing them.  It accepts
one low-precision input shard and one already-pruned reference shard.  Layers
0 and 47 are copied wholesale from the reference.  Every other layer keeps
the low-precision tensors, but its routed expert planes and router columns
are reduced using the original-expert mask in an RCO allocation file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


ALIGNMENT_DEFAULT = 32
BOUNDARY_LAYERS = frozenset({0, 47})
ORIGINAL_EXPERTS = 512
RETAINED_EXPERTS = 256
CHUNK_BYTES = 8 * 1024 * 1024


class FormatError(RuntimeError):
    pass


@dataclass
class KV:
    key: str
    value_type: int
    raw_value: bytes


@dataclass
class Tensor:
    name: str
    shape: list[int]
    value_type: int
    relative_offset: int
    data_offset: int
    size_bytes: int


@dataclass
class Model:
    path: Path
    version: int
    kvs: list[KV]
    tensors: list[Tensor]
    alignment: int
    data_start: int
    file_size: int

    @property
    def tensors_by_name(self) -> dict[str, Tensor]:
        return {tensor.name: tensor for tensor in self.tensors}

    @property
    def kv_by_key(self) -> dict[str, KV]:
        return {kv.key: kv for kv in self.kvs}


@dataclass
class OutputTensor:
    name: str
    shape: list[int]
    value_type: int
    source_model: Model
    source_tensor: Tensor
    mode: str
    selected_indices: list[int] | None
    size_bytes: int
    output_offset: int = 0


FIXED_VALUE_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


TYPE_BLOCKS = {
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    34: (256, 54),
    35: (256, 66),
    39: (32, 17),
    40: (64, 36),
    42: (64, 18),
}


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise FormatError(f"alignment must be a positive power of two: {alignment}")
    return (value + alignment - 1) & ~(alignment - 1)


def read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise FormatError(f"truncated GGUF data: wanted {size} bytes, got {len(data)}")
    return data


def read_u32(stream: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(stream, 4))[0]


def read_u64(stream: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(stream, 8))[0]


def read_string(stream: BinaryIO) -> bytes:
    return read_exact(stream, read_u64(stream))


def read_value(stream: BinaryIO, value_type: int) -> bytes:
    if value_type in FIXED_VALUE_SIZES:
        return read_exact(stream, FIXED_VALUE_SIZES[value_type])
    if value_type == 8:
        length = read_u64(stream)
        return struct.pack("<Q", length) + read_exact(stream, length)
    if value_type == 9:
        element_type = read_u32(stream)
        count = read_u64(stream)
        raw = bytearray(struct.pack("<IQ", element_type, count))
        for _ in range(count):
            raw.extend(read_value(stream, element_type))
        return bytes(raw)
    raise FormatError(f"unsupported GGUF metadata value type {value_type}")


def scalar_uint(kv: KV) -> int | None:
    formats = {0: "<B", 2: "<H", 4: "<I", 10: "<Q"}
    fmt = formats.get(kv.value_type)
    if fmt is None:
        return None
    return struct.unpack(fmt, kv.raw_value)[0]


def replace_scalar_uint(kv: KV, value: int) -> None:
    formats = {0: "<B", 2: "<H", 4: "<I", 10: "<Q"}
    fmt = formats.get(kv.value_type)
    if fmt is None:
        raise FormatError(
            f"metadata key {kv.key} has unsupported integer type {kv.value_type}"
        )
    try:
        kv.raw_value = struct.pack(fmt, value)
    except struct.error as exc:
        raise FormatError(f"metadata value {value} does not fit {kv.key}") from exc


def config_kv(model: Model, suffix: str) -> KV | None:
    values = model.kv_by_key
    architecture = values.get("general.architecture")
    prefixes: list[str] = []
    if architecture is not None and architecture.value_type == 8:
        length = struct.unpack("<Q", architecture.raw_value[:8])[0]
        prefixes.append(architecture.raw_value[8:8 + length].decode("utf-8") + ".")
    prefixes.extend(
        [
            "qwen35moe.",
            "qwen3moe.",
            "qwen2moe.",
            "qwen.",
            "glm-dsa.",
            "glm.",
            "glm4.",
            "llama.",
            "deepseek2.",
            "",
        ]
    )
    for prefix in prefixes:
        candidate = values.get(prefix + suffix)
        if candidate is not None:
            return candidate
    return None


def tensor_size(shape: Iterable[int], value_type: int) -> int:
    numel = 1
    for dimension in shape:
        if dimension <= 0:
            raise FormatError(f"invalid tensor dimension {dimension}")
        numel *= dimension
    if value_type == 0:
        return numel * 4
    if value_type in {1, 30}:
        return numel * 2
    if value_type in {26}:
        return numel * 4
    if value_type in {27}:
        return numel * 8
    block = TYPE_BLOCKS.get(value_type)
    if block is None:
        raise FormatError(f"cannot calculate size for GGML tensor type {value_type}")
    block_elements, block_bytes = block
    if numel % block_elements:
        raise FormatError(
            f"tensor with {numel} elements is not divisible by block size "
            f"{block_elements} for type {value_type}"
        )
    return (numel // block_elements) * block_bytes


def parse_model(path: Path) -> Model:
    if not path.is_file():
        raise FormatError(f"GGUF input is not a file: {path}")
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        if read_exact(stream, 4) != b"GGUF":
            raise FormatError(f"not a GGUF file: {path}")
        version = read_u32(stream)
        tensor_count = read_u64(stream)
        kv_count = read_u64(stream)
        kvs: list[KV] = []
        for _ in range(kv_count):
            key = read_string(stream).decode("utf-8")
            value_type = read_u32(stream)
            kvs.append(KV(key, value_type, read_value(stream, value_type)))

        tensors: list[Tensor] = []
        for _ in range(tensor_count):
            name = read_string(stream).decode("utf-8")
            n_dimensions = read_u32(stream)
            shape = [read_u64(stream) for _ in range(n_dimensions)]
            value_type = read_u32(stream)
            relative_offset = read_u64(stream)
            tensors.append(
                Tensor(
                    name=name,
                    shape=shape,
                    value_type=value_type,
                    relative_offset=relative_offset,
                    data_offset=0,
                    size_bytes=tensor_size(shape, value_type),
                )
            )

        alignment = ALIGNMENT_DEFAULT
        alignment_kv = next((kv for kv in kvs if kv.key == "general.alignment"), None)
        if alignment_kv is not None:
            alignment = scalar_uint(alignment_kv) or 0
        if alignment <= 0 or alignment & (alignment - 1):
            raise FormatError(f"invalid GGUF alignment {alignment} in {path}")
        data_start = align_up(stream.tell(), alignment)
        for tensor in tensors:
            tensor.data_offset = data_start + tensor.relative_offset
            if tensor.data_offset < data_start or tensor.data_offset + tensor.size_bytes > file_size:
                raise FormatError(
                    f"tensor {tensor.name} exceeds file bounds in {path}: "
                    f"offset={tensor.data_offset} size={tensor.size_bytes} "
                    f"file={file_size}"
                )

    return Model(path, version, kvs, tensors, alignment, data_start, file_size)


def write_string(stream: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)))
    stream.write(encoded)


def write_kv(stream: BinaryIO, kv: KV) -> None:
    write_string(stream, kv.key)
    stream.write(struct.pack("<I", kv.value_type))
    stream.write(kv.raw_value)


def write_tensor_info(stream: BinaryIO, tensor: OutputTensor) -> None:
    write_string(stream, tensor.name)
    stream.write(struct.pack("<I", len(tensor.shape)))
    stream.write(struct.pack("<" + "Q" * len(tensor.shape), *tensor.shape))
    stream.write(struct.pack("<I", tensor.value_type))
    stream.write(struct.pack("<Q", tensor.output_offset))


def parse_allocation(path: Path) -> dict[int, list[int]]:
    masks: dict[int, list[int]] = {}
    pattern = re.compile(
        r"^blk\.(\d+)\.ffn_gate_exps\.weight\[(\d+)\]:\s*(.*)$"
    )
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            match = pattern.match(raw_line.rstrip("\n"))
            if match is None:
                continue
            layer = int(match.group(1))
            expert = int(match.group(2))
            quant_type = match.group(3).strip()
            if quant_type:
                masks.setdefault(layer, []).append(expert)

    for layer in range(48):
        indices = masks.get(layer, [])
        if len(indices) != RETAINED_EXPERTS:
            raise FormatError(
                f"allocation mask layer {layer} retains {len(indices)} experts, "
                f"expected {RETAINED_EXPERTS}"
            )
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise FormatError(f"allocation mask layer {layer} is not sorted/unique")
        if indices[0] < 0 or indices[-1] >= ORIGINAL_EXPERTS:
            raise FormatError(f"allocation mask layer {layer} has an out-of-range expert")
    unexpected_layers = sorted(set(masks) - set(range(48)))
    if unexpected_layers:
        raise FormatError(f"allocation contains unexpected layers: {unexpected_layers}")
    return masks


def layer_from_name(name: str) -> int | None:
    match = re.match(r"^blk\.(\d+)\.", name)
    return int(match.group(1)) if match else None


PACKED_EXPERT_RE = re.compile(
    r"^blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight$"
)
ROUTER_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")


def validate_inputs(low: Model, high: Model, masks: dict[int, list[int]]) -> None:
    low_by_name = low.tensors_by_name
    high_by_name = high.tensors_by_name
    if set(low_by_name) != set(high_by_name):
        only_low = sorted(set(low_by_name) - set(high_by_name))
        only_high = sorted(set(high_by_name) - set(low_by_name))
        raise FormatError(
            "low and boundary GGUF shards have different tensor names: "
            f"only-low={only_low[:5]} only-high={only_high[:5]}"
        )
    for layer in BOUNDARY_LAYERS:
        names = [tensor.name for tensor in low.tensors if layer_from_name(tensor.name) == layer]
        if not names:
            raise FormatError(f"boundary layer blk.{layer} is absent from the low shard")
        for name in names:
            low_tensor = low_by_name[name]
            high_tensor = high_by_name[name]
            if high_tensor.shape[-1:] != [RETAINED_EXPERTS] and (
                name.endswith("ffn_gate_inp.weight")
                or PACKED_EXPERT_RE.match(name) is not None
            ):
                raise FormatError(
                    f"boundary tensor {name} has unexpected high shape {high_tensor.shape}"
                )

    for layer, indices in masks.items():
        for name in (
            f"blk.{layer}.ffn_gate_exps.weight",
            f"blk.{layer}.ffn_up_exps.weight",
            f"blk.{layer}.ffn_down_exps.weight",
            f"blk.{layer}.ffn_gate_inp.weight",
        ):
            tensor = low_by_name.get(name)
            if tensor is None:
                raise FormatError(f"low shard is missing required tensor {name}")
            if not tensor.shape or tensor.shape[-1] != ORIGINAL_EXPERTS:
                raise FormatError(
                    f"low tensor {name} does not have {ORIGINAL_EXPERTS} in its last dimension: "
                    f"{tensor.shape}"
                )
            if tensor.size_bytes % ORIGINAL_EXPERTS:
                raise FormatError(f"low tensor {name} is not evenly divisible by expert count")
            if name.endswith("ffn_gate_inp.weight") and tensor.value_type != 30:
                raise FormatError(
                    f"router {name} is type {tensor.value_type}, expected BF16 type 30"
                )

    expert_count = config_kv(low, "expert_count")
    if expert_count is None or scalar_uint(expert_count) != ORIGINAL_EXPERTS:
        actual = scalar_uint(expert_count) if expert_count is not None else None
        raise FormatError(f"low metadata expert_count={actual}, expected {ORIGINAL_EXPERTS}")
    high_expert_count = config_kv(high, "expert_count")
    if high_expert_count is None or scalar_uint(high_expert_count) != RETAINED_EXPERTS:
        actual = scalar_uint(high_expert_count) if high_expert_count is not None else None
        raise FormatError(
            f"boundary metadata expert_count={actual}, expected {RETAINED_EXPERTS}"
        )


def build_plan(low: Model, high: Model, masks: dict[int, list[int]]) -> list[OutputTensor]:
    low_by_name = low.tensors_by_name
    high_by_name = high.tensors_by_name
    plan: list[OutputTensor] = []
    for low_tensor in low.tensors:
        name = low_tensor.name
        layer = layer_from_name(name)
        source_model = low
        source_tensor = low_tensor
        shape = list(low_tensor.shape)
        mode = "copy"
        selected_indices: list[int] | None = None

        if layer in BOUNDARY_LAYERS:
            source_model = high
            source_tensor = high_by_name[name]
            shape = list(source_tensor.shape)
            mode = "boundary"
        elif layer is not None:
            expert_match = PACKED_EXPERT_RE.match(name)
            router_match = ROUTER_RE.match(name)
            if expert_match is not None or router_match is not None:
                selected_indices = masks[layer]
                shape[-1] = RETAINED_EXPERTS
                mode = "select"

        size_bytes = tensor_size(shape, source_tensor.value_type)
        if mode == "boundary" and size_bytes != source_tensor.size_bytes:
            raise FormatError(f"boundary size calculation mismatch for {name}")
        if mode == "copy" and size_bytes != source_tensor.size_bytes:
            raise FormatError(f"copy size calculation mismatch for {name}")
        if mode == "select":
            if source_tensor.size_bytes % ORIGINAL_EXPERTS:
                raise FormatError(f"cannot select expert planes from {name}")
            plane_bytes = source_tensor.size_bytes // ORIGINAL_EXPERTS
            if size_bytes != plane_bytes * RETAINED_EXPERTS:
                raise FormatError(f"selected size calculation mismatch for {name}")

        plan.append(
            OutputTensor(
                name=name,
                shape=shape,
                value_type=source_tensor.value_type,
                source_model=source_model,
                source_tensor=source_tensor,
                mode=mode,
                selected_indices=selected_indices,
                size_bytes=size_bytes,
            )
        )

    return plan


def write_padding(stream: BinaryIO, count: int) -> None:
    if count <= 0:
        return
    zero_block = b"\0" * min(count, CHUNK_BYTES)
    while count:
        chunk = min(count, len(zero_block))
        stream.write(zero_block[:chunk])
        count -= chunk


def copy_bytes(source: BinaryIO, destination: BinaryIO, offset: int, size: int) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        data = source.read(min(remaining, CHUNK_BYTES))
        if not data:
            raise FormatError(f"unexpected EOF while copying at offset {offset}")
        destination.write(data)
        remaining -= len(data)


def copy_selected_planes(
    source: BinaryIO,
    destination: BinaryIO,
    source_offset: int,
    plane_bytes: int,
    selected_indices: list[int],
) -> None:
    for expert in selected_indices:
        copy_bytes(source, destination, source_offset + expert * plane_bytes, plane_bytes)


def write_output(
    output_path: Path,
    low: Model,
    high: Model,
    plan: list[OutputTensor],
    dry_run: bool,
) -> None:
    current_offset = 0
    for tensor in plan:
        tensor.output_offset = align_up(current_offset, low.alignment)
        current_offset = tensor.output_offset + tensor.size_bytes

    expert_count_kv = config_kv(low, "expert_count")
    if expert_count_kv is None:
        raise FormatError("low metadata does not contain an expert_count key")
    replace_scalar_uint(expert_count_kv, RETAINED_EXPERTS)
    header_size = 4 + 4 + 8 + 8
    header_size += sum(8 + len(kv.key.encode("utf-8")) + 4 + len(kv.raw_value) for kv in low.kvs)
    header_size += sum(
        8
        + len(tensor.name.encode("utf-8"))
        + 4
        + 8 * len(tensor.shape)
        + 4
        + 8
        for tensor in plan
    )
    data_start = align_up(header_size, low.alignment)
    output_size = data_start + current_offset

    print(f"[stitch] low shard: {low.path} ({low.file_size / 1e9:.2f} GB)")
    print(f"[stitch] output: {output_path} ({output_size / 1e9:.2f} GB)")
    print(f"[stitch] tensors: {len(plan)}, data alignment: {low.alignment}")
    print(
        "[stitch] modes: "
        f"boundary={sum(t.mode == 'boundary' for t in plan)}, "
        f"pruned={sum(t.mode == 'select' for t in plan)}, "
        f"copied={sum(t.mode == 'copy' for t in plan)}"
    )
    if dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".partial")
    if temporary_path.exists():
        raise FormatError(f"temporary output already exists; remove it before retrying: {temporary_path}")

    try:
        with low.path.open("rb") as low_stream, high.path.open("rb") as high_stream, temporary_path.open("wb") as output:
            output.write(b"GGUF")
            output.write(struct.pack("<IQQ", low.version, len(plan), len(low.kvs)))
            for kv in low.kvs:
                write_kv(output, kv)
            for tensor in plan:
                write_tensor_info(output, tensor)
            write_padding(output, data_start - output.tell())

            source_streams = {low.path: low_stream}
            source_streams[high.path] = high_stream
            for index, tensor in enumerate(plan, 1):
                write_padding(output, data_start + tensor.output_offset - output.tell())
                source = source_streams[tensor.source_model.path]
                source_tensor = tensor.source_tensor
                if tensor.mode in {"copy", "boundary"}:
                    copy_bytes(source, output, source_tensor.data_offset, tensor.size_bytes)
                else:
                    plane_bytes = source_tensor.size_bytes // ORIGINAL_EXPERTS
                    copy_selected_planes(
                        source,
                        output,
                        source_tensor.data_offset,
                        plane_bytes,
                        tensor.selected_indices or [],
                    )
                if index % 100 == 0 or index == len(plan):
                    print(f"[stitch] wrote {index}/{len(plan)} tensors", flush=True)
            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def hash_range(stream: BinaryIO, offset: int, size: int) -> bytes:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = size
    while remaining:
        data = stream.read(min(remaining, CHUNK_BYTES))
        if not data:
            raise FormatError(f"unexpected EOF while hashing at offset {offset}")
        digest.update(data)
        remaining -= len(data)
    return digest.digest()


def hash_selected_planes(
    stream: BinaryIO,
    offset: int,
    plane_bytes: int,
    selected_indices: list[int],
) -> bytes:
    digest = hashlib.sha256()
    for expert in selected_indices:
        stream.seek(offset + expert * plane_bytes)
        remaining = plane_bytes
        while remaining:
            data = stream.read(min(remaining, CHUNK_BYTES))
            if not data:
                raise FormatError(f"unexpected EOF while hashing expert plane {expert}")
            digest.update(data)
            remaining -= len(data)
    return digest.digest()


def verify_output(output_path: Path, low: Model, high: Model, plan: list[OutputTensor]) -> None:
    output = parse_model(output_path)
    output_by_name = output.tensors_by_name
    if len(output.tensors) != len(plan):
        raise FormatError(f"output has {len(output.tensors)} tensors, expected {len(plan)}")
    expert_count = config_kv(output, "expert_count")
    if expert_count is None or scalar_uint(expert_count) != RETAINED_EXPERTS:
        actual = scalar_uint(expert_count) if expert_count is not None else None
        raise FormatError(f"output expert_count={actual}, expected {RETAINED_EXPERTS}")

    checked = 0
    with output.path.open("rb") as output_stream, low.path.open("rb") as low_stream, high.path.open("rb") as high_stream:
        source_streams = {low.path: low_stream, high.path: high_stream}
        for tensor in plan:
            actual = output_by_name.get(tensor.name)
            if actual is None:
                raise FormatError(f"output is missing tensor {tensor.name}")
            if actual.shape != tensor.shape or actual.value_type != tensor.value_type or actual.size_bytes != tensor.size_bytes:
                raise FormatError(
                    f"output geometry mismatch for {tensor.name}: "
                    f"got type={actual.value_type} shape={actual.shape} bytes={actual.size_bytes}, "
                    f"expected type={tensor.value_type} shape={tensor.shape} bytes={tensor.size_bytes}"
                )
            source = source_streams[tensor.source_model.path]
            if tensor.mode in {"copy", "boundary"}:
                expected_hash = hash_range(source, tensor.source_tensor.data_offset, tensor.size_bytes)
            else:
                plane_bytes = tensor.source_tensor.size_bytes // ORIGINAL_EXPERTS
                expected_hash = hash_selected_planes(
                    source,
                    tensor.source_tensor.data_offset,
                    plane_bytes,
                    tensor.selected_indices or [],
                )
            actual_hash = hash_range(output_stream, actual.data_offset, actual.size_bytes)
            if actual_hash != expected_hash:
                raise FormatError(f"output bytes mismatch for {tensor.name}")
            checked += 1
            if checked % 100 == 0 or checked == len(plan):
                print(f"[stitch] verified {checked}/{len(plan)} tensors", flush=True)
    print(f"[stitch] exact byte verification passed for {checked} tensors")


def companion_path(output_path: Path) -> Path:
    match = re.search(r"-00001-of-(\d+)\.gguf$", output_path.name)
    if match is None:
        raise FormatError(
            "an output using a split companion must end in -00001-of-NNNNN.gguf"
        )
    return output_path.with_name(
        output_path.name[: match.start()]
        + f"-00002-of-{match.group(1)}.gguf"
    )


def link_companion(output_path: Path, source_path: Path) -> None:
    if not source_path.is_file():
        raise FormatError(f"companion source is not a file: {source_path}")
    destination = companion_path(output_path)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source_path.resolve():
            print(f"[stitch] companion already linked: {destination}")
            return
        raise FormatError(f"companion output already exists: {destination}")
    destination.symlink_to(source_path)
    print(f"[stitch] linked companion shard: {destination} -> {source_path}")


def inspect_model(model: Model) -> None:
    print(f"path={model.path}")
    print(f"version={model.version} tensors={len(model.tensors)} kvs={len(model.kvs)}")
    print(f"alignment={model.alignment} data_start={model.data_start} size={model.file_size}")
    for key in ("general.architecture", "block_count", "expert_count", "expert_used_count", "split.count", "split.no"):
        kv = model.kv_by_key.get(key)
        if kv is None:
            print(f"{key}=<absent>")
        elif kv.value_type == 8:
            length = struct.unpack("<Q", kv.raw_value[:8])[0]
            print(f"{key}={kv.raw_value[8:8 + length].decode('utf-8', errors='replace')}")
        else:
            print(f"{key}={scalar_uint(kv)}")
    for tensor in model.tensors:
        if tensor.name.startswith(("token_embd", "output")) or tensor.name.startswith("blk.0.") or tensor.name.startswith("blk.47."):
            print(f"{tensor.name} type={tensor.value_type} shape={tensor.shape} bytes={tensor.size_bytes}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a 256-expert Qwen GGUF shard from a 512-expert base."
    )
    parser.add_argument("--low", type=Path, required=True, help="512-expert IQ2_XS GGUF shard")
    parser.add_argument("--boundary", type=Path, required=True, help="256-expert IQ1_M GGUF shard")
    parser.add_argument("--allocation", type=Path, required=True, help="RCO allocation manifest")
    parser.add_argument("--output", type=Path, required=True, help="new output GGUF shard")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    parser.add_argument("--inspect", action="store_true", help="print input metadata and tensor summaries")
    parser.add_argument("--verify-output", action="store_true", help="hash-check an existing output against the planned sources")
    parser.add_argument("--companion-source", type=Path, help="existing second GGUF shard to link beside the output")
    return parser


def main(argv: list[str]) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        low = parse_model(args.low)
        high = parse_model(args.boundary)
        masks = parse_allocation(args.allocation)
        validate_inputs(low, high, masks)
        if args.inspect:
            inspect_model(low)
            inspect_model(high)
        plan = build_plan(low, high, masks)
        if args.verify_output:
            verify_output(args.output, low, high, plan)
        else:
            write_output(args.output, low, high, plan, args.dry_run)
            if args.companion_source is not None and not args.dry_run:
                link_companion(args.output, args.companion_source)
        return 0
    except (OSError, UnicodeError, FormatError) as exc:
        print(f"[stitch] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
