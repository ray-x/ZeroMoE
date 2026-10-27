#!/usr/bin/env python3
"""Build a 3-shard Qwen4Exp GGUF with the shared MTP sidecar as shard 3.

The main shards are cloned with XFS reflinks when possible; the source GGUFs
are never modified. The sidecar's tensor bytes are copied into the new shard.
Only GGUF metadata and shard headers are rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import sys
from dataclasses import dataclass

FICLONE = 0x40049409
SCALARS = {
    0: ("B", "u8"), 1: ("b", "i8"), 2: ("H", "u16"),
    3: ("h", "i16"), 4: ("I", "u32"), 5: ("i", "i32"),
    6: ("f", "f32"), 7: ("B", "bool"), 10: ("Q", "u64"),
    11: ("q", "i64"), 12: ("d", "f64"),
}


@dataclass
class Field:
    key: str
    type_id: int
    value: object
    raw: bytes


@dataclass
class GGUF:
    path: Path
    version: int
    n_tensors: int
    fields: list[Field]
    tensor_info: bytes
    data_start: int
    file_size: int
    tensor_names: list[str]

    @property
    def metadata(self) -> dict[str, object]:
        return {field.key: field.value for field in self.fields}


def read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("truncated GGUF header")
    return data


def read_u32(f) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_string(f) -> str:
    n = read_u64(f)
    return read_exact(f, n).decode("utf-8")


def skip_value(f, type_id: int):
    if type_id in SCALARS:
        fmt, _ = SCALARS[type_id]
        data = read_exact(f, struct.calcsize("<" + fmt))
        return struct.unpack("<" + fmt, data)[0]
    if type_id == 8:  # STRING
        return read_string(f)
    if type_id == 9:  # ARRAY
        element_type = read_u32(f)
        count = read_u64(f)
        if element_type == 8:
            for _ in range(count):
                read_string(f)
            return None
        if element_type not in SCALARS:
            raise ValueError(f"unsupported GGUF array element type {element_type}")
        fmt, _ = SCALARS[element_type]
        f.seek(struct.calcsize("<" + fmt) * count, 1)
        return None
    raise ValueError(f"unsupported GGUF metadata type {type_id}")


def read_header(path: Path) -> GGUF:
    path = path.resolve()
    with path.open("rb") as f:
        magic = read_exact(f, 4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = read_u32(f)
        if version != 3:
            raise ValueError(f"{path}: expected GGUF v3, found v{version}")
        n_tensors = read_u64(f)
        n_fields = read_u64(f)
        fields = []
        for _ in range(n_fields):
            start = f.tell()
            key = read_string(f)
            type_id = read_u32(f)
            value = skip_value(f, type_id)
            end = f.tell()
            f.seek(start)
            raw = read_exact(f, end - start)
            fields.append(Field(key, type_id, value, raw))
        kv_end = f.tell()
        tensor_names = []
        for _ in range(n_tensors):
            tensor_names.append(read_string(f))
            n_dims = read_u32(f)
            f.seek(8 * n_dims, 1)
            f.seek(12, 1)  # tensor type and relative data offset
        tensor_end = f.tell()
        metadata = {field.key: field.value for field in fields}
        alignment = int(metadata.get("general.alignment", 32))
        data_start = (tensor_end + alignment - 1) // alignment * alignment
        f.seek(kv_end)
        tensor_info = read_exact(f, tensor_end - kv_end)
    return GGUF(path, version, n_tensors, fields, tensor_info, data_start,
                path.stat().st_size, tensor_names)


def encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def encode_field(key: str, type_id: int, value: object) -> bytes:
    prefix = encode_string(key) + struct.pack("<I", type_id)
    if type_id in SCALARS:
        fmt, _ = SCALARS[type_id]
        return prefix + struct.pack("<" + fmt, value)
    if type_id == 8:
        return prefix + encode_string(str(value))
    raise ValueError(f"cannot encode GGUF type {type_id}")


def array_meta(field: Field) -> tuple[int, int, int] | None:
    raw = field.raw
    key_size = struct.unpack_from("<Q", raw, 0)[0]
    type_offset = 8 + key_size
    metadata_type = struct.unpack_from("<I", raw, type_offset)[0]
    if metadata_type != 9:
        return None
    element_type = struct.unpack_from("<I", raw, type_offset + 4)[0]
    count = struct.unpack_from("<Q", raw, type_offset + 8)[0]
    return element_type, count, type_offset + 16


def array_values(field: Field) -> tuple[int, tuple[object, ...]]:
    info = array_meta(field)
    if info is None:
        raise ValueError(f"{field.key}: expected GGUF array metadata")
    element_type, count, offset = info
    if element_type == 8:
        values = []
        for _ in range(count):
            size = struct.unpack_from("<Q", field.raw, offset)[0]
            offset += 8
            values.append(field.raw[offset:offset + size].decode("utf-8"))
            offset += size
        if offset != len(field.raw):
            raise ValueError(f"{field.key}: trailing bytes in GGUF string array")
        return element_type, tuple(values)
    if element_type not in SCALARS:
        raise ValueError(f"{field.key}: unsupported GGUF array element type {element_type}")
    fmt, _ = SCALARS[element_type]
    expected = offset + struct.calcsize("<" + fmt) * count
    if expected != len(field.raw):
        raise ValueError(f"{field.key}: malformed GGUF scalar array")
    values = struct.unpack_from("<" + fmt * count, field.raw, offset) if count else ()
    return element_type, tuple(values)


def layer_array_replacements(main: GGUF, sidecar: GGUF,
                             trunk_blocks: int, total_blocks: int) -> dict[str, bytes]:
    side_fields = {field.key: field for field in sidecar.fields}
    replacements = {}
    for field in main.fields:
        info = array_meta(field)
        if not field.key.startswith("qwen4exp.") or info is None or info[1] != trunk_blocks:
            continue
        other = side_fields.get(field.key)
        if other is None or array_meta(other) is None:
            raise ValueError(f"{field.key}: sidecar is missing the extended layer array")
        other_info = array_meta(other)
        if other_info[0] != info[0] or other_info[1] != total_blocks:
            raise ValueError(f"{field.key}: expected sidecar array length {total_blocks}")
        main_type, main_values = array_values(field)
        side_type, side_values = array_values(other)
        if main_type != side_type or main_values != side_values[:trunk_blocks]:
            raise ValueError(f"{field.key}: sidecar's trunk-layer prefix differs from main metadata")
        replacements[field.key] = other.raw
    return replacements


def rebuild_header(info: GGUF, updates: dict[str, tuple[int, object]],
                   additions: dict[str, tuple[int, object]],
                   raw_replacements: dict[str, bytes] | None = None,
                   preserve_data_start: int | None = None) -> tuple[bytes, int]:
    raw_replacements = raw_replacements or {}
    known = {field.key: field for field in info.fields}
    missing_updates = (set(updates) | set(raw_replacements)) - set(known)
    if missing_updates:
        raise ValueError(f"missing expected metadata key(s) in {info.path}: {sorted(missing_updates)}")
    if set(updates) & set(raw_replacements):
        raise ValueError("a GGUF field cannot have both a scalar update and raw replacement")
    records = []
    for field in info.fields:
        if field.key in raw_replacements:
            records.append(raw_replacements[field.key])
        elif field.key in updates:
            type_id, value = updates[field.key]
            if type_id != field.type_id:
                raise ValueError(f"{field.key}: metadata type changed unexpectedly")
            records.append(encode_field(field.key, type_id, value))
        else:
            records.append(field.raw)
    for key, (type_id, value) in additions.items():
        if key in known:
            raise ValueError(f"refusing duplicate GGUF metadata key: {key}")
        records.append(encode_field(key, type_id, value))
    prefix = struct.pack("<4sIQQ", b"GGUF", info.version, info.n_tensors,
                         len(records))
    header_body = prefix + b"".join(records) + info.tensor_info
    alignment = int(info.metadata.get("general.alignment", 32))
    data_start = (len(header_body) + alignment - 1) // alignment * alignment
    if preserve_data_start is not None:
        if len(header_body) > preserve_data_start:
            raise ValueError(f"{info.path}: updated metadata exceeds the original tensor data start")
        data_start = preserve_data_start
    return header_body + bytes(data_start - len(header_body)), data_start


def validate_source(main1: GGUF, main2: GGUF, sidecar: GGUF) -> tuple[int, int]:
    m1, m2, mtp = main1.metadata, main2.metadata, sidecar.metadata
    if m1.get("general.architecture") != "qwen4exp" or mtp.get("general.architecture") != "qwen4exp":
        raise ValueError("target and sidecar must both use general.architecture=qwen4exp")
    if int(m1.get("qwen4exp.block_count", 0)) != 48:
        raise ValueError("expected 48 trunk layers in the Q2_0 main shards")
    if int(mtp.get("qwen4exp.block_count", 0)) != 49 or int(mtp.get("qwen4exp.nextn_predict_layers", 0)) != 1:
        raise ValueError("sidecar metadata must declare block_count=49 and nextn_predict_layers=1")
    if (int(m1.get("split.no", -1)), int(m2.get("split.no", -1))) != (0, 1):
        raise ValueError("main shards must be numbered 0 and 1")
    if int(m1.get("split.count", 0)) != 2 or int(m2.get("split.count", 0)) != 2:
        raise ValueError("main shards must declare split.count=2")
    if int(m1.get("split.tensors.count", -1)) != main1.n_tensors + main2.n_tensors:
        raise ValueError("main split.tensors.count does not equal its shard tensor total")
    names = set(main1.tensor_names) | set(main2.tensor_names)
    if len(names) != main1.n_tensors + main2.n_tensors:
        raise ValueError("main shards contain duplicate tensor names")
    side_names = set(sidecar.tensor_names)
    if names & side_names:
        raise ValueError(f"sidecar tensors duplicate target names: {sorted(names & side_names)[:5]}")
    if len(side_names) != sidecar.n_tensors or not all(n.startswith("blk.48.") for n in side_names):
        raise ValueError("expected 32 unique sidecar tensors in blk.48")
    total = main1.n_tensors + main2.n_tensors + sidecar.n_tensors
    return total, int(m1["qwen4exp.block_count"]) + 1


def copy_main_shard(source: GGUF, output: Path, header: bytes,
                    allow_copy: bool) -> str:
    if len(header) != source.data_start:
        raise ValueError(f"{source.path}: metadata update shifts data start; refusing main-shard copy")
    with source.path.open("rb") as src, output.open("xb+") as dst:
        try:
            fcntl.ioctl(dst.fileno(), FICLONE, src.fileno())
            mode = "reflink"
        except OSError as error:
            if not allow_copy:
                raise RuntimeError(f"reflink failed for {source.path} ({error}); "
                                   "use --allow-copy-main to stream-copy the large shard") from error
            dst.seek(0)
            dst.truncate(0)
            src.seek(0)
            shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
            mode = "full-copy"
        dst.seek(0)
        dst.write(header)
        dst.flush()
        os.fsync(dst.fileno())
    return mode


def copy_repacked(source: GGUF, output: Path, header: bytes, data_start: int) -> None:
    with source.path.open("rb") as src, output.open("xb") as dst:
        dst.write(header)
        src.seek(source.data_start)
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    expected = data_start + source.file_size - source.data_start
    if output.stat().st_size != expected:
        raise IOError(f"{output}: copied size {output.stat().st_size}, expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="directory containing the two Q2_0 split shards")
    parser.add_argument("--sidecar", type=Path, required=True,
                        help="shared Q4_K_M MTP GGUF")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new output directory; must not already exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-copy-main", action="store_true",
                        help="allow full streaming copies if reflink is unavailable")
    args = parser.parse_args()
    try:
        base_dir = args.base_dir.resolve()
        sidecar_path = args.sidecar.resolve()
        out_dir = args.out_dir.resolve()
        if out_dir.exists():
            raise ValueError(f"refusing to overwrite output directory {out_dir}")
        shards1 = list(base_dir.glob("*-00001-of-00002.gguf"))
        shards2 = list(base_dir.glob("*-00002-of-00002.gguf"))
        if len(shards1) != 1 or len(shards2) != 1:
            raise ValueError("base-dir must contain exactly one shard 1 and one shard 2")
        main1, main2, mtp = map(read_header, (shards1[0], shards2[0], sidecar_path))
        total_tensors, total_blocks = validate_source(main1, main2, mtp)
        layer_arrays = layer_array_replacements(
            main1, mtp, int(main1.metadata["qwen4exp.block_count"]), total_blocks)
        updates1 = {
            "qwen4exp.block_count": (4, total_blocks),
            "split.count": (2, 3),
            "split.tensors.count": (5, total_tensors),
            "general.description": (8, "Qwen3.8 Q2_0 MTP"),
        }
        add1 = {"qwen4exp.nextn_predict_layers": (4, 1)}
        updates2 = {"split.count": (2, 3), "split.tensors.count": (5, total_tensors)}
        add2 = {}
        updates3 = {}
        add3 = {"split.no": (2, 2), "split.count": (2, 3),
                "split.tensors.count": (5, total_tensors)}
        h1, d1 = rebuild_header(main1, updates1, add1, layer_arrays,
                                preserve_data_start=main1.data_start)
        h2, d2 = rebuild_header(main2, updates2, add2,
                                preserve_data_start=main2.data_start)
        h3, d3 = rebuild_header(mtp, updates3, add3)
        if d1 != main1.data_start or d2 != main2.data_start:
            raise ValueError("main shard header size changed; refusing to shift target tensor data")
        prefix = shards1[0].name.removesuffix("-00001-of-00002.gguf")
        outputs = [out_dir / f"{prefix}-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
        summary = {
            "main_shards": [str(main1.path), str(main2.path)],
            "mtp_sidecar": str(mtp.path),
            "output_shards": [str(x) for x in outputs],
            "architecture": "qwen4exp", "block_count": total_blocks,
            "nextn_predict_layers": 1, "split_count": 3,
            "split_tensor_count": total_tensors,
            "replaced_layer_arrays": sorted(layer_arrays),
            "shard_tensor_counts": [main1.n_tensors, main2.n_tensors, mtp.n_tensors],
            "data_starts": [d1, d2, d3],
            "new_sidecar_data_offset_delta": d3 - mtp.data_start,
            "copy_policy": "reflink main shards; stream-copy sidecar; --allow-copy-main enables full-copy fallback",
        }
        if args.dry_run:
            print(json.dumps(summary, indent=2))
            return 0
        out_dir.mkdir(parents=True)
        try:
            mode1 = copy_main_shard(main1, outputs[0], h1, args.allow_copy_main)
            mode2 = copy_main_shard(main2, outputs[1], h2, args.allow_copy_main)
            copy_repacked(mtp, outputs[2], h3, d3)
            summary["main_shard_copy_modes"] = [mode1, mode2]
            # Re-read headers and enforce split/model contract before publishing manifest.
            check = list(map(read_header, outputs))
            if [x.n_tensors for x in check] != [main1.n_tensors, main2.n_tensors, mtp.n_tensors]:
                raise ValueError("output shard tensor counts changed")
            for i, shard in enumerate(check):
                md = shard.metadata
                if int(md["split.no"]) != i or int(md["split.count"]) != 3 or int(md["split.tensors.count"]) != total_tensors:
                    raise ValueError(f"output shard {i} has inconsistent split metadata")
            md = check[0].metadata
            if int(md["qwen4exp.block_count"]) != total_blocks or int(md["qwen4exp.nextn_predict_layers"]) != 1:
                raise ValueError("output main metadata is not MTP-spliced")
            ratio_field = next((field for field in check[0].fields
                                if field.key == "qwen4exp.attention.compress_ratios"), None)
            ratio_info = array_meta(ratio_field) if ratio_field else None
            if not ratio_info or ratio_info[1] != total_blocks:
                raise ValueError("output compress_ratios metadata does not cover all 49 blocks")
            summary["output_bytes"] = [x.stat().st_size for x in outputs]
            (out_dir / "splice-manifest.json").write_text(json.dumps(summary, indent=2) + "\\n")
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, struct.error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
