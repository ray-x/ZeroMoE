#!/usr/bin/env python3
"""Stamp or verify a ZeroMoE binary against its source worktree.

Examples:
  tools/build_provenance.py stamp --binary build/zeromoe --repo . \
      --build-dir build --ggml-root third_party/llama.cpp/ggml
  tools/build_provenance.py check --binary build/zeromoe --repo .

The manifest is adjacent to the binary by default. Use --manifest to keep a
worktree's manifest in a shared results directory without writing into that
worktree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BUILD_PATHS = (
    "CMakeLists.txt",
    "CMakePresets.json",
    "Makefile",
    "cmake",
    "core",
    "models",
    "backends",
    "memory",
    "kernels",
    "include",
    "patches",
    "scripts",
)


def run_git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def source_snapshot(repo: Path) -> dict:
    commit = run_git(repo, "rev-parse", "HEAD").decode().strip()
    pathspec = ("--", *BUILD_PATHS)
    status = run_git(repo, "status", "--porcelain=v1", "--untracked-files=all", "-z", *pathspec)
    diff = run_git(repo, "diff", "--binary", "HEAD", *pathspec)
    untracked = run_git(repo, "ls-files", "--others", "--exclude-standard", "-z", *pathspec)

    digest = hashlib.sha256()
    digest.update(commit.encode())
    digest.update(b"\0STATUS\0")
    digest.update(status)
    digest.update(b"\0DIFF\0")
    digest.update(diff)
    untracked_paths = [p.decode(errors="surrogateescape") for p in untracked.split(b"\0") if p]
    for name in sorted(untracked_paths):
        path = repo / name
        digest.update(b"\0UNTRACKED\0")
        digest.update(name.encode(errors="surrogateescape"))
        if path.is_file():
            digest.update(path.read_bytes())

    status_text = status.decode(errors="surrogateescape").replace("\0", "\n").strip()
    return {
        "source_commit": commit,
        "source_dirty": bool(status),
        "source_status": status_text.splitlines() if status_text else [],
        "source_snapshot_sha256": digest.hexdigest(),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmake_cache_values(build_dir: Path | None) -> dict[str, str]:
    if not build_dir:
        return {}
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return {}
    values = {}
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("//") or not line or "=" not in line or ":" not in line:
            continue
        left, value = line.split("=", 1)
        key = left.split(":", 1)[0]
        values[key] = value
    keys = (
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CUDA_ARCHITECTURES",
        "GGML_CUDA",
        "GGML_CUDA_GRAPHS",
        "GGML_OPENMP",
        "ZEROMOE_GGML_DEBUG_PATCHES",
        "ZEROMOE_GGML_ROOT",
        "ZEROMOE_GGML_PREBUILT_DIR",
    )
    return {key: values[key] for key in keys if key in values}


def ggml_base_from(root: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if not root:
        return "unknown"
    candidates = [root / "ggml-src.txt", root.parent / "ggml-src.txt", root.parent.parent / "ggml-src.txt"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        match = re.search(r"@\s*([0-9a-f]{7,40})", candidate.read_text(errors="replace"))
        if match:
            return match.group(1)
    git_dir = root if (root / ".git").exists() else None
    if git_dir:
        try:
            return run_git(git_dir, "rev-parse", "HEAD").decode().strip()
        except RuntimeError:
            pass
    return "unknown"


def patch_manifest(repo: Path, include_debug: bool) -> tuple[str, list[str]]:
    candidates = [repo / "patches" / "README.md"]
    directories = ["ggml", "ggml-host-node"]
    if include_debug:
        directories.append("debug")
    for dirname in directories:
        folder = repo / "patches" / dirname
        if folder.is_dir():
            candidates.extend(sorted(folder.rglob("*.patch")))
    files = [p for p in candidates if p.is_file()]
    h = hashlib.sha256()
    names = []
    for path in sorted(files):
        rel = path.relative_to(repo).as_posix()
        names.append(rel)
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest(), names


def manifest_path(binary: Path, override: Path | None) -> Path:
    return override if override else Path(str(binary) + ".build-info.json")


def stamp(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    repo = args.repo.resolve()
    if not binary.is_file():
        raise RuntimeError(f"binary does not exist: {binary}")
    snapshot = source_snapshot(repo)
    if args.expected_commit and snapshot["source_commit"] != args.expected_commit:
        raise RuntimeError(
            f"source HEAD changed during build: configured {args.expected_commit}, "
            f"now {snapshot['source_commit']}"
        )
    if (args.expected_source_snapshot_sha256 and
            snapshot["source_snapshot_sha256"] != args.expected_source_snapshot_sha256):
        raise RuntimeError("source worktree changed after CMake configure and before post-build stamping")
    cmake = cmake_cache_values(args.build_dir.resolve() if args.build_dir else None)
    debug_on = cmake.get("ZEROMOE_GGML_DEBUG_PATCHES", "OFF").upper() in {"ON", "TRUE", "1"}
    patch_hash, patch_files = patch_manifest(repo, debug_on)
    ggml_root = args.ggml_root.resolve() if args.ggml_root else None
    info = {
        "schema": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "repo_root": str(repo),
        **snapshot,
        "build_notes": args.build_note,
        "ggml_base": ggml_base_from(ggml_root, args.ggml_base),
        "ggml_root": str(ggml_root) if ggml_root else "unknown",
        "ggml_patch_manifest_sha256": patch_hash,
        "ggml_patch_files": patch_files,
        "cmake": cmake,
    }
    out = manifest_path(binary, args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    text_path = Path(str(binary) + ".build-info.txt")
    cmake_info = info["cmake"]
    text_lines = [
        f"source commit: {info['source_commit']}",
        f"source dirty: {info['source_dirty']}",
        f"source snapshot SHA256: {info['source_snapshot_sha256']}",
        f"binary SHA256: {info['binary_sha256']}",
        f"GGML base: {info['ggml_base']}",
        f"GGML patch manifest SHA256: {patch_hash}",
        f"GGML patch files: {', '.join(patch_files)}",
        f"CMake build type: {cmake_info.get('CMAKE_BUILD_TYPE', 'unknown')}",
        f"CUDA compiler: {cmake_info.get('CMAKE_CUDA_COMPILER', 'unknown')}",
        f"CUDA architectures: {cmake_info.get('CMAKE_CUDA_ARCHITECTURES', 'unknown')}",
    ]
    text_lines.extend(f"build note: {note}" for note in args.build_note)
    text_path.write_text("\n".join(text_lines) + "\n")
    print(f"wrote build manifest: {out}")
    print(f"source={info['source_commit']} dirty={info['source_dirty']} binary_sha256={info['binary_sha256']}")
    print(f"ggml_base={info['ggml_base']} patch_manifest_sha256={patch_hash}")
    return 0


def check(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    repo = args.repo.resolve()
    manifest = manifest_path(binary, args.manifest)
    if not manifest.is_file():
        raise RuntimeError(f"missing build manifest: {manifest}; rebuild or stamp the binary")
    info = json.loads(manifest.read_text())
    errors = []
    actual_binary_hash = sha256_file(binary)
    if info.get("binary_sha256") != actual_binary_hash:
        errors.append("binary SHA256 differs from its manifest")
    current = source_snapshot(repo)
    expected_commit = args.expected_commit or current["source_commit"]
    if info.get("source_commit") != expected_commit:
        errors.append(
            f"binary source commit {info.get('source_commit')} != expected {expected_commit}"
        )
    if current["source_commit"] != expected_commit:
        errors.append(f"repo HEAD {current['source_commit']} != expected {expected_commit}")
    if info.get("source_snapshot_sha256") != current["source_snapshot_sha256"]:
        errors.append("source worktree snapshot differs from the snapshot used to stamp the binary")
    if info.get("build_notes") and not args.allow_build_notes:
        errors.append("manifest records build workarounds; pass --allow-build-notes only for a scoped diagnostic")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: source={info['source_commit']} dirty={info['source_dirty']} "
        f"binary_sha256={actual_binary_hash} ggml_base={info['ggml_base']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = sub.add_parser("snapshot", help="print the current source snapshot identity as JSON")
    snapshot_parser.add_argument("--repo", type=Path, required=True)
    for name in ("stamp", "check"):
        p = sub.add_parser(name)
        p.add_argument("--binary", type=Path, required=True)
        p.add_argument("--repo", type=Path, required=True)
        p.add_argument("--manifest", type=Path)
        if name == "stamp":
            p.add_argument("--build-dir", type=Path)
            p.add_argument("--ggml-root", type=Path)
            p.add_argument("--ggml-base")
            p.add_argument("--expected-commit")
            p.add_argument("--expected-source-snapshot-sha256")
            p.add_argument("--build-note", action="append", default=[])
        else:
            p.add_argument("--expected-commit")
            p.add_argument("--allow-build-notes", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            print(json.dumps(source_snapshot(args.repo.resolve()), sort_keys=True))
            return 0
        return stamp(args) if args.command == "stamp" else check(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
