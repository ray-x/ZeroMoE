#!/usr/bin/env python3
"""Inventory configuration names and source locations; never infer undocumented defaults."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("core", "memory", "models", "backends", "kernels")
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cu", ".cuh", ".mm"}
PATTERNS = {
    "environment": re.compile(r'"(ZEROMOE_[A-Z][A-Z0-9_]*)"'),
    "cli": re.compile(r'"(--[a-z][a-z0-9_-]*)"'),
    "toml": re.compile(r'\bget_(?:str|int|double|float|bool|int_array|type)\s*\(\s*"([a-z][a-z0-9_]*)"'),
    "graph_setting": re.compile(r'\{"([a-z][a-z0-9_]*)",\s*ZEROMOE_SETTINGS_REF'),
}


def inventory(root=ROOT):
    entries = defaultdict(set)
    for directory in SOURCE_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
                continue
            text = path.read_text()
            for kind, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    if kind == "environment" and match[1] in {
                        "ZEROMOE_LING_", "ZEROMOE_GLM5NEXT_"
                    }:
                        # Prefixes used by ZEROMOE_GLM_ALIAS are expanded
                        # into concrete environment names below.
                        continue
                    line = text.count("\n", 0, match.start()) + 1
                    entries[kind, match[1]].add(f"{path.relative_to(root)}:{line}")
            for match in re.finditer(r'ZEROMOE_GLM_ALIAS\("([A-Z0-9_]+)"\)', text):
                line = text.count("\n", 0, match.start()) + 1
                location = f"{path.relative_to(root)}:{line}"
                suffix = match[1]
                entries["environment", "ZEROMOE_GLM5NEXT_" + suffix].add(location)
            for number, line in enumerate(text.splitlines(), 1):
                if re.match(r"^\s*#\s*(?:if|ifdef|ifndef|elif)\b", line):
                    for name in re.findall(r"\b(?:ZEROMOE|GGML)_[A-Z0-9_]+", line.split("//", 1)[0]):
                        entries["macro", name].add(f"{path.relative_to(root)}:{number}")
    # Graph CLI flags are generated from the canonical graph-setting table,
    # rather than stored as individual string literals in the CLI parser.
    for (kind, name), locations in list(entries.items()):
        if kind == "graph_setting":
            entries["cli", "--graph-" + name.replace("_", "-")].update(locations)
    cmake_files = [root / "CMakeLists.txt", root / "server" / "CMakeLists.txt"]
    cmake_files.extend(sorted((root / "cmake").glob("*.cmake")))
    for path in cmake_files:
        if not path.is_file():
            continue
        text = path.read_text()
        for match in re.finditer(r"\boption\s*\(\s*(ZEROMOE_[A-Z0-9_]+)\b", text):
            line = text.count("\n", 0, match.start()) + 1
            entries["build_option", match[1]].add(f"{path.relative_to(root)}:{line}")
        for match in re.finditer(
            r"\bset\s*\(\s*(ZEROMOE_[A-Z0-9_]+)\b[^)]*?\bCACHE\s+(BOOL|STRING|PATH|FILEPATH|INTERNAL)\b",
            text,
        ):
            line = text.count("\n", 0, match.start()) + 1
            kind = "internal_build_cache" if match[2] == "INTERNAL" else "build_cache"
            entries[kind, match[1]].add(f"{path.relative_to(root)}:{line}")
    return [
        {"kind": kind, "name": name, "locations": sorted(locations)}
        for (kind, name), locations in sorted(entries.items())
    ]


def check_migrated_reads(root=ROOT):
    failures = []
    migrated = {
        "ZEROMOE_QWEN_CPU_THREADS",
        "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_LAYERS",
        "ZEROMOE_QWEN4EXP_GRAPH_PRELOAD_TRAILING",
    }
    for path in (root / "models" / "qwen").glob("*.[ch]pp"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if any(f'"{name}"' in line for name in migrated):
                failures.append(f"{path.relative_to(root)}:{number}: migrated setting bypasses typed resolver")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit names and exact source references as JSON")
    parser.add_argument("--check", action="store_true", help="Reject direct reads of migrated Qwen knobs")
    args = parser.parse_args()
    if args.check:
        failures = check_migrated_reads()
        if failures:
            parser.exit(1, "\n".join(failures) + "\n")
        print("Migrated Qwen settings use the typed resolver.")
        return
    entries = inventory()
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        counts = defaultdict(int)
        for entry in entries:
            counts[entry["kind"]] += 1
        for kind, count in sorted(counts.items()):
            print(f"{kind}: {count}")
        print("Lexical inventory, not a count of independent behaviors.")
        print("Aliases, diagnostics and compile-time capability guards are included.")
        print("Use --json for names and source locations; defaults/effects require reader inspection.")


if __name__ == "__main__":
    main()
