#!/usr/bin/env python3
"""Fail when a relative link in a tracked Markdown file has no target.

Onboarding lives in Markdown, and a link that points at a file which was moved
or never added is invisible until someone clicks it. This checks every tracked
``*.md`` file; absolute URLs, mail links and bare anchors are ignored.

Usage:
    python3 scripts/check_docs_links.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# ](target) and ](target "title"); <target> is allowed.
LINK = re.compile(r"\]\(\s*<?([^)>\s]+)")

SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#")

# Dated research notes record local runs and reference artifacts (logs/,
# patches/, private include/ paths) that are not published with the repository.
# They are history, not documentation, so they are not link-checked.
DATED_NOTE = re.compile(r"-20\d{2}-\d{2}-\d{2}\.md$")


def tracked_markdown() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [Path(line) for line in out.splitlines() if line]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sorted(p for p in Path(".").rglob("*.md") if ".git" not in p.parts)


def main() -> int:
    broken: list[str] = []
    checked = 0
    scanned = 0
    docs = tracked_markdown()
    for doc in docs:
        if not doc.is_file() or DATED_NOTE.search(doc.name):
            continue
        scanned += 1
        for lineno, line in enumerate(
            doc.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for target in LINK.findall(line):
                if target.startswith(SKIP_PREFIXES):
                    continue
                if any(marker in target for marker in ("$", "{", "<")):
                    continue  # template placeholder, not a real path
                path = target.split("#", 1)[0]
                if not path:
                    continue
                resolved = (
                    Path(path.lstrip("/")) if path.startswith("/") else doc.parent / path
                )
                checked += 1
                if not resolved.exists():
                    broken.append(f"{doc}:{lineno}: {target} -> {resolved}")

    for item in broken:
        print(item)
    print(f"checked {checked} relative link(s) in {scanned} file(s), {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
