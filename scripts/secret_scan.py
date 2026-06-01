from __future__ import annotations

import sys
from pathlib import Path

from succession_fragility.utils.secrets import scan_file


SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "data",
    "raw",
    "interim",
    "processed",
    "feature_store",
}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    root = Path(argv[0] if argv else ".").resolve()
    findings: list[str] = []
    for path in iter_files(root):
        findings.extend(scan_file(path))
    if findings:
        print("\n".join(findings))
        return 1
    print("secret_scan_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
