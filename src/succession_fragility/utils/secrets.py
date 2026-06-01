from __future__ import annotations

import os
import re
from pathlib import Path


SECRET_NAME_RE = re.compile(
    r"(API[_-]?KEY|PASSWORD|PASSWD|TOKEN|SECRET|PGPASS|WRDS_USERNAME)", re.IGNORECASE
)
SECRET_VALUE_RE = re.compile(
    r"(?i)(fred|bea|bls|eia|api|password|token|secret)[^=\n]{0,40}[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"
)


def safe_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def assert_no_secret_text(text: str, source: str = "<text>") -> None:
    if SECRET_VALUE_RE.search(text):
        raise ValueError(f"Potential secret-like value detected in {source}")


def scan_file(path: Path) -> list[str]:
    if path.is_dir() or path.stat().st_size > 5_000_000:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    findings: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if SECRET_VALUE_RE.search(line):
            findings.append(f"{path}:{i}: possible secret value")
    return findings
