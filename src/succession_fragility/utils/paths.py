from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(os.environ.get("OCF_PROJECT_ROOT", Path.cwd())).expanduser().resolve()


def data_dir(root: Path | None = None) -> Path:
    return Path(os.environ.get("OCF_DATA_DIR", (root or project_root()) / "data")).expanduser()


def report_dir(root: Path | None = None) -> Path:
    return Path(os.environ.get("OCF_OUTPUT_DIR", (root or project_root()) / "reports")).expanduser()


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
