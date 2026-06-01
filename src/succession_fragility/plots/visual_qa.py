from __future__ import annotations

from pathlib import Path

from PIL import Image


def inspect_png(path: Path) -> dict[str, object]:
    with Image.open(path) as img:
        extrema = img.convert("L").getextrema()
        width, height = img.size
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "nonblank": extrema != (255, 255) and extrema != (0, 0),
    }


def inspect_figure_dir(directory: Path) -> list[dict[str, object]]:
    return [inspect_png(path) for path in sorted(directory.glob("*.png"))]
