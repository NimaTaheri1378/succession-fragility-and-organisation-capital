from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in args.source.rglob("*.manifest.json"):
        name = "__".join(path.relative_to(args.source).parts)
        shutil.copy2(path, args.dest / name)
        count += 1
    print(f"copied {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
