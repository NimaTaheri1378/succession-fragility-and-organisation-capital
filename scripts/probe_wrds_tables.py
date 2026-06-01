from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wrds

from succession_fragility.utils.manifest import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/manifests/wrds_table_probe.json"))
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["ibes:stat", "comp:funda", "crsp:monthly", "crsp:ccmxpf", "boardex_na:wrds"],
    )
    args = parser.parse_args()

    result: dict[str, list[str]] = {}
    with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
        for item in args.patterns:
            library, pattern = item.split(":", 1)
            result[item] = [t for t in db.list_tables(library) if pattern.lower() in t.lower()]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(args.output.with_suffix(".manifest.json"), {"kind": "wrds_table_probe", "output": str(args.output)})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
