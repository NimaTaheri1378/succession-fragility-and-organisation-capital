from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import wrds

from succession_fragility.utils.manifest import write_manifest

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("table", nargs="+", help="schema.table")
    args = parser.parse_args()

    by_schema: dict[str, list[str]] = {}
    for item in args.table:
        schema, table = item.split(".", 1)
        if not IDENT_RE.match(schema) or not IDENT_RE.match(table):
            raise ValueError(f"Unsafe identifier: {item}")
        by_schema.setdefault(schema, []).append(table)

    result: dict[str, list[dict[str, object]]] = {}
    with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
        for schema, tables in by_schema.items():
            quoted_tables = ", ".join("'" + t + "'" for t in sorted(set(tables)))
            sql = f"""
                select table_schema, table_name, column_name, ordinal_position, data_type
                from information_schema.columns
                where table_schema = '{schema}'
                  and table_name in ({quoted_tables})
                order by table_schema, table_name, ordinal_position
            """
            cols = db.raw_sql(sql)
            for table, group in cols.groupby("table_name", observed=True):
                key = f"{schema}.{table}"
                result[key] = [
                    {
                        "name": str(row["column_name"]),
                        "ordinal_position": int(row["ordinal_position"]),
                        "type": str(row["data_type"]),
                    }
                    for _, row in group.iterrows()
                ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(args.output.with_suffix(".manifest.json"), {"kind": "wrds_column_probe", "output": str(args.output)})
    print(json.dumps({k: len(v) for k, v in result.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
