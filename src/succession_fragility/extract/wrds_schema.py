from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from succession_fragility.utils.manifest import stable_hash, write_manifest


DEFAULT_LIBRARIES = ("boardex_na", "crsp", "comp", "ibes", "taq", "ff")


@dataclass(frozen=True)
class SchemaAuditConfig:
    libraries: tuple[str, ...] = DEFAULT_LIBRARIES
    table_sample_limit: int = 40


def _load_wrds_module() -> Any:
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The wrds package is required for schema audits.") from exc
    return wrds


def run_schema_audit(output_path: Path, config: SchemaAuditConfig | None = None) -> dict[str, Any]:
    """Discover visible WRDS libraries/tables without pulling licensed rows."""

    wrds = _load_wrds_module()
    config = config or SchemaAuditConfig()
    with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
        visible_libraries = db.list_libraries()
        audit: dict[str, Any] = {
            "visible_library_count": len(visible_libraries),
            "required_or_optional_libraries": {},
        }
        for library in config.libraries:
            library_info: dict[str, Any] = {"visible": library in visible_libraries}
            if library_info["visible"]:
                tables = db.list_tables(library)
                library_info["table_count"] = len(tables)
                library_info["tables_sample"] = tables[: config.table_sample_limit]
            audit["required_or_optional_libraries"][library] = library_info

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        output_path.with_suffix(".manifest.json"),
        {
            "kind": "wrds_schema_audit",
            "output": str(output_path),
            "hash": stable_hash(json.dumps(audit, sort_keys=True)),
        },
    )
    return audit


def describe_tables(output_path: Path, library: str, tables: list[str]) -> dict[str, Any]:
    """Describe selected WRDS tables. This remains metadata-only."""

    wrds = _load_wrds_module()
    descriptions: dict[str, Any] = {"library": library, "tables": {}}
    with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
        for table in tables:
            desc = db.describe_table(library=library, table=table)
            descriptions["tables"][table] = {
                "columns": [
                    {"name": str(row["name"]), "type": str(row["type"])}
                    for _, row in desc.iterrows()
                ]
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(descriptions, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        output_path.with_suffix(".manifest.json"),
        {"kind": "wrds_describe_tables", "library": library, "tables": tables},
    )
    return descriptions
