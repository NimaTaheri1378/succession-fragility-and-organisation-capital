from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import pandas as pd

from succession_fragility.utils.manifest import write_manifest


def _load_wrds_module() -> Any:
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The wrds package is required for coverage diagnostics.") from exc
    return wrds


def run_boardex_coverage(
    output_dir: Path,
    start_year: int = 1995,
    end_year: int = 2025,
) -> dict[str, object]:
    """Count active BoardEx U.S. role coverage by calendar year."""

    output_dir.mkdir(parents=True, exist_ok=True)
    query = f"""
        with years as (
            select generate_series({start_year}, {end_year})::int as year
        )
        select
            y.year,
            count(*) as active_role_rows,
            count(distinct r.companyid) as active_companies,
            count(distinct r.directorid) as active_people,
            count(distinct substring(r.isin from 3 for 8)) as active_cusip8,
            count(*) filter (
                where lower(coalesce(r.leadershipteam::text, '')) in ('1', 'y', 'yes', 'true', 't')
            ) as leadership_role_rows
        from years y
        left join boardex_na.na_wrds_dir_profile_emp r
          on r.datestartrole <= make_date(y.year, 12, 31)
         and (r.dateendrole is null or r.dateendrole >= make_date(y.year, 1, 1))
         and r.hocountryname in ('United States', 'USA', 'United States of America')
         and substring(r.isin from 1 for 2) = 'US'
        group by y.year
        order by y.year
    """
    wrds = _load_wrds_module()
    with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
        coverage = db.raw_sql(query)

    coverage_path = output_dir / "boardex_coverage_by_year.csv"
    coverage.to_csv(coverage_path, index=False)
    nonzero = coverage[coverage["active_companies"] > 0]
    max_companies = int(nonzero["active_companies"].max()) if len(nonzero) else 0
    threshold = max(1000, int(max_companies * 0.5))
    stable = coverage[coverage["active_companies"] >= threshold]
    recommended_start = int(stable["year"].min()) if len(stable) else None
    payload = {
        "kind": "boardex_coverage",
        "start_year": start_year,
        "end_year": end_year,
        "max_active_companies": max_companies,
        "stable_threshold_active_companies": threshold,
        "recommended_start_year": recommended_start,
        "rows": int(len(coverage)),
        "query": query,
        "output": str(coverage_path),
    }
    write_manifest(output_dir / "boardex_coverage_manifest.json", payload)
    return payload
