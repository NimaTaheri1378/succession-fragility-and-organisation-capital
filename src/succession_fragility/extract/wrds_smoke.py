from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from succession_fragility.utils.manifest import stable_hash, write_manifest


def _load_wrds_module() -> Any:
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The wrds package is required for WRDS smoke pulls.") from exc
    return wrds


def _save(df: pd.DataFrame, path: Path, query: str, kind: str) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    manifest = {
        "kind": kind,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "query_hash": stable_hash(query),
        "query": query,
    }
    write_manifest(path.with_suffix(".manifest.json"), manifest)
    return manifest


def run_wrds_smoke(output_dir: Path, start_date: str = "2020-01-01", end_date: str = "2020-12-31") -> dict[str, object]:
    """Pull tiny, restartable WRDS smoke shards from discovered core tables."""

    wrds = _load_wrds_module()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, object] = {}

    queries = {
        "crsp_monthly": f"""
            with boardex_cusips as (
                select distinct substring(isin from 3 for 8) as cusip8
                from boardex_na.na_wrds_dir_profile_emp
                where datestartrole <= '{end_date}'
                  and (dateendrole is null or dateendrole >= '{start_date}')
                  and substring(isin from 1 for 2) = 'US'
                limit 500
            )
            select m.permno, m.permco, m.date, m.prc, m.ret, m.retx, m.shrout,
                   n.shrcd, n.exchcd, n.siccd, n.ticker, n.comnam, n.cusip, n.ncusip
            from crsp.msf as m
            join crsp.msenames as n
              on m.permno = n.permno
             and n.namedt <= m.date
             and m.date <= coalesce(n.nameendt, '9999-12-31')
            join boardex_cusips as b
              on n.cusip = b.cusip8
              or n.ncusip = b.cusip8
            where m.date between '{start_date}' and '{end_date}'
              and n.shrcd in (10, 11)
              and n.exchcd in (1, 2, 3)
            order by m.date, m.permno
            limit 20000
        """,
        "crsp_delist": f"""
            select permno, dlstdt, dlstcd, dlret, dlretx, dlprc
            from crsp.msedelist
            where dlstdt between '{start_date}' and '{end_date}'
            limit 5000
        """,
        "comp_funda": f"""
            select gvkey, datadate, fyear, tic, cusip, cik, conm, at, ceq, seq, txditc,
                   pstk, pstkrv, pstkl, sale, cogs, xsga, xrd, capx, dltt, dlc,
                   che, ni, sich, fic
            from comp.funda
            where indfmt = 'INDL'
              and datafmt = 'STD'
              and popsrc = 'D'
              and consol = 'C'
              and datadate between '{start_date}' and '{end_date}'
            limit 5000
        """,
        "ccm_links": """
            select gvkey, lpermno as permno, lpermco as permco, linkdt, linkenddt,
                   linktype, linkprim
            from crsp.ccmxpf_lnkhist
            where lpermno is not null
              and linktype in ('LU', 'LC')
              and linkprim in ('P', 'C')
            limit 5000
        """,
        "boardex_roles": f"""
            select directorid, companyid, directorname, companyname, rolename,
                   brdposition, datestartrole, dateendrole, leadershipteam,
                   hocountryname, sector, orgtype, isin
            from boardex_na.na_wrds_dir_profile_emp
            where datestartrole <= '{end_date}'
              and (dateendrole is null or dateendrole >= '{start_date}')
              and hocountryname in ('United States', 'USA', 'United States of America')
              and substring(isin from 1 for 2) = 'US'
            order by companyid, directorid
            limit 10000
        """,
        "boardex_company": """
            select boardid, boardname, ticker, isin, cikcode, hocountryname,
                   sector, orgtype, countryofquote, primarystock, currency
            from boardex_na.na_wrds_company_profile
            where countryofquote in ('United States', 'USA', 'United States of America')
               or hocountryname in ('United States', 'USA', 'United States of America')
            limit 5000
        """,
        "ff5": f"""
            select date, mktrf, smb, hml, rmw, cma, rf, umd
            from ff.fivefactors_monthly
            where date between '{start_date}' and '{end_date}'
        """,
        "ibes_attention": f"""
            select ticker, cusip, statpers, measure, fiscalp, fpi, numest, numup,
                   numdown, stdev, meanest, medest, usfirm, fpedats
            from ibes.statsum_epsus
            where statpers between '{start_date}' and '{end_date}'
              and measure = 'EPS'
              and usfirm = 1
            limit 5000
        """,
    }

    with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
        for name, query in queries.items():
            try:
                df = db.raw_sql(query)
            except (SQLAlchemyError, Exception) as exc:
                manifests[name] = {
                    "kind": f"wrds_smoke_{name}",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc).splitlines()[0],
                    "query_hash": stable_hash(query),
                }
                continue
            manifests[name] = {"status": "ok", **_save(df, output_dir / f"{name}.parquet", query, f"wrds_smoke_{name}")}

    write_manifest(output_dir / "wrds_smoke_summary.json", {"kind": "wrds_smoke_summary", "manifests": manifests})
    return manifests
