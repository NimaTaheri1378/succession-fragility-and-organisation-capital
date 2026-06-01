from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from succession_fragility.utils.manifest import stable_hash, write_manifest


def _load_wrds_module() -> Any:
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The wrds package is required for WRDS extraction.") from exc
    return wrds


@dataclass(frozen=True)
class ExtractResult:
    dataset: str
    year: int | None
    status: str
    path: str
    rows: int = 0
    error: str | None = None


def _year_bounds(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def _already_done(path: Path) -> bool:
    manifest = path.with_suffix(".manifest.json")
    return path.exists() and manifest.exists()


def _write(df: pd.DataFrame, path: Path, query: str, dataset: str, year: int | None) -> ExtractResult:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    write_manifest(
        path.with_suffix(".manifest.json"),
        {
            "kind": "wrds_full_extract",
            "dataset": dataset,
            "year": year,
            "path": str(path),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "query_hash": stable_hash(query),
            "query": query,
            "status": "ok",
        },
    )
    return ExtractResult(dataset=dataset, year=year, status="ok", path=str(path), rows=int(len(df)))


def query_for(dataset: str, year: int | None = None) -> str:
    if year is not None:
        start, end = _year_bounds(year)

    if dataset == "crsp_monthly":
        return f"""
            select m.permno, m.permco, m.date, m.prc, m.ret, m.retx, m.shrout,
                   n.shrcd, n.exchcd, n.siccd, n.ticker, n.comnam, n.cusip, n.ncusip
            from crsp.msf as m
            join crsp.msenames as n
              on m.permno = n.permno
             and n.namedt <= m.date
             and m.date <= coalesce(n.nameendt, '9999-12-31')
            where m.date between '{start}' and '{end}'
              and n.shrcd in (10, 11)
              and n.exchcd in (1, 2, 3)
            order by m.date, m.permno
        """
    if dataset == "crsp_delist":
        return f"""
            select permno, dlstdt, dlstcd, dlret, dlretx, dlprc
            from crsp.msedelist
            where dlstdt between '{start}' and '{end}'
            order by dlstdt, permno
        """
    if dataset == "crsp_daily":
        return f"""
            select d.permno, d.permco, d.date, d.prc, d.ret, d.retx, d.vol, d.shrout,
                   n.shrcd, n.exchcd, n.siccd, n.ticker, n.comnam, n.cusip, n.ncusip
            from crsp.dsf as d
            join crsp.msenames as n
              on d.permno = n.permno
             and n.namedt <= d.date
             and d.date <= coalesce(n.nameendt, '9999-12-31')
            where d.date between '{start}' and '{end}'
              and n.shrcd in (10, 11)
              and n.exchcd in (1, 2, 3)
            order by d.date, d.permno
        """
    if dataset == "comp_funda":
        return f"""
            select gvkey, datadate, fyear, tic, cusip, cik, conm, at, ceq, seq,
                   txditc, pstk, pstkrv, pstkl, sale, cogs, xsga, xrd, capx,
                   dltt, dlc, che, ni, sich, fic
            from comp.funda
            where indfmt = 'INDL'
              and datafmt = 'STD'
              and popsrc = 'D'
              and consol = 'C'
              and datadate between '{start}' and '{end}'
            order by datadate, gvkey
        """
    if dataset == "boardex_roles":
        return f"""
            select directorid, companyid, directorname, companyname, rolename,
                   brdposition, datestartrole, dateendrole, leadershipteam,
                   hocountryname, sector, orgtype, isin
            from boardex_na.na_wrds_dir_profile_emp
            where datestartrole <= '{end}'
              and (dateendrole is null or dateendrole >= '{start}')
              and hocountryname in ('United States', 'USA', 'United States of America')
              and substring(isin from 1 for 2) = 'US'
            order by companyid, directorid
        """
    if dataset == "boardex_company":
        return """
            select boardid, boardname, ticker, isin, cikcode, hocountryname,
                   sector, orgtype, countryofquote, primarystock, currency,
                   mktcapitalisation, noemployees, revenue
            from boardex_na.na_wrds_company_profile
            where countryofquote in ('United States', 'USA', 'United States of America')
               or hocountryname in ('United States', 'USA', 'United States of America')
            order by boardid
        """
    if dataset == "ccm_links":
        return """
            select gvkey, lpermno as permno, lpermco as permco, linkdt, linkenddt,
                   linktype, linkprim
            from crsp.ccmxpf_lnkhist
            where lpermno is not null
              and linktype in ('LU', 'LC')
              and linkprim in ('P', 'C')
            order by gvkey, linkdt
        """
    if dataset == "ff5":
        return """
            select date, mktrf, smb, hml, rmw, cma, rf, umd
            from ff.fivefactors_monthly
            order by date
        """
    if dataset == "ibes_attention":
        return f"""
            select ticker, cusip, statpers, measure, fiscalp, fpi, numest, numup,
                   numdown, stdev, meanest, medest, usfirm, fpedats
            from ibes.statsum_epsus
            where statpers between '{start}' and '{end}'
              and measure = 'EPS'
              and usfirm = 1
            order by statpers, ticker
        """
    raise ValueError(f"Unknown dataset: {dataset}")


def output_path(root: Path, dataset: str, year: int | None) -> Path:
    if year is None:
        return root / dataset / "all.parquet"
    return root / dataset / f"year={year}" / "part.parquet"


def extract_dataset(
    dataset: str,
    output_root: Path,
    year: int | None = None,
    force: bool = False,
    db: Any | None = None,
) -> ExtractResult:
    path = output_path(output_root, dataset, year)
    if not force and _already_done(path):
        return ExtractResult(dataset=dataset, year=year, status="skipped_existing", path=str(path))

    query = query_for(dataset, year)
    try:
        if db is None:
            wrds = _load_wrds_module()
            with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as conn:
                df = conn.raw_sql(query)
        else:
            df = db.raw_sql(query)
    except SQLAlchemyError as exc:
        write_manifest(
            path.with_suffix(".manifest.json"),
            {
                "kind": "wrds_full_extract",
                "dataset": dataset,
                "year": year,
                "path": str(path),
                "query_hash": stable_hash(query),
                "status": "failed",
                "error": str(exc).splitlines()[0],
            },
        )
        return ExtractResult(dataset=dataset, year=year, status="failed", path=str(path), error=str(exc).splitlines()[0])
    return _write(df, path, query, dataset, year)


def extract_many(
    datasets: list[str],
    output_root: Path,
    years: list[int],
    force: bool = False,
) -> list[ExtractResult]:
    results: list[ExtractResult] = []
    static = {"boardex_company", "ccm_links", "ff5"}
    tasks: list[tuple[str, int | None]] = []
    for dataset in datasets:
        if dataset in static:
            tasks.append((dataset, None))
        else:
            tasks.extend((dataset, year) for year in years)

    pending: list[tuple[str, int | None]] = []
    for dataset, year in tasks:
        path = output_path(output_root, dataset, year)
        if not force and _already_done(path):
            results.append(ExtractResult(dataset=dataset, year=year, status="skipped_existing", path=str(path)))
        else:
            pending.append((dataset, year))

    if pending:
        wrds = _load_wrds_module()
        with wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME")) as db:
            for dataset, year in pending:
                results.append(extract_dataset(dataset, output_root, year, force=force, db=db))
    return results
