from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.extract.sec_edgar import SecClient, filing_rows_for_forms, historical_submission_file_names
from succession_fragility.features.ocf import role_weight
from succession_fragility.pipeline.full_panel import _isin_to_crsp_cusip
from succession_fragility.plots.figures import plot_event_paths
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest


TOP_ROLE_MIN_WEIGHT = 0.75


def _read_year_shards(root: Path, dataset: str, years: list[int], columns: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for year in years:
        path = root / dataset / f"year={year}" / "part.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=columns))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _make_events(roles: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    if roles.empty:
        return pd.DataFrame()
    df = roles.copy()
    df["title"] = (df["rolename"].fillna("").astype(str) + " " + df["brdposition"].fillna("").astype(str)).str.strip()
    df["role_weight"] = df["title"].map(role_weight)
    df = df[df["role_weight"] >= TOP_ROLE_MIN_WEIGHT].copy()
    df["boardex_companyid"] = df["companyid"].astype(str)
    df["boardex_cusip8"] = df["isin"].map(_isin_to_crsp_cusip)
    df["start"] = pd.to_datetime(df["datestartrole"], errors="coerce", format="mixed")
    df["end"] = pd.to_datetime(df["dateendrole"], errors="coerce", format="mixed")
    min_date = pd.Timestamp(f"{min(years)}-01-01")
    max_date = pd.Timestamp(f"{max(years)}-12-31")
    keep = [
        "directorid",
        "boardex_companyid",
        "boardex_cusip8",
        "companyname",
        "title",
        "role_weight",
        "start",
        "end",
    ]
    base = df[keep].drop_duplicates()
    starts = base.dropna(subset=["start"]).copy()
    starts = starts[starts["start"].between(min_date, max_date)]
    starts["event_date"] = starts["start"]
    starts["event_type"] = "appointment"
    exits = base.dropna(subset=["end"]).copy()
    exits = exits[exits["end"].between(min_date, max_date)]
    exits["event_date"] = exits["end"]
    exits["event_type"] = "departure"
    events = pd.concat([starts, exits], ignore_index=True)
    if events.empty:
        return events
    events["event_month"] = events["event_date"].dt.to_period("M")
    events = events.sort_values(["boardex_companyid", "event_date", "event_type", "directorid"])
    return events.drop_duplicates(["boardex_companyid", "directorid", "event_date", "event_type", "title"])


def _event_panel_link(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["boardex_companyid", "permno", "date", "ocf", "ticker", "siccd"]
    if "gvkey" in panel:
        cols.append("gvkey")
    link = panel[cols].drop_duplicates().copy()
    link["date"] = pd.to_datetime(link["date"])
    link["event_month"] = link["date"].dt.to_period("M")
    link = link.sort_values(["permno", "date"])
    link["pre_event_ocf"] = link.groupby("permno", observed=True)["ocf"].shift(1)
    link["pre_event_bucket"] = link.groupby("event_month", observed=True)["pre_event_ocf"].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop") + 1 if x.nunique() >= 5 else np.nan
    )
    return link.dropna(subset=["pre_event_ocf", "pre_event_bucket"])


def _compute_event_returns(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    max_horizon: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty or daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce").fillna(0.0)
    daily_by_permno = {
        int(permno): group.sort_values("date")[["date", "ret"]]
        for permno, group in daily.dropna(subset=["permno"]).groupby("permno", observed=True)
    }
    path_rows: list[dict[str, object]] = []
    horizon_rows: list[dict[str, object]] = []
    horizons = (1, 5, 20, 60)
    for row in events.itertuples(index=False):
        series = daily_by_permno.get(int(row.permno))
        if series is None or series.empty:
            continue
        dates = series["date"].to_numpy()
        rets = series["ret"].to_numpy(dtype=float)
        start = dates.searchsorted(np.datetime64(row.event_date), side="right")
        window = rets[start : start + max_horizon]
        if len(window) < 1:
            continue
        car = np.cumprod(1.0 + window) - 1.0
        for day, value in enumerate(car, start=1):
            path_rows.append(
                {
                    "event_day": day,
                    "ocf_bucket": int(row.pre_event_bucket),
                    "event_type": row.event_type,
                    "car": float(value),
                }
            )
        for horizon in horizons:
            if len(car) >= horizon:
                horizon_rows.append(
                    {
                        "horizon": horizon,
                        "ocf_bucket": int(row.pre_event_bucket),
                        "event_type": row.event_type,
                        "car": float(car[horizon - 1]),
                    }
                )
    return pd.DataFrame(path_rows), pd.DataFrame(horizon_rows)


def _attach_cik(linked: pd.DataFrame, raw_root: Path, years: list[int]) -> pd.DataFrame:
    if "gvkey" not in linked:
        return linked.assign(cik=pd.NA)
    linked = linked.copy()
    linked["gvkey"] = linked["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    comp = _read_year_shards(raw_root, "comp_funda", years, columns=["gvkey", "datadate", "cik"])
    if comp.empty:
        return linked.assign(cik=pd.NA)
    comp = comp.dropna(subset=["gvkey", "datadate", "cik"]).copy()
    comp["gvkey"] = comp["gvkey"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    comp["available_date"] = pd.to_datetime(comp["datadate"], errors="coerce") + pd.DateOffset(months=6)
    comp["cik"] = pd.to_numeric(comp["cik"], errors="coerce").astype("Int64").astype(str).replace("<NA>", pd.NA)
    comp = comp.dropna(subset=["available_date", "cik"]).sort_values(["gvkey", "available_date"])
    merged: list[pd.DataFrame] = []
    for gvkey, events in linked.sort_values(["gvkey", "event_date"]).groupby("gvkey", observed=True):
        c = comp[comp["gvkey"].eq(str(gvkey))]
        if c.empty:
            tmp = events.copy()
            tmp["cik"] = pd.NA
            merged.append(tmp)
            continue
        merged.append(
            pd.merge_asof(
                events.sort_values("event_date"),
                c[["available_date", "cik"]].sort_values("available_date"),
                left_on="event_date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    return pd.concat(merged, ignore_index=True) if merged else linked.assign(cik=pd.NA)


def _sec_timing_audit(
    linked: pd.DataFrame,
    raw_root: Path,
    years: list[int],
    table_dir: Path,
    max_ciks: int | None,
) -> dict[str, object]:
    if not os.environ.get("SEC_USER_AGENT"):
        return {"status": "skipped_no_sec_user_agent"}
    linked_cik = _attach_cik(linked, raw_root, years).dropna(subset=["cik"]).copy()
    if linked_cik.empty:
        return {"status": "skipped_no_cik_links"}
    cik_counts = linked_cik["cik"].value_counts()
    if max_ciks is not None and max_ciks > 0:
        cik_values = cik_counts.head(max_ciks).index.astype(str).tolist()
        linked_cik = linked_cik[linked_cik["cik"].isin(cik_values)].copy()
    else:
        cik_values = cik_counts.index.astype(str).tolist()

    client = SecClient()
    forms = {"8-K", "8-K/A", "DEF 14A", "DEFA14A"}
    filing_rows: list[dict[str, object]] = []
    errors = 0
    for cik in cik_values:
        try:
            payload = client.submissions(cik)
            rows = filing_rows_for_forms(payload, forms)
            for file_name in historical_submission_file_names(payload):
                try:
                    rows.extend(filing_rows_for_forms(client.submissions_file(file_name), forms))
                except Exception:
                    errors += 1
            for row in rows:
                row["cik"] = str(int(cik)) if str(cik).isdigit() else str(cik)
            filing_rows.extend(rows)
        except Exception:
            errors += 1

    filings = pd.DataFrame(filing_rows)
    if filings.empty:
        return {
            "status": "ok_no_matching_sec_filings",
            "n_ciks_requested": len(cik_values),
            "fetch_errors": errors,
        }
    filings["cik"] = filings["cik"].astype(str)
    filings["filing_date"] = pd.to_datetime(filings["filing_date"], errors="coerce")
    filings["acceptance_datetime"] = pd.to_datetime(filings["acceptance_datetime"], errors="coerce")
    filings = filings.dropna(subset=["filing_date"]).sort_values(["cik", "filing_date"])
    event_match = linked_cik[["cik", "event_date", "event_type", "pre_event_bucket"]].copy()
    event_match["cik"] = event_match["cik"].astype(str)
    matched_parts: list[pd.DataFrame] = []
    filing_groups = {cik: group.sort_values("filing_date") for cik, group in filings.groupby("cik", observed=True)}
    for cik, events in event_match.groupby("cik", observed=True):
        f = filing_groups.get(cik)
        events = events.sort_values("event_date")
        if f is None or f.empty:
            tmp = events.copy()
            tmp["filing_date"] = pd.NaT
            tmp["acceptance_datetime"] = pd.NaT
            tmp["form"] = pd.NA
            matched_parts.append(tmp)
            continue
        matched_parts.append(
            pd.merge_asof(
                events,
                f[["filing_date", "acceptance_datetime", "form"]],
                left_on="event_date",
                right_on="filing_date",
                direction="nearest",
                tolerance=pd.Timedelta(days=30),
            )
        )
    matched = pd.concat(matched_parts, ignore_index=True) if matched_parts else event_match.assign(filing_date=pd.NaT, acceptance_datetime=pd.NaT, form=pd.NA)
    matched["event_year"] = matched["event_date"].dt.year
    matched["days_to_nearest_sec_filing"] = (matched["filing_date"] - matched["event_date"]).dt.days
    coverage = (
        matched.groupby(["event_year", "event_type"], observed=True)
        .agg(
            n_events_with_cik=("cik", "size"),
            n_with_nearby_sec_filing=("filing_date", lambda x: int(x.notna().sum())),
            median_days_to_filing=("days_to_nearest_sec_filing", "median"),
        )
        .reset_index()
    )
    by_form = (
        matched.dropna(subset=["form"])
        .groupby(["event_year", "event_type", "form"], observed=True)
        .size()
        .rename("n_matches")
        .reset_index()
    )
    coverage.to_csv(table_dir / "sec_timing_audit_coverage.csv", index=False)
    by_form.to_csv(table_dir / "sec_timing_audit_by_form.csv", index=False)
    return {
        "status": "ok",
        "n_ciks_requested": len(cik_values),
        "n_events_with_cik": int(len(linked_cik)),
        "n_sec_filing_rows_private_not_saved": int(len(filings)),
        "n_events_with_nearby_sec_filing": int(matched["filing_date"].notna().sum()),
        "fetch_errors": errors,
        "cik_cap": max_ciks,
    }


def run_event_study(
    panel_path: Path,
    raw_root: Path,
    output_dir: Path,
    years: list[int],
    sec_overlay: bool = False,
    sec_max_ciks: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, figure_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    panel_cols = ["boardex_companyid", "permno", "date", "ocf", "ticker", "siccd", "gvkey"]
    panel = pd.read_parquet(panel_path, columns=panel_cols)
    roles = _read_year_shards(
        raw_root,
        "boardex_roles",
        years,
        columns=[
            "directorid",
            "companyid",
            "companyname",
            "rolename",
            "brdposition",
            "datestartrole",
            "dateendrole",
            "isin",
        ],
    )
    daily = _read_year_shards(raw_root, "crsp_daily", years, columns=["permno", "date", "ret"])

    events = _make_events(roles, years)
    link = _event_panel_link(panel)
    linked = events.merge(link, on=["boardex_companyid", "event_month"], how="inner")
    linked = linked.drop_duplicates(["permno", "event_date", "event_type", "directorid", "title"])

    paths, horizons = _compute_event_returns(linked, daily)
    if not paths.empty:
        path_summary = (
            paths.groupby(["event_day", "ocf_bucket"], observed=True)
            .agg(mean_car=("car", "mean"), median_car=("car", "median"), n_events=("car", "size"))
            .reset_index()
        )
        type_path_summary = (
            paths.groupby(["event_type", "event_day", "ocf_bucket"], observed=True)
            .agg(mean_car=("car", "mean"), n_events=("car", "size"))
            .reset_index()
        )
    else:
        path_summary = pd.DataFrame(columns=["event_day", "ocf_bucket", "mean_car", "median_car", "n_events"])
        type_path_summary = pd.DataFrame(columns=["event_type", "event_day", "ocf_bucket", "mean_car", "n_events"])
    if not horizons.empty:
        horizon_summary = (
            horizons.groupby(["horizon", "ocf_bucket", "event_type"], observed=True)
            .agg(mean_car=("car", "mean"), median_car=("car", "median"), n_events=("car", "size"))
            .reset_index()
        )
    else:
        horizon_summary = pd.DataFrame(columns=["horizon", "ocf_bucket", "event_type", "mean_car", "median_car", "n_events"])

    event_counts = (
        linked.assign(year=linked["event_date"].dt.year)
        .groupby(["year", "event_type"], observed=True)
        .size()
        .rename("n_events")
        .reset_index()
    )
    path_summary.to_csv(table_dir / "event_path_by_ocf_bucket.csv", index=False)
    type_path_summary.to_csv(table_dir / "event_path_by_type_ocf_bucket.csv", index=False)
    horizon_summary.to_csv(table_dir / "event_car_by_horizon.csv", index=False)
    event_counts.to_csv(table_dir / "event_counts.csv", index=False)

    if not path_summary.empty:
        plot_event_paths(path_summary, figure_dir / "event_paths_by_ocf_bucket.png")

    sec_audit = (
        _sec_timing_audit(linked, raw_root, years, table_dir, max_ciks=sec_max_ciks) if sec_overlay else {"status": "not_requested"}
    )
    qa = inspect_figure_dir(figure_dir)
    manifest = {
        "kind": "event_study",
        "status": "ok",
        "years": years,
        "panel_path": str(panel_path),
        "raw_root": str(raw_root),
        "n_candidate_role_events": int(len(events)),
        "n_linked_events": int(len(linked)),
        "n_event_path_rows_private_not_saved": int(len(paths)),
        "sec_timing_audit": sec_audit,
        "public_outputs_aggregate_only": True,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "event_study.json", manifest)
    return manifest
