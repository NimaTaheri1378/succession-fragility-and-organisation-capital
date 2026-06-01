from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import long_short_returns, performance_summary
from succession_fragility.features.ocf import add_ocf_score, build_team_features
from succession_fragility.labels.returns import add_forward_monthly_returns
from succession_fragility.models.fama_macbeth import fama_macbeth, nw_summary
from succession_fragility.plots.figures import plot_feature_heatmap, plot_long_short, plot_rank_ic
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def isin_to_cusip(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value or "").strip().upper()
    if len(text) >= 10 and text[:2].isalpha():
        return text[2:10]
    return None


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "y", "yes", "true", "t"}


def _expand_roles_to_months(roles: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    roles = roles.copy()
    roles = roles.drop_duplicates(
        [
            "directorid",
            "companyid",
            "rolename",
            "brdposition",
            "datestartrole",
            "dateendrole",
            "isin",
        ]
    )
    roles["role_start_date"] = pd.to_datetime(roles["datestartrole"], errors="coerce", format="mixed")
    roles["role_end_date"] = pd.to_datetime(roles["dateendrole"], errors="coerce", format="mixed")
    roles["role_end_date"] = roles["role_end_date"].fillna(pd.Timestamp(end))
    roles["role_start_date"] = roles["role_start_date"].fillna(pd.Timestamp(start))
    roles["title"] = (
        roles["rolename"].fillna("").astype(str) + " " + roles["brdposition"].fillna("").astype(str)
    ).str.strip()
    rows: list[dict[str, object]] = []
    for row in roles.itertuples(index=False):
        active_start = max(pd.Timestamp(start), row.role_start_date)
        active_end = min(pd.Timestamp(end), row.role_end_date)
        if pd.isna(active_start) or pd.isna(active_end) or active_start > active_end:
            continue
        for date in pd.date_range(active_start, active_end, freq="ME"):
            rows.append(
                {
                    "gvkey": str(row.companyid),
                    "date": date,
                    "person_id": str(row.directorid),
                    "title": row.title,
                    "role_start_date": row.role_start_date,
                    "role_end_date": row.role_end_date,
                    "outside_roles": 0,
                    "prior_employers": 0,
                    "internal_candidate": int(("ceo" not in str(row.title).lower()) and _truthy(row.leadershipteam)),
                    "isin": row.isin,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    role_counts = out.groupby(["person_id", "date"], observed=True)["gvkey"].transform("nunique")
    out["outside_roles"] = (role_counts - 1).clip(lower=0)
    employer_counts = out.groupby("person_id", observed=True)["gvkey"].transform("nunique")
    out["prior_employers"] = (employer_counts - 1).clip(lower=0)
    return out


def run_wrds_smoke_panel(
    smoke_dir: Path,
    output_dir: Path,
    start: str = "2020-01-31",
    end: str = "2020-12-31",
) -> dict[str, object]:
    """Build a tiny real-data linked panel from cached WRDS smoke shards."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    roles = _read(smoke_dir / "boardex_roles.parquet")
    crsp = _read(smoke_dir / "crsp_monthly.parquet")
    ff5 = _read(smoke_dir / "ff5.parquet")

    person_month = _expand_roles_to_months(roles, start, end)
    if person_month.empty:
        manifest = {"kind": "wrds_smoke_panel", "status": "failed", "reason": "no active BoardEx role-months"}
        write_manifest(manifest_dir / "wrds_smoke_panel.json", manifest)
        return manifest

    person_month["cusip_from_isin"] = person_month["isin"].map(isin_to_cusip)
    person_month["month"] = person_month["date"].dt.to_period("M")
    crsp = crsp.copy()
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["month"] = crsp["date"].dt.to_period("M")
    crsp["cusip"] = crsp["cusip"].astype(str).str.upper().str.strip()
    crsp["ncusip"] = crsp["ncusip"].astype(str).str.upper().str.strip()

    link = (
        person_month[["gvkey", "date", "month", "cusip_from_isin"]]
        .dropna()
        .drop_duplicates()
        .merge(
            crsp[["permno", "date", "month", "cusip", "ncusip"]]
            .rename(columns={"date": "crsp_date"})
            .drop_duplicates(),
            left_on=["month", "cusip_from_isin"],
            right_on=["month", "cusip"],
            how="inner",
        )
    )
    person_month = person_month.merge(link[["gvkey", "date", "permno"]].drop_duplicates(), on=["gvkey", "date"], how="left")
    person_month = person_month.dropna(subset=["permno"])

    if person_month.empty:
        manifest = {"kind": "wrds_smoke_panel", "status": "failed", "reason": "no BoardEx-CRSP smoke links"}
        write_manifest(manifest_dir / "wrds_smoke_panel.json", manifest)
        return manifest

    features = add_ocf_score(build_team_features(person_month))
    features = features.merge(link[["gvkey", "date", "permno", "crsp_date"]].drop_duplicates(), on=["gvkey", "date"], how="left")
    features["date"] = pd.to_datetime(features["crsp_date"])
    features = features.drop(columns=["crsp_date"])
    panel = features.merge(crsp, on=["permno", "date"], how="inner")
    panel["ret_excess"] = pd.to_numeric(panel["ret"], errors="coerce")
    ff5["date"] = pd.to_datetime(ff5["date"])
    if "rf" in ff5:
        panel = panel.merge(ff5[["date", "rf"]], on="date", how="left")
        panel["ret_excess"] = panel["ret_excess"] - pd.to_numeric(panel["rf"], errors="coerce").fillna(0.0) / 100.0
    panel["mktcap"] = pd.to_numeric(panel["prc"], errors="coerce").abs() * pd.to_numeric(panel["shrout"], errors="coerce")
    panel["size"] = np.log(panel["mktcap"].replace(0, np.nan))
    panel = add_forward_monthly_returns(panel)

    controls = ["ocf", "size", "key_person_concentration", "succession_depth_gap", "bench_depth"]
    usable = panel.dropna(subset=["ret_excess_fwd1m", *controls])
    if usable["date"].nunique() >= 3 and len(usable) >= 30:
        fm = fama_macbeth(usable, "ret_excess_fwd1m", controls, min_obs=5)
        fm_summary = nw_summary(fm)
    else:
        fm = pd.DataFrame()
        fm_summary = pd.DataFrame()
    ls = long_short_returns(panel, signal="ocf", q=min(5, max(2, panel.groupby("date").size().min())))
    perf = performance_summary(ls["long_short"]) if "long_short" in ls else {}
    rank_ic = pd.DataFrame(
        {
            "horizon": ["1m"],
            "spearman_ic": [usable[["ocf", "ret_excess_fwd1m"]].corr(method="spearman").iloc[0, 1] if len(usable) > 2 else np.nan],
        }
    )

    panel.drop(columns=["directorname", "companyname"], errors="ignore").to_parquet(table_dir / "wrds_smoke_panel.parquet", index=False)
    fm.to_csv(table_dir / "wrds_smoke_fama_macbeth_by_month.csv", index=False)
    fm_summary.to_csv(table_dir / "wrds_smoke_fama_macbeth_summary.csv", index=False)
    ls.to_csv(table_dir / "wrds_smoke_long_short.csv", index=False)
    pd.DataFrame([perf]).to_csv(table_dir / "wrds_smoke_performance.csv", index=False)

    if not ls.empty:
        plot_long_short(ls, figure_dir / "wrds_smoke_ocf_long_short.png")
    plot_rank_ic(rank_ic, figure_dir / "wrds_smoke_rank_ic_decay.png")
    plot_feature_heatmap(features, figure_dir / "wrds_smoke_ocf_component_heatmap.png")
    qa = inspect_figure_dir(figure_dir)

    manifest = {
        "kind": "wrds_smoke_panel",
        "status": "ok",
        "n_person_month_rows": int(len(person_month)),
        "n_feature_rows": int(len(features)),
        "n_panel_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "n_permnos": int(panel["permno"].nunique()),
        "performance": perf,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "wrds_smoke_panel.json", manifest)
    return manifest
