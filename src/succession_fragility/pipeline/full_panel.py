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


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _year_paths(root: Path, dataset: str, years: list[int]) -> list[Path]:
    return [root / dataset / f"year={year}" / "part.parquet" for year in years]


def _isin_to_crsp_cusip(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text[2:10] if len(text) >= 10 and text[:2].isalpha() else None


def _expand_roles(roles: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
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
    roles["role_start_date"] = roles["role_start_date"].fillna(pd.Timestamp(start_date))
    roles["role_end_date"] = roles["role_end_date"].fillna(pd.Timestamp(end_date))
    roles["title"] = (
        roles["rolename"].fillna("").astype(str) + " " + roles["brdposition"].fillna("").astype(str)
    ).str.strip()
    rows: list[dict[str, object]] = []
    for row in roles.itertuples(index=False):
        active_start = max(pd.Timestamp(start_date), row.role_start_date)
        active_end = min(pd.Timestamp(end_date), row.role_end_date)
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
                    "internal_candidate": int("ceo" not in str(row.title).lower() and str(row.leadershipteam).lower() in {"1", "y", "yes", "true", "t"}),
                    "boardex_companyid": str(row.companyid),
                    "boardex_cusip8": _isin_to_crsp_cusip(row.isin),
                }
            )
    person_month = pd.DataFrame(rows)
    if person_month.empty:
        return person_month
    role_counts = person_month.groupby(["person_id", "date"], observed=True)["boardex_companyid"].transform("nunique")
    person_month["outside_roles"] = (role_counts - 1).clip(lower=0)
    employer_counts = person_month.groupby("person_id", observed=True)["boardex_companyid"].transform("nunique")
    person_month["prior_employers"] = (employer_counts - 1).clip(lower=0)
    return person_month


def _attach_ccm(panel: pd.DataFrame, ccm: pd.DataFrame) -> pd.DataFrame:
    ccm = ccm.copy()
    ccm["linkdt"] = pd.to_datetime(ccm["linkdt"], errors="coerce")
    ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    panel = panel.merge(ccm, on="permno", how="left")
    valid = (panel["linkdt"].isna() | (panel["linkdt"] <= panel["date"])) & (
        panel["linkenddt"].isna() | (panel["date"] <= panel["linkenddt"])
    )
    panel = panel[valid].copy()
    panel = panel.sort_values(["permno", "date", "linkdt"]).drop_duplicates(["permno", "date"], keep="last")
    return panel.rename(columns={"gvkey_y": "gvkey"}).drop(columns=["gvkey_x"], errors="ignore")


def _attach_compustat(panel: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or comp.empty or "gvkey" not in panel:
        return panel
    comp = comp.copy()
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["available_date"] = comp["datadate"] + pd.DateOffset(months=6)
    for col in ["at", "ceq", "seq", "txditc", "pstk", "pstkrv", "pstkl", "sale", "cogs", "xsga", "xrd", "capx", "dltt", "dlc", "che", "ni"]:
        if col in comp:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
    comp["be"] = comp["seq"].fillna(comp["ceq"]) + comp["txditc"].fillna(0) - comp["pstk"].fillna(comp["pstkrv"]).fillna(comp["pstkl"]).fillna(0)
    comp["profitability"] = comp["ni"] / comp["be"].replace(0, np.nan)
    comp["investment"] = (
        comp.sort_values(["gvkey", "datadate"])
        .groupby("gvkey", observed=True)["at"]
        .pct_change(fill_method=None)
    )
    comp["leverage"] = (comp["dltt"].fillna(0) + comp["dlc"].fillna(0)) / comp["at"].replace(0, np.nan)
    comp["intangibility"] = (comp["xrd"].fillna(0) + comp["xsga"].fillna(0)) / comp["at"].replace(0, np.nan)
    keep = ["gvkey", "available_date", "be", "profitability", "investment", "leverage", "intangibility", "at"]
    left = panel.dropna(subset=["gvkey"]).copy()
    right = comp[keep].dropna(subset=["gvkey", "available_date"]).sort_values(["gvkey", "available_date"])
    right_by_gvkey = {gvkey: group.sort_values("available_date") for gvkey, group in right.groupby("gvkey", observed=True)}
    merged: list[pd.DataFrame] = []
    for gvkey, group in left.groupby("gvkey", observed=True):
        r = right_by_gvkey.get(gvkey)
        g = group.sort_values("date")
        if r is None or r.empty:
            merged.append(g)
            continue
        merged.append(
            pd.merge_asof(
                g,
                r.drop(columns=["gvkey"]),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    return pd.concat(merged, ignore_index=True) if merged else left


def _attach_ibes(panel: pd.DataFrame, ibes: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or ibes.empty:
        return panel
    ibes = ibes.copy()
    ibes["date"] = pd.to_datetime(ibes["statpers"], errors="coerce") + pd.offsets.MonthEnd(0)
    ibes["cusip8"] = ibes["cusip"].astype(str).str.upper().str[:8]
    attn = (
        ibes.groupby(["date", "cusip8"], observed=True)
        .agg(analyst_coverage=("numest", "max"), forecast_dispersion=("stdev", "mean"))
        .reset_index()
    )
    panel["cusip8"] = panel["cusip"].astype(str).str.upper().str[:8]
    return panel.merge(attn, on=["date", "cusip8"], how="left")


def _attach_daily_downside(panel: pd.DataFrame, daily: pd.DataFrame, horizons: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    if panel.empty or daily.empty:
        return panel
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ret"] = pd.to_numeric(daily["ret"], errors="coerce")
    out = panel.copy()
    for horizon in horizons:
        out[f"downside_{horizon}d"] = np.nan
        out[f"ret_fwd_{horizon}d"] = np.nan
    daily_by_permno = {
        permno: group.sort_values("date")[["date", "ret"]]
        for permno, group in daily.groupby("permno", observed=True)
    }
    for permno, pidx in out.groupby("permno", observed=True).groups.items():
        d = daily_by_permno.get(permno)
        if d is None or d.empty:
            continue
        dates = d["date"].to_numpy()
        rets = d["ret"].fillna(0.0).to_numpy()
        pdates = out.loc[pidx, "date"].to_numpy()
        for row_idx, pdate in zip(pidx, pdates, strict=True):
            start = dates.searchsorted(pdate, side="right")
            for horizon in horizons:
                window = rets[start : start + horizon]
                if len(window) < horizon:
                    continue
                total = float(np.prod(1 + window) - 1)
                out.at[row_idx, f"ret_fwd_{horizon}d"] = total
                out.at[row_idx, f"downside_{horizon}d"] = float(total < 0)
    return out


def analyze_panel(panel: pd.DataFrame, output_dir: Path, years: list[int]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    for path in [figure_dir, table_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        "ocf",
        "size",
        "bm",
        "momentum",
        "profitability",
        "investment",
        "leverage",
        "intangibility",
        "analyst_coverage",
        "ret_excess_fwd1m",
        "ret_fwd_20d",
        "ret_fwd_60d",
    ]
    for col in numeric_cols:
        if col in panel:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.replace([np.inf, -np.inf], np.nan)

    controls = [
        "ocf",
        "size",
        "bm",
        "momentum",
        "profitability",
        "investment",
        "leverage",
        "intangibility",
        "analyst_coverage",
    ]
    available_controls = [c for c in controls if c in panel and panel[c].notna().sum() > 0]
    usable = panel.dropna(subset=["ret_excess_fwd1m", *available_controls])
    fm = fama_macbeth(usable, "ret_excess_fwd1m", available_controls, min_obs=20) if len(usable) else pd.DataFrame()
    fm_summary = nw_summary(fm) if not fm.empty else pd.DataFrame()
    ls = long_short_returns(panel, signal="ocf", q=5)
    perf = performance_summary(ls["long_short"]) if not ls.empty else {}
    rank_rows = []
    for label, col in [("1m", "ret_excess_fwd1m"), ("20d", "ret_fwd_20d"), ("60d", "ret_fwd_60d")]:
        if col in panel:
            sub = panel[["ocf", col]].dropna()
            rank_rows.append(
                {
                    "horizon": label,
                    "spearman_ic": sub.corr(method="spearman").iloc[0, 1] if len(sub) > 2 else np.nan,
                }
            )
    rank_ic = pd.DataFrame(rank_rows)

    fm_summary.to_csv(table_dir / "fama_macbeth_summary.csv", index=False)
    ls.to_csv(table_dir / "ocf_long_short.csv", index=False)
    pd.DataFrame([perf]).to_csv(table_dir / "performance.csv", index=False)
    panel.groupby("date").agg(n_permno=("permno", "nunique"), n_obs=("permno", "size")).reset_index().to_csv(
        table_dir / "coverage_by_month.csv", index=False
    )

    if not ls.empty:
        plot_long_short(ls, figure_dir / "ocf_long_short.png")
    plot_rank_ic(rank_ic, figure_dir / "rank_ic_decay.png")
    feature_cols = [
        "key_person_concentration",
        "succession_depth_gap",
        "external_load",
        "bench_depth",
        "team_cohesion_decay",
    ]
    plot_feature_heatmap(panel.drop_duplicates(["permno", "date"])[feature_cols], figure_dir / "ocf_component_heatmap.png")
    qa = inspect_figure_dir(figure_dir)

    manifest = {
        "kind": "full_panel",
        "years": years,
        "status": "ok",
        "n_panel_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "n_permnos": int(panel["permno"].nunique()),
        "n_gvkeys": int(panel["gvkey"].nunique()) if "gvkey" in panel else 0,
        "controls": available_controls,
        "has_daily_downside": bool("downside_20d" in panel and panel["downside_20d"].notna().any()),
        "performance": perf,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "full_panel.json", manifest)
    return manifest


def analyze_panel_file(panel_path: Path, output_dir: Path, years: list[int]) -> dict[str, object]:
    return analyze_panel(pd.read_parquet(panel_path), output_dir, years)


def run_full_panel(
    raw_root: Path,
    output_dir: Path,
    years: list[int],
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    start_date = start_date or f"{min(years)}-01-31"
    end_date = end_date or f"{max(years)}-12-31"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    data_root = raw_root.parents[1] if len(raw_root.parents) > 1 else raw_root.parent
    private_dir = data_root / "processed" / "panels" / f"{min(years)}_{max(years)}"
    for path in [figure_dir, table_dir, manifest_dir, private_dir]:
        path.mkdir(parents=True, exist_ok=True)

    roles = _read_many(_year_paths(raw_root, "boardex_roles", years))
    crsp = _read_many(_year_paths(raw_root, "crsp_monthly", years))
    daily = _read_many(_year_paths(raw_root, "crsp_daily", years))
    comp = _read_many(_year_paths(raw_root, "comp_funda", years))
    ibes = _read_many(_year_paths(raw_root, "ibes_attention", years))
    ccm = pd.read_parquet(raw_root / "ccm_links" / "all.parquet")
    ff5 = pd.read_parquet(raw_root / "ff5" / "all.parquet")

    person_month = _expand_roles(roles, start_date, end_date)
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["month"] = crsp["date"].dt.to_period("M")
    person_month["month"] = person_month["date"].dt.to_period("M")
    crsp["cusip"] = crsp["cusip"].astype(str).str.upper().str.strip()
    crsp["ncusip"] = crsp["ncusip"].astype(str).str.upper().str.strip()

    link = (
        person_month[["gvkey", "date", "month", "boardex_cusip8"]]
        .dropna()
        .drop_duplicates()
        .merge(
            crsp[["permno", "date", "month", "cusip", "ncusip"]]
            .rename(columns={"date": "crsp_date"})
            .drop_duplicates(),
            left_on=["month", "boardex_cusip8"],
            right_on=["month", "cusip"],
            how="inner",
        )
    )
    person_month = person_month.merge(link[["gvkey", "date", "permno"]].drop_duplicates(), on=["gvkey", "date"], how="inner")
    features = add_ocf_score(build_team_features(person_month)).rename(columns={"gvkey": "boardex_companyid"})
    features = features.merge(
        link.rename(columns={"gvkey": "boardex_companyid"})[["boardex_companyid", "date", "permno", "crsp_date"]].drop_duplicates(),
        on=["boardex_companyid", "date"],
        how="inner",
    )
    features["date"] = pd.to_datetime(features["crsp_date"])
    features = features.drop(columns=["crsp_date"])
    panel = features.merge(crsp, on=["permno", "date"], how="inner")
    panel = _attach_ccm(panel, ccm)
    panel = _attach_compustat(panel, comp)
    panel = _attach_ibes(panel, ibes)
    ff5["date"] = pd.to_datetime(ff5["date"])
    panel["_month"] = panel["date"].dt.to_period("M")
    ff5["_month"] = ff5["date"].dt.to_period("M")
    panel = panel.merge(
        ff5[["_month", "mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"]].drop_duplicates("_month"),
        on="_month",
        how="left",
    ).drop(columns=["_month"])

    panel["ret"] = pd.to_numeric(panel["ret"], errors="coerce")
    panel["rf"] = pd.to_numeric(panel["rf"], errors="coerce").fillna(0) / 100.0
    panel["ret_excess"] = panel["ret"] - panel["rf"]
    panel["mktcap"] = pd.to_numeric(panel["prc"], errors="coerce").abs() * pd.to_numeric(panel["shrout"], errors="coerce")
    panel["size"] = np.log(panel["mktcap"].replace(0, np.nan))
    panel["bm"] = panel["be"] / panel["mktcap"].replace(0, np.nan)
    panel = panel.sort_values(["permno", "date"])
    panel["momentum"] = panel.groupby("permno", observed=True)["ret_excess"].transform(
        lambda x: (1 + x.shift(1)).rolling(11, min_periods=6).apply(np.prod, raw=True) - 1
    )
    panel = add_forward_monthly_returns(panel)
    panel = _attach_daily_downside(panel, daily)
    panel.to_parquet(private_dir / "panel.parquet", index=False)
    manifest = analyze_panel(panel, output_dir, years)
    manifest["n_person_month_rows"] = int(len(person_month))
    write_manifest(manifest_dir / "full_panel.json", manifest)
    return manifest
