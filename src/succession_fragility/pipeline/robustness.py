from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import (
    apply_cost_haircut,
    long_short_returns,
    long_short_weights,
    one_way_turnover,
    performance_summary,
    weighted_long_short_returns,
)
from succession_fragility.features.ocf import zscore_by_date
from succession_fragility.models.fama_macbeth import fama_macbeth, nw_summary
from succession_fragility.pipeline.regimes import infer_market_regimes_from_panel
from succession_fragility.plots.figures import plot_rolling_alpha, plot_turnover_cost_frontier
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None


BASE_CONTROLS = [
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


def _clean_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.replace([np.inf, -np.inf], np.nan)


def _sic_between(frame: pd.DataFrame, low: int, high: int) -> pd.Series:
    sic = pd.to_numeric(frame.get("siccd"), errors="coerce")
    return sic.between(low, high)


def add_alternative_ocf_signals(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    component_cols = [
        "key_person_concentration",
        "succession_depth_gap",
        "external_load",
        "poaching_pressure",
        "team_cohesion_decay",
        "bench_depth",
    ]
    for col in component_cols:
        if col in out and f"z_{col}" not in out:
            out[f"z_{col}"] = zscore_by_date(out, col)
    if "z_key_person_concentration" in out:
        out["ocf_kpc_only"] = out["z_key_person_concentration"]
    pieces = [c for c in ["z_key_person_concentration", "z_succession_depth_gap", "z_team_cohesion_decay"] if c in out]
    if pieces:
        out["ocf_core_roles"] = out[pieces].sum(axis=1)
    pieces = [c for c in ["z_key_person_concentration", "z_succession_depth_gap", "z_external_load", "z_poaching_pressure"] if c in out]
    if pieces:
        out["ocf_no_bench_credit"] = out[pieces].sum(axis=1)
    return out


def _factor_frame(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["date", "mktrf", "smb", "hml", "rmw", "cma", "umd"] if c in panel]
    ff = panel[cols].drop_duplicates("date").copy()
    for col in cols:
        if col == "date":
            continue
        ff[col] = pd.to_numeric(ff[col], errors="coerce")
        median_abs = ff[col].abs().median(skipna=True)
        if pd.notna(median_abs) and median_abs > 0.5:
            ff[col] = ff[col] / 100.0
    return ff


def _attach_ff5_by_month(panel: pd.DataFrame, ff5_path: Path | None) -> pd.DataFrame:
    if ff5_path is None or not ff5_path.exists():
        return panel
    ff = pd.read_parquet(ff5_path)
    if "date" not in ff:
        return panel
    factor_cols = [c for c in ["mktrf", "smb", "hml", "rmw", "cma", "rf", "umd"] if c in ff]
    if not factor_cols:
        return panel
    out = panel.drop(columns=[c for c in factor_cols if c in panel], errors="ignore").copy()
    out["_month"] = pd.to_datetime(out["date"]).dt.to_period("M")
    ff = ff[["date", *factor_cols]].copy()
    ff["_month"] = pd.to_datetime(ff["date"]).dt.to_period("M")
    ff = ff.drop(columns=["date"]).drop_duplicates("_month")
    out = out.merge(ff, on="_month", how="left").drop(columns=["_month"])
    return out


def factor_alpha(
    returns: pd.DataFrame,
    panel: pd.DataFrame,
    ret_col: str = "long_short",
    factors: tuple[str, ...] = ("mktrf", "smb", "hml", "rmw", "cma", "umd"),
) -> dict[str, float | int]:
    if sm is None or returns.empty:
        return {"alpha_month": np.nan, "alpha_ann": np.nan, "t_alpha": np.nan, "n_months": 0}
    ff = _factor_frame(panel)
    data = returns[["date", ret_col]].merge(ff, on="date", how="left").dropna()
    use_factors = [f for f in factors if f in data and data[f].notna().sum() > 0]
    if len(data) < max(24, len(use_factors) + 2):
        return {"alpha_month": np.nan, "alpha_ann": np.nan, "t_alpha": np.nan, "n_months": int(len(data))}
    fit = sm.OLS(data[ret_col].astype(float), sm.add_constant(data[use_factors].astype(float), has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )
    alpha = float(fit.params["const"])
    return {
        "alpha_month": alpha,
        "alpha_ann": alpha * 12.0,
        "t_alpha": float(fit.tvalues["const"]),
        "n_months": int(len(data)),
    }


def moving_block_ci(
    returns: pd.Series,
    block_len: int = 12,
    reps: int = 500,
    seed: int = 20260601,
) -> dict[str, float]:
    clean = returns.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < block_len * 2:
        return {"boot_mean_ann": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_len + 1)
    means = []
    for _ in range(reps):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(clean[start : start + block_len])
        means.append(np.mean(sample[:n]) * 12.0)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"boot_mean_ann": float(np.mean(means)), "boot_ci_low": float(lo), "boot_ci_high": float(hi)}


def _slice_definitions(panel: pd.DataFrame) -> dict[str, pd.Series]:
    defs: dict[str, pd.Series] = {"baseline": pd.Series(True, index=panel.index)}
    if "siccd" in panel:
        defs["exclude_financials"] = ~_sic_between(panel, 6000, 6999)
        defs["exclude_utilities"] = ~_sic_between(panel, 4900, 4999)
        defs["exclude_financials_utilities"] = defs["exclude_financials"] & defs["exclude_utilities"]
    if "mktcap" in panel:
        cutoff = panel.groupby("date", observed=True)["mktcap"].transform(lambda x: x.quantile(0.2))
        defs["exclude_smallest_covered_firms"] = panel["mktcap"] > cutoff
    if "team_size" in panel:
        cutoff = panel.groupby("date", observed=True)["team_size"].transform("median")
        defs["deeper_boardex_coverage"] = panel["team_size"] >= cutoff
    if "analyst_coverage" in panel:
        cov = panel.groupby("date", observed=True)["analyst_coverage"].transform("median")
        defs["high_analyst_coverage"] = panel["analyst_coverage"] >= cov
        defs["low_analyst_coverage"] = panel["analyst_coverage"] < cov
    if "intangibility" in panel:
        tan = panel.groupby("date", observed=True)["intangibility"].transform("median")
        defs["intangible_heavy"] = panel["intangibility"] >= tan
        defs["tangible_heavy"] = panel["intangibility"] < tan
    if "high_vix" in panel:
        defs["high_vix"] = panel["high_vix"].eq(1)
        defs["low_vix"] = panel["high_vix"].eq(0)
    if "nber_recession" in panel:
        defs["recession"] = panel["nber_recession"].eq(1)
        defs["expansion"] = panel["nber_recession"].eq(0)
    if "tightening" in panel:
        defs["tightening"] = panel["tightening"].eq(1)
        defs["easing_or_flat"] = panel["tightening"].eq(0)
    if "high_market_vol" in panel:
        defs["high_market_vol"] = panel["high_market_vol"].eq(1)
        defs["low_market_vol"] = panel["high_market_vol"].eq(0)
    return defs


def _double_sort(panel: pd.DataFrame, secondary: str, signal: str = "ocf", q: int = 5) -> pd.DataFrame:
    if secondary not in panel:
        return pd.DataFrame()
    df = panel.dropna(subset=[signal, secondary, "ret_excess_fwd1m"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["secondary_bucket"] = df.groupby("date", observed=True)[secondary].transform(
        lambda x: pd.qcut(x, q=2, labels=False, duplicates="drop") + 1 if x.nunique() >= 2 else np.nan
    )
    rows: list[dict[str, float | str | int]] = []
    for bucket, group in df.dropna(subset=["secondary_bucket"]).groupby("secondary_bucket", observed=True):
        ls = long_short_returns(group, signal=signal, q=q)
        perf = performance_summary(ls["long_short"]) if not ls.empty else {}
        rows.append({"secondary": secondary, "secondary_bucket": int(bucket), "n_obs": int(len(group)), **perf})
    return pd.DataFrame(rows)


def _rolling_alpha(ls: pd.DataFrame, panel: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ls = ls.sort_values("date").reset_index(drop=True)
    for end in range(window, len(ls) + 1):
        sub = ls.iloc[end - window : end]
        alpha = factor_alpha(sub, panel)
        rows.append({"date": sub["date"].iloc[-1], **alpha})
    return pd.DataFrame(rows)


def _evaluate_variant(panel: pd.DataFrame, name: str, signal: str, q: int, weight_col: str | None) -> dict[str, object]:
    sub = panel.dropna(subset=[signal, "ret_excess_fwd1m"])
    ls = long_short_returns(sub, signal=signal, q=q, weight_col=weight_col)
    perf = performance_summary(ls["long_short"]) if not ls.empty else {}
    alpha = factor_alpha(ls, sub)
    boot = moving_block_ci(ls["long_short"]) if not ls.empty else {}
    weights = long_short_weights(sub, signal=signal, q=q, weight_col=weight_col)
    turnover = one_way_turnover(weights)
    mean_turnover = float(turnover["one_way_turnover"].mean()) if not turnover.empty else np.nan
    return {
        "variant": name,
        "signal": signal,
        "q": q,
        "weighting": "value" if weight_col else "equal",
        "n_obs": int(len(sub)),
        "n_months": int(sub["date"].nunique()) if "date" in sub else 0,
        "mean_turnover": mean_turnover,
        **perf,
        **alpha,
        **boot,
    }


def run_robustness(
    panel_path: Path,
    output_dir: Path,
    regimes_path: Path | None = None,
    ff5_path: Path | None = Path("data/raw/wrds/ff5/all.parquet"),
    q: int = 5,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, figure_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    panel = add_alternative_ocf_signals(_clean_panel(pd.read_parquet(panel_path)))
    panel = _attach_ff5_by_month(panel, ff5_path)
    if regimes_path and regimes_path.exists():
        regimes = pd.read_csv(regimes_path, parse_dates=["date"])
        panel = panel.merge(regimes, on="date", how="left")
    else:
        panel = panel.merge(infer_market_regimes_from_panel(panel), on="date", how="left")

    signals = [s for s in ["ocf", "ocf_kpc_only", "ocf_core_roles", "ocf_no_bench_credit"] if s in panel]
    rows: list[dict[str, object]] = []
    slices = _slice_definitions(panel)
    for slice_name, mask in slices.items():
        sub = panel[mask.fillna(False)].copy()
        if sub.empty:
            continue
        for signal in signals:
            rows.append(_evaluate_variant(sub, slice_name, signal, q=q, weight_col=None))
            if "mktcap" in sub:
                rows.append(_evaluate_variant(sub, slice_name, signal, q=q, weight_col="mktcap"))
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "robustness_summary.csv", index=False)

    controls = [c for c in BASE_CONTROLS if c in panel and panel[c].notna().sum() > 0]
    fm = fama_macbeth(panel.dropna(subset=["ret_excess_fwd1m", *controls]), "ret_excess_fwd1m", controls, min_obs=40)
    nw = nw_summary(fm)
    fm.to_csv(table_dir / "fama_macbeth_monthly_betas.csv", index=False)
    nw.to_csv(table_dir / "fama_macbeth_nw_summary.csv", index=False)

    baseline_ls = long_short_returns(panel, signal="ocf", q=q)
    baseline_ls.to_csv(table_dir / "baseline_long_short.csv", index=False)
    alpha = factor_alpha(baseline_ls, panel)
    pd.DataFrame([alpha]).to_csv(table_dir / "factor_alpha_ff5_umd.csv", index=False)
    rolling = _rolling_alpha(baseline_ls, panel)
    rolling.to_csv(table_dir / "rolling_factor_alpha.csv", index=False)
    if not rolling.empty:
        plot_rolling_alpha(rolling, figure_dir / "rolling_ff5_umd_alpha.png")

    cost_rows: list[dict[str, object]] = []
    weights = long_short_weights(panel, signal="ocf", q=q)
    gross = weighted_long_short_returns(weights)
    turnover = one_way_turnover(weights)
    turnover.to_csv(table_dir / "baseline_turnover.csv", index=False)
    for bps in [0, 5, 10, 25, 50, 100]:
        net = apply_cost_haircut(gross, turnover, bps)
        perf = performance_summary(net["long_short_net"])
        cost_rows.append({"one_way_bps": bps, **perf, "alpha_ann": perf.get("mean_ann", np.nan)})
    cost_frontier = pd.DataFrame(cost_rows)
    cost_frontier.to_csv(table_dir / "turnover_cost_frontier.csv", index=False)
    plot_turnover_cost_frontier(cost_frontier, figure_dir / "turnover_cost_frontier.png")

    doubles = pd.concat(
        [_double_sort(panel, secondary) for secondary in ["analyst_coverage", "intangibility"]], ignore_index=True
    )
    doubles.to_csv(table_dir / "double_sorts.csv", index=False)

    subperiod_rows = []
    for label, years in {
        "1995_2004": range(1995, 2005),
        "2005_2014": range(2005, 2015),
        "2015_2024": range(2015, 2025),
    }.items():
        sub = panel[panel["date"].dt.year.isin(years)]
        ls = long_short_returns(sub, signal="ocf", q=q)
        perf = performance_summary(ls["long_short"]) if not ls.empty else {}
        subperiod_rows.append({"subperiod": label, "n_months": int(sub["date"].nunique()), **perf})
    pd.DataFrame(subperiod_rows).to_csv(table_dir / "subperiod_stability.csv", index=False)

    qa = inspect_figure_dir(figure_dir)
    manifest = {
        "kind": "robustness",
        "status": "ok",
        "panel_path": str(panel_path),
        "regimes_path": str(regimes_path) if regimes_path else None,
        "ff5_path": str(ff5_path) if ff5_path else None,
        "n_panel_rows": int(len(panel)),
        "n_variants": int(len(summary)),
        "signals": signals,
        "controls": controls,
        "factor_alpha": alpha,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "robustness.json", manifest)
    return manifest
