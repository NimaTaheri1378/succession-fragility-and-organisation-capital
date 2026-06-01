from __future__ import annotations

from pathlib import Path
from typing import Any

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
from succession_fragility.pipeline.model_ladder import TABULAR_FEATURES, _available_features, _lightgbm_or_fallback
from succession_fragility.pipeline.robustness import _attach_ff5_by_month, factor_alpha, moving_block_ci
from succession_fragility.plots.figures import savefig, set_style
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest


STRATEGY_COLUMNS = [
    "date",
    "permno",
    "ret_excess_fwd1m",
    "mktcap",
    "siccd",
    "analyst_coverage",
    "intangibility",
    "ocf",
    *TABULAR_FEATURES,
]


def _load_strategy_panel(panel_path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(panel_path).schema_arrow.names)
    except Exception:
        available = set(STRATEGY_COLUMNS)
    cols = [c for c in dict.fromkeys(STRATEGY_COLUMNS) if c in available]
    panel = pd.read_parquet(panel_path, columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    for col in panel.columns:
        if col != "date":
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel.dropna(subset=["ret_excess_fwd1m"]).copy()


def _walk_forward_lightgbm(panel: pd.DataFrame, features: list[str], test_start_year: int, n_jobs: int, seed: int) -> pd.DataFrame:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    estimator, model_name = _lightgbm_or_fallback(seed, n_jobs)
    frames: list[pd.DataFrame] = []
    for year in sorted(y for y in panel["date"].dt.year.unique() if y >= test_start_year):
        train = panel[panel["date"].dt.year < year].copy()
        test = panel[panel["date"].dt.year == year].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        pipe.fit(train[features], train["ret_excess_fwd1m"].to_numpy(dtype=float))
        keep_cols = [c for c in ["date", "permno", "ret_excess_fwd1m", "mktcap", "analyst_coverage", "intangibility", "ocf"] if c in test]
        out = test[keep_cols].copy()
        out["score"] = pipe.predict(test[features])
        out["test_year"] = int(year)
        frames.append(out)
        print(f"[sharp_strategy] predicted {model_name} target_year={year} n_test={len(out)}", flush=True)
    if not frames:
        raise RuntimeError("No walk-forward LightGBM predictions were produced.")
    return pd.concat(frames, ignore_index=True)


def _nw_tstat_mean(series: pd.Series, lags: int = 6) -> float:
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        return np.nan
    try:
        import statsmodels.api as sm

        fit = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        return float(fit.tvalues[0])
    except Exception:
        stderr = clean.std(ddof=1) / np.sqrt(len(clean))
        return float(clean.mean() / stderr) if stderr else np.nan


def _date_median_mask(frame: pd.DataFrame, column: str, high: bool) -> pd.Series:
    threshold = frame.groupby("date", observed=True)[column].transform("median")
    if high:
        return frame[column] >= threshold
    return frame[column] < threshold


def _candidate_universes(pred: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {"all": pd.Series(True, index=pred.index)}
    if "mktcap" in pred:
        mkt = pred.groupby("date", observed=True)["mktcap"].transform(lambda x: x.quantile(0.20))
        masks["nonmicrocap"] = pred["mktcap"] > mkt
    if "analyst_coverage" in pred:
        masks["low_analyst_coverage"] = _date_median_mask(pred, "analyst_coverage", high=False)
        masks["high_analyst_coverage"] = _date_median_mask(pred, "analyst_coverage", high=True)
    if "intangibility" in pred:
        masks["intangible_heavy"] = _date_median_mask(pred, "intangibility", high=True)
        masks["tangible_heavy"] = _date_median_mask(pred, "intangibility", high=False)
    if "ocf" in pred:
        masks["high_ocf_state"] = _date_median_mask(pred, "ocf", high=True)
    if "analyst_coverage" in pred and "intangibility" in pred:
        masks["low_analyst_intangible"] = masks["low_analyst_coverage"] & masks["intangible_heavy"]
    return masks


def _monthly_score_spread(pred: pd.DataFrame, q: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in pred.dropna(subset=["score"]).groupby("date", observed=True):
        if group["score"].nunique() < q:
            continue
        bucket = pd.qcut(group["score"], q=q, labels=False, duplicates="drop") + 1
        work = group.assign(_bucket=bucket)
        rows.append(
            {
                "date": date,
                "score_spread": float(work.loc[work["_bucket"].eq(q), "score"].mean() - work.loc[work["_bucket"].eq(1), "score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("date")


def _expanding_high_spread_dates(pred: pd.DataFrame, q: int, min_history: int = 24) -> set[pd.Timestamp]:
    spread = _monthly_score_spread(pred, q)
    if spread.empty:
        return set()
    expanding_median = spread["score_spread"].shift(1).expanding(min_periods=min_history).median()
    keep = spread["score_spread"] >= expanding_median
    return set(pd.to_datetime(spread.loc[keep.fillna(False), "date"]))


def _evaluate_candidate(
    pred: pd.DataFrame,
    panel_for_alpha: pd.DataFrame,
    name: str,
    mask: pd.Series,
    q: int,
    weighting: str,
    high_spread_dates: set[pd.Timestamp] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    sub = pred[mask.fillna(False)].copy()
    if high_spread_dates is not None:
        sub = sub[sub["date"].isin(high_spread_dates)].copy()
    weight_col = "mktcap" if weighting == "value" else None
    ls = long_short_returns(sub, signal="score", ret_col="ret_excess_fwd1m", weight_col=weight_col, q=q)
    weights = long_short_weights(sub, signal="score", ret_col="ret_excess_fwd1m", weight_col=weight_col, q=q)
    turnover = one_way_turnover(weights)
    gross = weighted_long_short_returns(weights) if not weights.empty else ls[["date", "long_short"]].copy()
    net_10 = apply_cost_haircut(gross, turnover, one_way_bps=10)
    perf = performance_summary(ls["long_short"]) if not ls.empty else {}
    net_perf = performance_summary(net_10["long_short_net"]) if not net_10.empty else {}
    alpha = factor_alpha(ls, panel_for_alpha) if not ls.empty else {}
    boot = moving_block_ci(ls["long_short"], reps=300) if not ls.empty else {}
    row = {
        "strategy": name,
        "q": q,
        "weighting": weighting,
        "n_obs": int(len(sub)),
        "n_months": int(ls["date"].nunique()) if not ls.empty else 0,
        "mean_turnover": float(turnover["one_way_turnover"].mean()) if not turnover.empty else np.nan,
        **perf,
        "net10_mean_ann": net_perf.get("mean_ann", np.nan),
        "net10_sharpe": net_perf.get("sharpe", np.nan),
        **alpha,
        **boot,
        "nw_t_monthly_return": _nw_tstat_mean(ls["long_short"]) if not ls.empty else np.nan,
    }
    return row, ls.assign(strategy=name, q=q, weighting=weighting), turnover.assign(strategy=name, q=q, weighting=weighting)


def _plot_strategy_cumulative(returns: pd.DataFrame, summary: pd.DataFrame, output: Path, top_n: int = 5) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    set_style()
    ranked = summary.sort_values(["sharpe", "mean_ann"], ascending=False).head(top_n)
    keep = set(zip(ranked["strategy"], ranked["q"], ranked["weighting"]))
    plt.figure(figsize=(8.8, 5.0))
    ax = plt.gca()
    for (strategy, q, weighting), group in returns.groupby(["strategy", "q", "weighting"], observed=True):
        if (strategy, q, weighting) not in keep:
            continue
        df = group.dropna(subset=["long_short"]).sort_values("date").copy()
        df["cum"] = (1 + df["long_short"]).cumprod() - 1
        ax.plot(df["date"], df["cum"], linewidth=1.9, label=_strategy_label(strategy, q, weighting))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Sharpened Model-Sort Cumulative Return")
    ax.set_xlabel("Portfolio month")
    ax.set_ylabel("Cumulative excess return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left")
    savefig(output)


def _plot_strategy_bars(summary: pd.DataFrame, output: Path, top_n: int = 10) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    set_style()
    df = summary.sort_values(["sharpe", "mean_ann"], ascending=False).head(top_n).copy()
    df["label"] = [_strategy_label(row.strategy, int(row.q), row.weighting, compact=True) for row in df.itertuples()]
    df = df.sort_values("mean_ann")
    plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()
    ax.barh(df["label"], df["mean_ann"], color="#1f6f8b")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top Strategy Variants: Annualized Return")
    ax.set_xlabel("Annualized excess return")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=8.5)
    savefig(output)


def _strategy_label(strategy: str, q: int, weighting: str, compact: bool = False) -> str:
    names = {
        "all": "All firms",
        "nonmicrocap": "Nonmicrocap",
        "low_analyst_coverage": "Low analyst coverage",
        "high_analyst_coverage": "High analyst coverage",
        "intangible_heavy": "Intangible-heavy",
        "tangible_heavy": "Tangible-heavy",
        "tangible_heavy_high_spread_months": "Tangible-heavy, high-spread months",
        "all_high_spread_months": "All firms, high-spread months",
        "nonmicrocap_high_spread_months": "Nonmicrocap, high-spread months",
        "low_analyst_coverage_high_spread_months": "Low analyst, high-spread months",
        "low_analyst_intangible": "Low analyst, intangible-heavy",
        "high_ocf_state": "High OCF state",
    }
    label = names.get(strategy, strategy.replace("_", " ").title())
    suffix = f"Q{q}, {weighting}"
    if compact and len(label) > 24:
        label = label.replace(", high-spread months", "\nhigh-spread months")
        label = label.replace(", intangible-heavy", "\nintangible-heavy")
    return f"{label} ({suffix})"


def run_sharp_strategy(
    panel_path: Path,
    output_dir: Path,
    ff5_path: Path | None = Path("data/raw/wrds/ff5/all.parquet"),
    test_start_year: int = 2015,
    n_jobs: int = 8,
    seed: int = 20260601,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, figure_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    panel = _load_strategy_panel(panel_path)
    features = _available_features(panel)
    pred = _walk_forward_lightgbm(panel, features, test_start_year=test_start_year, n_jobs=n_jobs, seed=seed)
    panel_for_alpha = _attach_ff5_by_month(panel[["date"]].drop_duplicates(), ff5_path)
    universe_masks = _candidate_universes(pred)

    rows: list[dict[str, object]] = []
    returns: list[pd.DataFrame] = []
    turnovers: list[pd.DataFrame] = []
    for q in [5, 10]:
        high_spread_dates = _expanding_high_spread_dates(pred, q=q)
        for universe, mask in universe_masks.items():
            for weighting in ["equal", "value"]:
                row, ls, turnover = _evaluate_candidate(pred, panel_for_alpha, universe, mask, q, weighting)
                rows.append(row)
                returns.append(ls)
                turnovers.append(turnover)
                if high_spread_dates and universe in {"all", "nonmicrocap", "low_analyst_coverage"}:
                    gated_name = f"{universe}_high_spread_months"
                    row, ls, turnover = _evaluate_candidate(
                        pred, panel_for_alpha, gated_name, mask, q, weighting, high_spread_dates=high_spread_dates
                    )
                    rows.append(row)
                    returns.append(ls)
                    turnovers.append(turnover)

    summary = pd.DataFrame(rows).sort_values(["sharpe", "mean_ann"], ascending=False)
    all_returns = pd.concat(returns, ignore_index=True) if returns else pd.DataFrame()
    all_turnover = pd.concat(turnovers, ignore_index=True) if turnovers else pd.DataFrame()
    summary.to_csv(table_dir / "sharp_strategy_summary.csv", index=False)
    all_returns.to_csv(table_dir / "sharp_strategy_long_short_returns.csv", index=False)
    all_turnover.to_csv(table_dir / "sharp_strategy_turnover.csv", index=False)
    _plot_strategy_cumulative(all_returns, summary, figure_dir / "sharp_strategy_cumulative.png")
    _plot_strategy_bars(summary, figure_dir / "sharp_strategy_top_returns.png")

    best = summary.head(1).to_dict(orient="records")
    qa = inspect_figure_dir(figure_dir)
    manifest = {
        "kind": "sharp_strategy",
        "status": "ok",
        "panel_path": str(panel_path),
        "ff5_path": str(ff5_path) if ff5_path else None,
        "test_start_year": test_start_year,
        "model": "lightgbm",
        "features": features,
        "n_panel_rows": int(len(panel)),
        "n_prediction_rows": int(len(pred)),
        "n_candidates": int(len(summary)),
        "best_candidate": best,
        "timing_audit": {
            "walk_forward_by_target_year": True,
            "target_year_model_uses_only_prior_years": True,
            "high_spread_month_gate_uses_only_prior_months_for_threshold": True,
            "row_level_predictions_public": False,
        },
        "public_outputs_aggregate_only": True,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "sharp_strategy.json", manifest)
    return manifest
