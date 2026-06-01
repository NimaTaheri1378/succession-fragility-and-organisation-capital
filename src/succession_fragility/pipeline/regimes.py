from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.extract.fred import FredClient, REGIME_SERIES
from succession_fragility.plots.figures import plot_regime_timeline
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest


def _monthly_series(frame: pd.DataFrame, name: str, how: str = "last") -> pd.DataFrame:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    if how == "max":
        out = df["value"].resample("ME").max()
    elif how == "mean":
        out = df["value"].resample("ME").mean()
    else:
        out = df["value"].resample("ME").last()
    return out.rename(name).reset_index()


def build_regime_controls(
    output_dir: Path,
    start_date: str = "1995-01-01",
    end_date: str | None = None,
) -> dict[str, object]:
    """Build monthly public macro-regime controls from FRED.

    The function uses an API key only if one is already present in the process
    environment; otherwise it uses FRED's public graph CSV endpoint and records
    that keyless mode in the manifest.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, figure_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    client = FredClient()
    pulls: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for name, series_id in REGIME_SERIES.items():
        try:
            pulls[name] = client.series_observations(series_id, start=start_date, end=end_date)
        except Exception as exc:  # pragma: no cover - network availability varies
            errors[name] = str(exc).splitlines()[0]

    if not pulls:
        raise RuntimeError("No FRED regime series could be pulled.")

    monthly: list[pd.DataFrame] = []
    if "vix" in pulls:
        monthly.append(_monthly_series(pulls["vix"], "vix", how="mean"))
    if "nber_recession" in pulls:
        monthly.append(_monthly_series(pulls["nber_recession"], "nber_recession", how="max"))
    if "fed_funds" in pulls:
        monthly.append(_monthly_series(pulls["fed_funds"], "fed_funds", how="last"))
    if "term_spread" in pulls:
        monthly.append(_monthly_series(pulls["term_spread"], "term_spread", how="last"))
    if "credit_spread" in pulls:
        monthly.append(_monthly_series(pulls["credit_spread"], "credit_spread", how="last"))

    regimes = monthly[0]
    for item in monthly[1:]:
        regimes = regimes.merge(item, on="date", how="outer")
    regimes = regimes.sort_values("date").reset_index(drop=True)
    for col in regimes.columns:
        if col != "date":
            regimes[col] = pd.to_numeric(regimes[col], errors="coerce")

    if end_date is not None:
        regimes = regimes[regimes["date"] <= pd.Timestamp(end_date)]
    if "vix" in regimes:
        vix_median = regimes["vix"].median(skipna=True)
        regimes["high_vix"] = (regimes["vix"] >= vix_median).astype(float)
    if "fed_funds" in regimes:
        regimes["fed_funds_12m_change"] = regimes["fed_funds"].diff(12)
        regimes["tightening"] = (regimes["fed_funds_12m_change"] > 0.25).astype(float)
    if "term_spread" in regimes:
        regimes["inverted_curve"] = (regimes["term_spread"] < 0).astype(float)
    if "credit_spread" in regimes:
        regimes["high_credit_spread"] = (
            regimes["credit_spread"] >= regimes["credit_spread"].median(skipna=True)
        ).astype(float)
    if "nber_recession" in regimes:
        regimes["nber_recession"] = regimes["nber_recession"].fillna(0).round().clip(0, 1)

    path = table_dir / "macro_regimes.csv"
    regimes.to_csv(path, index=False)
    if "vix" in regimes:
        plot_regime_timeline(regimes, figure_dir / "macro_regime_timeline.png")
    qa = inspect_figure_dir(figure_dir)

    manifest = {
        "kind": "macro_regimes",
        "status": "ok",
        "start_date": start_date,
        "end_date": end_date,
        "series": {name: REGIME_SERIES[name] for name in pulls},
        "pull_errors": errors,
        "rows": int(len(regimes)),
        "columns": list(regimes.columns),
        "keyless_fred_csv_mode": True,
        "visual_qa": qa,
        "path": str(path),
    }
    write_manifest(manifest_dir / "macro_regimes.json", manifest)
    return manifest


def attach_regimes(panel: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    regs = regimes.copy()
    regs["date"] = pd.to_datetime(regs["date"])
    return out.merge(regs, on="date", how="left")


def infer_market_regimes_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    dates = panel[["date", "mktrf"]].drop_duplicates().copy()
    dates["date"] = pd.to_datetime(dates["date"])
    dates["mktrf"] = pd.to_numeric(dates["mktrf"], errors="coerce")
    if dates["mktrf"].abs().median(skipna=True) > 0.5:
        dates["mktrf"] = dates["mktrf"] / 100.0
    dates = dates.sort_values("date")
    dates["market_vol_12m"] = dates["mktrf"].rolling(12, min_periods=6).std()
    dates["high_market_vol"] = (
        dates["market_vol_12m"] >= dates["market_vol_12m"].median(skipna=True)
    ).astype(float)
    dates["market_drawdown_12m"] = (1 + dates["mktrf"].fillna(0)).rolling(12, min_periods=6).apply(np.prod, raw=True) - 1
    return dates[["date", "market_vol_12m", "high_market_vol", "market_drawdown_12m"]]
