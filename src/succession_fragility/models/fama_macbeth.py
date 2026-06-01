from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - exercised in minimal local runtimes
    sm = None


def fama_macbeth(
    panel: pd.DataFrame,
    y: str,
    x: list[str],
    date_col: str = "date",
    min_obs: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    cols = [y, *x]
    for date, group in panel.dropna(subset=cols).groupby(date_col, observed=True):
        if len(group) < max(min_obs, len(x) + 2):
            continue
        yv = group[y].astype(float)
        xv = group[x].astype(float)
        row: dict[str, float | pd.Timestamp] = {"date": date}
        if sm is not None:
            fit_x = sm.add_constant(xv, has_constant="add")
            fit = sm.OLS(yv, fit_x).fit()
            row.update({f"beta_{k}": float(v) for k, v in fit.params.items()})
        else:
            fit_x = np.column_stack([np.ones(len(xv)), xv.to_numpy()])
            beta = np.linalg.lstsq(fit_x, yv.to_numpy(), rcond=None)[0]
            names = ["const", *x]
            row.update({f"beta_{k}": float(v) for k, v in zip(names, beta, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def nw_summary(coefs: pd.DataFrame, lags: int = 6) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for col in [c for c in coefs.columns if c.startswith("beta_")]:
        series = coefs[col].dropna().astype(float)
        if series.empty:
            continue
        if sm is not None:
            x = np.ones((len(series), 1))
            fit = sm.OLS(series.to_numpy(), x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            mean_coef = float(fit.params[0])
            nw_t = float(fit.tvalues[0])
        else:
            mean_coef = float(series.mean())
            stderr = float(series.std(ddof=1) / np.sqrt(len(series))) if len(series) > 1 else np.nan
            nw_t = float(mean_coef / stderr) if stderr and not np.isnan(stderr) else np.nan
        rows.append(
            {
                "term": col.removeprefix("beta_"),
                "mean_coef": mean_coef,
                "nw_t": nw_t,
                "n_months": int(len(series)),
            }
        )
    return pd.DataFrame(rows)
