from __future__ import annotations

import numpy as np
import pandas as pd


def assign_quantiles(df: pd.DataFrame, signal: str, date_col: str = "date", q: int = 10) -> pd.Series:
    def _bucket(x: pd.Series) -> pd.Series:
        valid = x.dropna()
        if valid.nunique() < q:
            return pd.Series(np.nan, index=x.index)
        return pd.qcut(x, q=q, labels=False, duplicates="drop") + 1

    return df.groupby(date_col, observed=True)[signal].transform(_bucket)


def long_short_returns(
    panel: pd.DataFrame,
    signal: str = "ocf",
    ret_col: str = "ret_excess_fwd1m",
    weight_col: str | None = None,
    q: int = 10,
) -> pd.DataFrame:
    df = panel.dropna(subset=[signal, ret_col]).copy()
    df["bucket"] = assign_quantiles(df, signal, q=q)
    df = df.dropna(subset=["bucket"])
    if weight_col is None:
        grouped = df.groupby(["date", "bucket"], observed=True)[ret_col].mean()
    else:
        work = df.dropna(subset=[weight_col]).copy()
        work["_wret"] = work[ret_col] * work[weight_col]
        grouped = work.groupby(["date", "bucket"], observed=True).apply(
            lambda x: x["_wret"].sum() / x[weight_col].sum(), include_groups=False
        )
    piv = grouped.rename("ret").reset_index().pivot(index="date", columns="bucket", values="ret")
    result = pd.DataFrame(index=piv.index)
    result["long"] = piv.get(q)
    result["short"] = piv.get(1)
    result["long_short"] = result["long"] - result["short"]
    return result.reset_index()


def long_short_weights(
    panel: pd.DataFrame,
    signal: str = "ocf",
    ret_col: str = "ret_excess_fwd1m",
    id_col: str = "permno",
    date_col: str = "date",
    weight_col: str | None = None,
    q: int = 10,
) -> pd.DataFrame:
    """Return signed long-short formation weights by month.

    Top-bucket names sum to +1 and bottom-bucket names sum to -1 each month.
    The weights are formation-date weights paired with the next-period return in
    ``ret_col`` so turnover and cost haircuts can be recomputed deterministically.
    """

    df = panel.dropna(subset=[signal, ret_col, id_col, date_col]).copy()
    df["bucket"] = assign_quantiles(df, signal, date_col=date_col, q=q)
    df = df.dropna(subset=["bucket"])
    df = df[df["bucket"].isin([1, q])].copy()
    if df.empty:
        return pd.DataFrame(columns=[date_col, id_col, "weight", ret_col])

    if weight_col is None:
        df["_base_weight"] = 1.0
    else:
        df["_base_weight"] = pd.to_numeric(df[weight_col], errors="coerce").clip(lower=0)
        df["_base_weight"] = df["_base_weight"].replace(0, np.nan)
    df = df.dropna(subset=["_base_weight"])
    signs = np.where(df["bucket"].eq(q), 1.0, -1.0)
    denom = df.groupby([date_col, "bucket"], observed=True)["_base_weight"].transform("sum")
    df["weight"] = signs * df["_base_weight"] / denom.replace(0, np.nan)
    return df[[date_col, id_col, "weight", ret_col]].dropna(subset=["weight"]).copy()


def weighted_long_short_returns(weights: pd.DataFrame, ret_col: str = "ret_excess_fwd1m") -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=["date", "long_short"])
    out = (
        weights.assign(_wret=weights["weight"] * pd.to_numeric(weights[ret_col], errors="coerce"))
        .groupby("date", observed=True)["_wret"]
        .sum()
        .rename("long_short")
        .reset_index()
    )
    return out


def one_way_turnover(
    weights: pd.DataFrame,
    id_col: str = "permno",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute one-way long-short turnover from signed weights.

    The 0.5 multiplier converts the sum of absolute signed-weight changes into a
    one-way trading measure. Entry into the first portfolio month is excluded.
    """

    if weights.empty:
        return pd.DataFrame(columns=[date_col, "one_way_turnover"])
    wide = (
        weights[[date_col, id_col, "weight"]]
        .drop_duplicates([date_col, id_col])
        .pivot(index=date_col, columns=id_col, values="weight")
        .fillna(0.0)
        .sort_index()
    )
    turnover = 0.5 * wide.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = np.nan
    return turnover.rename("one_way_turnover").reset_index()


def apply_cost_haircut(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    one_way_bps: float,
    date_col: str = "date",
) -> pd.DataFrame:
    out = returns.merge(turnover, on=date_col, how="left")
    out["cost"] = out["one_way_turnover"].fillna(0.0) * (one_way_bps / 10000.0)
    out["long_short_net"] = out["long_short"] - out["cost"]
    return out


def performance_summary(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {"mean_ann": np.nan, "vol_ann": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    wealth = (1 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    vol = clean.std(ddof=1) * np.sqrt(periods_per_year)
    mean = clean.mean() * periods_per_year
    return {
        "mean_ann": float(mean),
        "vol_ann": float(vol),
        "sharpe": float(mean / vol) if vol else np.nan,
        "max_drawdown": float(drawdown.min()),
    }
