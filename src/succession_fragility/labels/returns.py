from __future__ import annotations

import pandas as pd


def add_forward_monthly_returns(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["permno", "date"]).copy()
    out["ret_excess_fwd1m"] = out.groupby("permno", observed=True)["ret_excess"].shift(-1)
    return out


def add_downside_labels(daily_returns: pd.DataFrame, horizons: tuple[int, ...] = (20, 60)) -> pd.DataFrame:
    df = daily_returns.sort_values(["permno", "date"]).copy()
    for horizon in horizons:
        label = f"downside_{horizon}d"
        roll = (
            df.groupby("permno", observed=True)["ret_excess"]
            .rolling(horizon, min_periods=horizon)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df[label] = roll.groupby(df["permno"], observed=True).shift(-(horizon - 1)) < 0
    return df
