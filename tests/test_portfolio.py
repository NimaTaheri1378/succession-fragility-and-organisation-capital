from __future__ import annotations

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import long_short_returns, performance_summary


def test_long_short_returns_uses_top_minus_bottom() -> None:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 20),
            "permno": np.tile(np.arange(20), 2),
            "ocf": np.tile(np.arange(20), 2),
            "ret_excess_fwd1m": np.tile(np.linspace(-0.05, 0.05, 20), 2),
        }
    )
    out = long_short_returns(panel, q=10)
    assert (out["long_short"] > 0).all()


def test_performance_summary_returns_core_metrics() -> None:
    perf = performance_summary(pd.Series([0.01, -0.02, 0.03]))
    assert {"mean_ann", "vol_ann", "sharpe", "max_drawdown"} <= set(perf)
