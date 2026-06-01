from __future__ import annotations

import pandas as pd

from succession_fragility.pipeline.sharp_strategy import _expanding_high_spread_dates


def test_expanding_high_spread_gate_uses_prior_months_only() -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    rows = []
    spreads = [1.0, 2.0, 3.0, 0.5, 4.0]
    for date, spread in zip(dates, spreads, strict=True):
        rows.extend(
            [
                {"date": date, "score": 0.0, "ret_excess_fwd1m": 0.0},
                {"date": date, "score": spread, "ret_excess_fwd1m": 0.0},
            ]
        )
    pred = pd.DataFrame(rows)

    keep = _expanding_high_spread_dates(pred, q=2, min_history=2)

    assert dates[2] in keep
    assert dates[3] not in keep
    assert dates[4] in keep
