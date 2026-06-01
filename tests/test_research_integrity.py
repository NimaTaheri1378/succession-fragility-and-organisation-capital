from __future__ import annotations

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import long_short_weights, one_way_turnover
from succession_fragility.pipeline.event_study import _make_events


def test_signed_long_short_weights_are_dollar_neutral_by_month() -> None:
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 20),
            "permno": np.tile(np.arange(20), 2),
            "ocf": np.tile(np.arange(20), 2),
            "ret_excess_fwd1m": 0.01,
            "mktcap": np.tile(np.arange(1, 21), 2),
        }
    )
    weights = long_short_weights(panel, q=5, weight_col="mktcap")
    net = weights.groupby("date", observed=True)["weight"].sum()
    gross = weights.groupby("date", observed=True)["weight"].apply(lambda x: x.abs().sum())
    assert np.allclose(net.to_numpy(), 0.0)
    assert np.allclose(gross.to_numpy(), 2.0)


def test_turnover_excludes_first_entry_month() -> None:
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]),
            "permno": [1, 2, 1, 3],
            "weight": [1.0, -1.0, 1.0, -1.0],
            "ret_excess_fwd1m": 0.0,
        }
    )
    turnover = one_way_turnover(weights)
    assert turnover["one_way_turnover"].isna().iloc[0]
    assert turnover["one_way_turnover"].iloc[1] == 1.0


def test_event_builder_uses_only_top_roles_inside_sample() -> None:
    roles = pd.DataFrame(
        {
            "directorid": ["a", "b"],
            "companyid": ["1", "1"],
            "companyname": ["A", "A"],
            "rolename": ["Chief Executive Officer", "Director"],
            "brdposition": ["", ""],
            "datestartrole": ["2020-03-15", "2020-03-15"],
            "dateendrole": ["2020-10-01", None],
            "isin": ["US1234567890", "US1234567890"],
        }
    )
    events = _make_events(roles, [2020])
    assert set(events["event_type"]) == {"appointment", "departure"}
    assert events["title"].str.contains("Chief Executive Officer").all()
