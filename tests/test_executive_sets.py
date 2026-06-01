from __future__ import annotations

import numpy as np
import pandas as pd

from succession_fragility.pipeline.executive_sets import _expand_roles_for_year


def _roles() -> pd.DataFrame:
    roles = pd.DataFrame(
        [
            {
                "boardex_companyid": "100",
                "person_id": "p1",
                "title": "CEO",
                "role_weight": 1.0,
                "role_start_date": pd.Timestamp("2019-01-01"),
                "role_end_date": pd.Timestamp("2020-12-31"),
                "leadership_flag": True,
            },
            {
                "boardex_companyid": "200",
                "person_id": "p1",
                "title": "CFO",
                "role_weight": 0.8,
                "role_start_date": pd.Timestamp("2021-01-01"),
                "role_end_date": pd.Timestamp("2021-12-31"),
                "leadership_flag": True,
            },
        ]
    )
    roles.attrs["person_company_first_starts"] = {
        "p1": (
            np.array(["2019-01-01", "2021-01-01"], dtype="datetime64[ns]"),
            np.array(["100", "200"]),
        )
    }
    return roles


def test_prior_employers_excludes_future_roles() -> None:
    expanded = _expand_roles_for_year(_roles(), 2020)
    row = expanded.loc[expanded["boardex_companyid"].eq("100")].iloc[-1]

    assert row["prior_employers"] == 0


def test_prior_employers_includes_only_known_previous_companies() -> None:
    expanded = _expand_roles_for_year(_roles(), 2021)
    row = expanded.loc[expanded["boardex_companyid"].eq("200")].iloc[0]

    assert row["prior_employers"] == 1
