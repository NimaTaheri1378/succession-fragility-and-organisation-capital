from __future__ import annotations

import pandas as pd

from succession_fragility.features.ocf import add_ocf_score, build_team_features, role_weight


def test_role_weight_prioritizes_core_officers() -> None:
    assert role_weight("Chief Executive Officer") > role_weight("Director")
    assert role_weight("Chief Financial Officer") > role_weight("General Counsel")


def test_ocf_components_are_firm_month_unique() -> None:
    rows = [
        {
            "gvkey": "001000",
            "date": "2020-01-31",
            "person_id": "a",
            "title": "CEO",
            "role_start_date": "2017-01-01",
            "role_end_date": pd.NaT,
            "outside_roles": 2,
            "prior_employers": 4,
            "internal_candidate": 0,
        },
        {
            "gvkey": "001000",
            "date": "2020-01-31",
            "person_id": "b",
            "title": "CFO",
            "role_start_date": "2018-01-01",
            "role_end_date": pd.NaT,
            "outside_roles": 0,
            "prior_employers": 2,
            "internal_candidate": 1,
        },
        {
            "gvkey": "001001",
            "date": "2020-01-31",
            "person_id": "c",
            "title": "CEO",
            "role_start_date": "2019-01-01",
            "role_end_date": pd.NaT,
            "outside_roles": 1,
            "prior_employers": 1,
            "internal_candidate": 0,
        },
    ]
    features = add_ocf_score(build_team_features(pd.DataFrame(rows)))
    assert features[["gvkey", "date"]].duplicated().sum() == 0
    assert "ocf" in features
    assert features["key_person_concentration"].between(0, 1).all()
