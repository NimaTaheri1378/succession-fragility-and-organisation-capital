from __future__ import annotations

import numpy as np
import pandas as pd


ROLE_CRITICALITY = {
    "ceo": 1.00,
    "chief executive officer": 1.00,
    "cfo": 0.85,
    "chief financial officer": 0.85,
    "coo": 0.80,
    "chief operating officer": 0.80,
    "president": 0.75,
    "cto": 0.70,
    "chief technology officer": 0.70,
    "general counsel": 0.55,
    "director": 0.25,
}


def role_weight(title: object) -> float:
    text = str(title or "").lower()
    for key, value in ROLE_CRITICALITY.items():
        if key in text:
            return value
    return 0.40


def zscore_by_date(df: pd.DataFrame, column: str, date_col: str = "date") -> pd.Series:
    grouped = df.groupby(date_col, observed=True)[column]
    mu = grouped.transform("mean")
    sigma = grouped.transform("std").replace(0, np.nan)
    return ((df[column] - mu) / sigma).fillna(0.0)


def build_team_features(person_month: pd.DataFrame) -> pd.DataFrame:
    """Build firm-month OCF components from person-month executive states.

    Required columns: gvkey, date, person_id, title, role_start_date, role_end_date,
    outside_roles, prior_employers, internal_candidate.
    """

    df = person_month.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["role_start_date"] = pd.to_datetime(df["role_start_date"], errors="coerce")
    df["role_weight"] = df["title"].map(role_weight)
    df["tenure_years"] = ((df["date"] - df["role_start_date"]).dt.days / 365.25).clip(lower=0)
    df["hc_share_raw"] = df["role_weight"] * np.log1p(df["tenure_years"])

    key = ["gvkey", "date"]
    denom = df.groupby(key, observed=True)["hc_share_raw"].transform("sum").replace(0, np.nan)
    df["hc_share"] = (df["hc_share_raw"] / denom).fillna(0.0)
    df["hc_share_sq"] = df["hc_share"] ** 2

    team = (
        df.groupby(key, observed=True)
        .agg(
            team_size=("person_id", "nunique"),
            key_person_concentration=("hc_share_sq", "sum"),
            mean_tenure=("tenure_years", "mean"),
            external_load=("outside_roles", "mean"),
            poaching_pressure=("prior_employers", "mean"),
            bench_depth=("internal_candidate", "sum"),
        )
        .reset_index()
    )
    team["succession_depth_gap"] = (3.0 - team["bench_depth"]).clip(lower=0)
    team["team_cohesion_decay"] = (
        team.sort_values(key)
        .groupby("gvkey", observed=True)["mean_tenure"]
        .diff()
        .mul(-1)
        .clip(lower=0)
        .fillna(0.0)
    )
    return team


def add_ocf_score(team_features: pd.DataFrame) -> pd.DataFrame:
    out = team_features.copy()
    components = [
        "key_person_concentration",
        "succession_depth_gap",
        "external_load",
        "poaching_pressure",
        "team_cohesion_decay",
    ]
    for col in components + ["bench_depth"]:
        out[f"z_{col}"] = zscore_by_date(out, col)
    out["ocf"] = (
        out["z_key_person_concentration"]
        + out["z_succession_depth_gap"]
        + out["z_external_load"]
        + out["z_poaching_pressure"]
        + out["z_team_cohesion_decay"]
        - out["z_bench_depth"]
    )
    return out
