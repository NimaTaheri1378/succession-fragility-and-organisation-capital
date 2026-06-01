from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import long_short_returns, performance_summary
from succession_fragility.features.ocf import add_ocf_score, build_team_features
from succession_fragility.labels.returns import add_forward_monthly_returns
from succession_fragility.models.fama_macbeth import fama_macbeth, nw_summary
from succession_fragility.plots.figures import (
    plot_feature_heatmap,
    plot_long_short,
    plot_rank_ic,
)
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest


def make_synthetic_person_months(
    n_firms: int = 120,
    n_months: int = 72,
    seed: int = 20260601,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    rows: list[dict[str, object]] = []
    titles = ["CEO", "CFO", "COO", "President", "CTO", "General Counsel", "Director"]
    for firm in range(n_firms):
        gvkey = f"{firm + 1000:06d}"
        base_team = rng.integers(4, 10)
        for date in dates:
            team_size = max(3, int(base_team + rng.normal(0, 1)))
            for j in range(team_size):
                title = titles[min(j, len(titles) - 1)]
                start_offset = int(rng.integers(30, 3650))
                rows.append(
                    {
                        "gvkey": gvkey,
                        "date": date,
                        "person_id": f"{gvkey}-{j}",
                        "title": title,
                        "role_start_date": date - pd.Timedelta(days=start_offset),
                        "role_end_date": pd.NaT,
                        "outside_roles": int(rng.poisson(0.6)),
                        "prior_employers": int(rng.poisson(2.0)),
                        "internal_candidate": int(j > 1 and rng.random() > 0.35),
                    }
                )
    return pd.DataFrame(rows)


def make_synthetic_returns(features: pd.DataFrame, seed: int = 20260601) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    panel = features.copy()
    firm_codes = {gvkey: i + 10_000 for i, gvkey in enumerate(sorted(panel["gvkey"].unique()))}
    panel["permno"] = panel["gvkey"].map(firm_codes)
    panel["mktcap"] = np.exp(rng.normal(8.5, 1.1, size=len(panel)))
    panel["size"] = np.log(panel["mktcap"])
    panel["bm"] = rng.normal(0.55, 0.18, size=len(panel))
    panel["momentum"] = rng.normal(0.05, 0.22, size=len(panel))
    panel["profitability"] = rng.normal(0.08, 0.07, size=len(panel))
    panel["investment"] = rng.normal(0.04, 0.08, size=len(panel))
    noise = rng.normal(0, 0.075, size=len(panel))
    panel["ret_excess"] = 0.003 + 0.0015 * panel["ocf"] + 0.006 * panel["momentum"] + noise
    return add_forward_monthly_returns(panel)


def run_synthetic(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    person_month = make_synthetic_person_months()
    features = add_ocf_score(build_team_features(person_month))
    panel = make_synthetic_returns(features)

    controls = ["ocf", "size", "bm", "momentum", "profitability", "investment"]
    fm = fama_macbeth(panel, "ret_excess_fwd1m", controls, min_obs=50)
    fm_summary = nw_summary(fm)
    ls = long_short_returns(panel, signal="ocf")
    perf = performance_summary(ls["long_short"])

    rank_ic = pd.DataFrame(
        {
            "horizon": ["1m", "20d", "60d"],
            "spearman_ic": [
                panel[["ocf", "ret_excess_fwd1m"]].corr(method="spearman").iloc[0, 1],
                0.6 * panel[["ocf", "ret_excess_fwd1m"]].corr(method="spearman").iloc[0, 1],
                0.35 * panel[["ocf", "ret_excess_fwd1m"]].corr(method="spearman").iloc[0, 1],
            ],
        }
    )

    fm_summary.to_csv(table_dir / "synthetic_fama_macbeth.csv", index=False)
    ls.to_csv(table_dir / "synthetic_long_short.csv", index=False)
    pd.DataFrame([perf]).to_csv(table_dir / "synthetic_performance.csv", index=False)

    plot_long_short(ls, figure_dir / "synthetic_ocf_long_short.png")
    plot_rank_ic(rank_ic, figure_dir / "synthetic_rank_ic_decay.png")
    plot_feature_heatmap(features, figure_dir / "synthetic_ocf_component_heatmap.png")
    qa = inspect_figure_dir(figure_dir)

    manifest = {
        "kind": "synthetic_end_to_end",
        "n_person_month_rows": int(len(person_month)),
        "n_firm_month_rows": int(len(panel)),
        "fama_macbeth_months": int(len(fm)),
        "performance": perf,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "synthetic_run.json", manifest)
    return manifest
