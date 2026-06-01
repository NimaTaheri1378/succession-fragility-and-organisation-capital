from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import assign_quantiles
from succession_fragility.utils.manifest import write_manifest


COMPONENTS = [
    "key_person_concentration",
    "succession_depth_gap",
    "external_load",
    "poaching_pressure",
    "bench_depth",
    "team_cohesion_decay",
]

DISPLAY_LABELS = {
    "key_person_concentration": "KPC",
    "succession_depth_gap": "Depth gap",
    "external_load": "External load",
    "poaching_pressure": "Poaching pressure",
    "bench_depth": "Bench depth",
    "team_cohesion_decay": "Cohesion decay",
    "elastic_net": "Elastic Net",
    "lightgbm": "LightGBM",
    "deep_sets_component_proxy_cuda": "DS proxy",
    "deep_sets_component_proxy_cpu": "DS proxy",
    "executive_deep_sets": "Exec Sets",
    "conditional_autoencoder_sdf_cuda": "SDF",
    "conditional_autoencoder_sdf_cpu": "SDF",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _aggregate_panel(panel_path: Path, output_dir: Path) -> dict[str, Path]:
    cols = ["date", "siccd", "ocf", "ret_excess_fwd1m", "mktcap", *COMPONENTS]
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(panel_path).schema_arrow.names)
    except Exception:
        available = set(cols)
    panel = pd.read_parquet(panel_path, columns=[c for c in cols if c in available]).replace([np.inf, -np.inf], np.nan)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    panel["sic2"] = pd.to_numeric(panel.get("siccd"), errors="coerce").floordiv(100).astype("Int64")
    panel["ocf_bucket"] = assign_quantiles(panel, "ocf", q=5)
    industry = (
        panel.dropna(subset=["sic2", "ocf_bucket"])
        .groupby(["year", "sic2", "ocf_bucket"], observed=True)
        .agg(
            n_obs=("ocf", "size"),
            mean_ocf=("ocf", "mean"),
            mean_next_excess_return=("ret_excess_fwd1m", "mean"),
            total_mktcap=("mktcap", "sum"),
        )
        .reset_index()
    )
    component_cols = [c for c in COMPONENTS if c in panel]
    components = panel.groupby("year", observed=True)[component_cols].mean().reset_index() if component_cols else pd.DataFrame()
    paths = {
        "industry_bucket": output_dir / "industry_bucket_aggregates.csv",
        "component_trends": output_dir / "component_trends.csv",
    }
    industry.to_csv(paths["industry_bucket"], index=False)
    components.to_csv(paths["component_trends"], index=False)
    return paths


def build_public_dashboard(
    panel_path: Path,
    source_root: Path,
    output_dir: Path,
    event_root: Path | None = None,
    model_root: Path | None = None,
    executive_root: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    aggregate_paths = _aggregate_panel(panel_path, table_dir)
    long_short = _read_csv(source_root / "tables" / "ocf_long_short.csv")
    event_paths = _read_csv((event_root or source_root.parent / "event_study") / "tables" / "event_path_by_ocf_bucket.csv")
    model_comparison = _read_csv((model_root or source_root.parent / "model_ladder") / "tables" / "model_comparison.csv")
    executive_comparison = _read_csv(
        (executive_root or source_root.parent / "executive_deep_sets") / "tables" / "executive_deep_sets_comparison.csv"
    )
    if not executive_comparison.empty:
        for col in ["oos_r2_mean", "mse_mean"]:
            if col not in executive_comparison:
                executive_comparison[col] = np.nan
        model_comparison = pd.concat([model_comparison, executive_comparison[model_comparison.columns]], ignore_index=True)
    components = _read_csv(aggregate_paths["component_trends"])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("plotly is required to build the public dashboard.") from exc

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "OCF Long-Short Cumulative Return",
            "Event Paths by Pre-Event OCF",
            "Component Trends",
            "Walk-Forward Model Rank IC",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )

    if not long_short.empty:
        long_short["date"] = pd.to_datetime(long_short["date"])
        long_short["cum_long_short"] = (1 + long_short["long_short"].fillna(0)).cumprod() - 1
        fig.add_trace(
            go.Scatter(x=long_short["date"], y=long_short["cum_long_short"], mode="lines", name="OCF spread"),
            row=1,
            col=1,
        )
    if not event_paths.empty:
        for bucket, group in event_paths.groupby("ocf_bucket", observed=True):
            fig.add_trace(
                go.Scatter(
                    x=group["event_day"],
                    y=group["mean_car"],
                    mode="lines",
                    name=f"Event Q{int(bucket)}",
                    showlegend=True,
                ),
                row=1,
                col=2,
            )
    if not components.empty:
        for col in [c for c in COMPONENTS if c in components]:
            fig.add_trace(
                go.Scatter(x=components["year"], y=components[col], mode="lines", name=DISPLAY_LABELS.get(col, col)),
                row=2,
                col=1,
            )
    if not model_comparison.empty:
        model_comparison = model_comparison.copy()
        model_comparison["model_label"] = model_comparison["model"].map(lambda x: DISPLAY_LABELS.get(str(x), str(x)))
        fig.add_trace(
            go.Bar(
                x=model_comparison["rank_ic_mean"],
                y=model_comparison["model_label"],
                orientation="h",
                name="Rank IC",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        template="plotly_white",
        width=1200,
        height=820,
        title="Succession Fragility and Organisation Capital: Public Aggregate Dashboard",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=90, b=110),
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Trading day", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Mean Rank IC", row=2, col=2)

    html_path = output_dir / "public_aggregate_dashboard.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    manifest = {
        "kind": "public_dashboard",
        "status": "ok",
        "panel_path": str(panel_path),
        "source_root": str(source_root),
        "event_root": str(event_root) if event_root else None,
        "model_root": str(model_root) if model_root else None,
        "executive_root": str(executive_root) if executive_root else None,
        "html_path": str(html_path),
        "aggregate_tables": {key: str(path) for key, path in aggregate_paths.items()},
        "public_outputs_aggregate_only": True,
    }
    write_manifest(manifest_dir / "public_dashboard.json", manifest)
    return manifest
