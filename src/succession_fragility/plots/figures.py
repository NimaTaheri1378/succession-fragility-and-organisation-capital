from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns


STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "font.size": 10,
}

DISPLAY_LABELS = {
    "key_person_concentration": "Key-person\nconcentration",
    "succession_depth_gap": "Succession\ndepth gap",
    "external_load": "External\nload",
    "bench_depth": "Bench\ndepth",
    "team_cohesion_decay": "Cohesion\ndecay",
}

MODEL_LABELS = {
    "elastic_net": "Elastic Net",
    "lightgbm": "LightGBM",
    "hist_gradient_boosting_fallback": "HistGB fallback",
    "deep_sets_component_proxy_cuda": "Deep Sets proxy (GPU)",
    "deep_sets_component_proxy_cpu": "Deep Sets proxy",
    "executive_deep_sets": "Executive Deep Sets",
    "conditional_autoencoder_sdf_cuda": "SDF autoencoder (GPU)",
    "conditional_autoencoder_sdf_cpu": "SDF autoencoder",
}


def set_style() -> None:
    sns.set_theme(style="whitegrid", rc=STYLE)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_long_short(ls_returns: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = ls_returns.dropna(subset=["long_short"]).copy()
    df["cum_long_short"] = (1 + df["long_short"]).cumprod() - 1
    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    ax.plot(df["date"], df["cum_long_short"], color="#1f6f8b", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("OCF Long-Short Cumulative Excess Return")
    ax.set_xlabel("Portfolio month")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    savefig(output)
    return output


def plot_rank_ic(rank_ic: pd.DataFrame, output: Path) -> Path:
    set_style()
    plt.figure(figsize=(7, 4))
    plt.bar(rank_ic["horizon"].astype(str), rank_ic["spearman_ic"], color="#8c6d31", width=0.55)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Rank IC Decay")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Spearman IC")
    savefig(output)
    return output


def plot_feature_heatmap(features: pd.DataFrame, output: Path) -> Path:
    set_style()
    cols = [
        "key_person_concentration",
        "succession_depth_gap",
        "external_load",
        "bench_depth",
        "team_cohesion_decay",
    ]
    corr = features[cols].corr().rename(index=DISPLAY_LABELS, columns=DISPLAY_LABELS)
    plt.figure(figsize=(7.2, 5.6))
    ax = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.78, "label": "Correlation"},
        annot_kws={"fontsize": 9},
    )
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    plt.title("OCF Component Correlations")
    savefig(output)
    return output


def plot_event_paths(paths: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = paths.dropna(subset=["event_day", "mean_car", "ocf_bucket"]).copy()
    plt.figure(figsize=(8, 4.6))
    ax = plt.gca()
    for bucket, group in df.groupby("ocf_bucket", observed=True):
        label = f"Q{int(bucket)}" if pd.notna(bucket) else "Missing"
        ax.plot(group["event_day"], group["mean_car"], linewidth=1.9, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Leadership-Change Event Returns by Pre-Event OCF")
    ax.set_xlabel("Trading days after event")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="OCF bucket", ncol=5, frameon=False, loc="best")
    savefig(output)
    return output


def plot_rolling_alpha(rolling: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = rolling.dropna(subset=["date", "alpha_ann"]).copy()
    plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()
    ax.plot(df["date"], df["alpha_ann"], color="#1f6f8b", linewidth=1.8, label="Rolling alpha")
    if "t_alpha" in df:
        sig = df["t_alpha"].abs() >= 1.96
        ax.scatter(df.loc[sig, "date"], df.loc[sig, "alpha_ann"], s=16, color="#b23a48", label="|t| >= 1.96")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rolling FF5+UMD Alpha")
    ax.set_xlabel("Window end")
    ax.set_ylabel("Annualized alpha")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(frameon=False)
    savefig(output)
    return output


def plot_turnover_cost_frontier(frontier: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = frontier.dropna(subset=["one_way_bps", "alpha_ann"]).copy()
    plt.figure(figsize=(7.2, 4.2))
    ax = plt.gca()
    ax.plot(df["one_way_bps"], df["alpha_ann"], marker="o", color="#1f6f8b", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Turnover-Cost Frontier")
    ax.set_xlabel("One-way cost assumption (bps)")
    ax.set_ylabel("Annualized net return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    savefig(output)
    return output


def plot_model_comparison(metrics: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = metrics.dropna(subset=["model", "rank_ic_mean"]).copy().sort_values("rank_ic_mean")
    df["model_label"] = df["model"].map(lambda value: MODEL_LABELS.get(str(value), str(value)))
    plt.figure(figsize=(8, 4.4))
    ax = plt.gca()
    ax.barh(df["model_label"], df["rank_ic_mean"], color="#1f6f8b")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Walk-Forward Model Rank IC")
    ax.set_xlabel("Mean annual test-block Spearman IC")
    ax.set_ylabel("")
    savefig(output)
    return output


def plot_feature_importance(importance: pd.DataFrame, output: Path, title: str = "Model Feature Importance") -> Path:
    set_style()
    df = importance.dropna(subset=["feature", "importance"]).copy()
    df = df.groupby("feature", observed=True)["importance"].mean().sort_values().tail(15).reset_index()
    plt.figure(figsize=(7.6, 5.0))
    ax = plt.gca()
    ax.barh(df["feature"], df["importance"], color="#4f7cac")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    savefig(output)
    return output


def plot_partial_dependence(pd_frame: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = pd_frame.dropna(subset=["ocf_bin_mid", "mean_prediction"]).copy()
    plt.figure(figsize=(7.4, 4.2))
    ax = plt.gca()
    ax.plot(df["ocf_bin_mid"], df["mean_prediction"], marker="o", color="#1f6f8b")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Partial Dependence of Predicted Returns on OCF")
    ax.set_xlabel("OCF bin midpoint")
    ax.set_ylabel("Predicted next-month excess return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    savefig(output)
    return output


def plot_regime_timeline(regimes: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = regimes.copy()
    plt.figure(figsize=(8.4, 4.4))
    ax = plt.gca()
    ax.plot(df["date"], df["vix"], color="#1f6f8b", linewidth=1.4, label="VIX")
    if "nber_recession" in df:
        ymin, ymax = ax.get_ylim()
        for _, row in df[df["nber_recession"].eq(1)].iterrows():
            ax.axvspan(row["date"] - pd.offsets.MonthBegin(1), row["date"], color="#b23a48", alpha=0.12, linewidth=0)
        ax.set_ylim(ymin, ymax)
    ax.set_title("Macro Regime Controls")
    ax.set_xlabel("Month")
    ax.set_ylabel("VIX level")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(frameon=False)
    savefig(output)
    return output


def plot_shap_like_summary(values: pd.DataFrame, output: Path) -> Path:
    set_style()
    df = values.dropna(subset=["feature", "mean_abs_contribution"]).copy()
    df = df.sort_values("mean_abs_contribution").tail(15)
    plt.figure(figsize=(7.6, 5.0))
    ax = plt.gca()
    colors = np.where(df.get("mean_signed_contribution", 0) >= 0, "#1f6f8b", "#b23a48")
    ax.barh(df["feature"], df["mean_abs_contribution"], color=colors)
    ax.set_title("SHAP / Contribution Summary")
    ax.set_xlabel("Mean absolute contribution")
    ax.set_ylabel("")
    savefig(output)
    return output
