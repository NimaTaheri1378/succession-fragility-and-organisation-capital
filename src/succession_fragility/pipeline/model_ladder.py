from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import long_short_returns, performance_summary
from succession_fragility.models.deep_sets import DeepSetConfig, build_deep_set_model
from succession_fragility.plots.figures import (
    plot_feature_importance,
    plot_model_comparison,
    plot_partial_dependence,
    plot_shap_like_summary,
)
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for the model ladder.") from exc


TABULAR_FEATURES = [
    "ocf",
    "team_size",
    "key_person_concentration",
    "succession_depth_gap",
    "external_load",
    "poaching_pressure",
    "bench_depth",
    "team_cohesion_decay",
    "size",
    "bm",
    "momentum",
    "profitability",
    "investment",
    "leverage",
    "intangibility",
    "analyst_coverage",
    "forecast_dispersion",
]

COMPONENT_SET_FEATURES = [
    "key_person_concentration",
    "succession_depth_gap",
    "external_load",
    "poaching_pressure",
    "bench_depth",
    "team_cohesion_decay",
]


@dataclass
class PredictionBlock:
    model: str
    frame: pd.DataFrame
    feature_importance: pd.DataFrame | None = None
    last_estimator: Any | None = None
    last_test_x: pd.DataFrame | None = None


def _rank_ic(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 5:
        return np.nan
    if spearmanr is not None:
        return float(spearmanr(y[mask], yhat[mask]).statistic)
    return float(pd.Series(y[mask]).corr(pd.Series(yhat[mask]), method="spearman"))


def _oos_r2(y: np.ndarray, yhat: np.ndarray, benchmark: float) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 5:
        return np.nan
    denom = np.square(y[mask] - benchmark).sum()
    if denom <= 0:
        return np.nan
    return float(1.0 - np.square(y[mask] - yhat[mask]).sum() / denom)


def _load_panel(panel_path: Path) -> pd.DataFrame:
    cols = [
        "date",
        "permno",
        "ret_excess_fwd1m",
        "downside_20d",
        "mktcap",
        *TABULAR_FEATURES,
    ]
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(panel_path).schema_arrow.names)
    except Exception:
        available = set(pd.read_parquet(panel_path, columns=[]).columns)
    panel = pd.read_parquet(panel_path, columns=[c for c in cols if c in available])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    for col in panel.columns:
        if col not in {"date"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel.dropna(subset=["ret_excess_fwd1m"]).copy()


def _available_features(panel: pd.DataFrame) -> list[str]:
    return [c for c in TABULAR_FEATURES if c in panel and panel[c].notna().sum() > 100]


def _lightgbm_or_fallback(random_state: int, n_jobs: int):
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=250,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            min_child_samples=80,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        ), "lightgbm"
    except Exception:
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.035, l2_regularization=0.01, random_state=random_state), "hist_gradient_boosting_fallback"


def _walk_forward_tabular(
    panel: pd.DataFrame,
    features: list[str],
    model_name: str,
    estimator: Any,
    test_start_year: int,
) -> PredictionBlock:
    frames: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    last_estimator = None
    last_test_x = None
    years = sorted(y for y in panel["date"].dt.year.unique() if y >= test_start_year)
    for year in years:
        train = panel[panel["date"].dt.year < year].copy()
        test = panel[panel["date"].dt.year == year].copy()
        if len(train) < 1000 or len(test) < 100:
            continue
        x_train = train[features]
        y_train = train["ret_excess_fwd1m"].to_numpy(dtype=float)
        x_test = test[features]
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler() if model_name == "elastic_net" else "passthrough"),
                ("model", estimator),
            ]
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        frames.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "permno": test["permno"].to_numpy(),
                    "score": pred,
                    "ret_excess_fwd1m": test["ret_excess_fwd1m"].to_numpy(dtype=float),
                    "mktcap": test.get("mktcap", pd.Series(np.nan, index=test.index)).to_numpy(dtype=float),
                    "test_year": year,
                }
            )
        )
        model = pipe.named_steps["model"]
        if hasattr(model, "coef_"):
            importances.append(pd.DataFrame({"feature": features, "importance": np.abs(model.coef_), "test_year": year}))
        elif hasattr(model, "feature_importances_"):
            importances.append(pd.DataFrame({"feature": features, "importance": model.feature_importances_, "test_year": year}))
        last_estimator = pipe
        last_test_x = x_test
    return PredictionBlock(
        model=model_name,
        frame=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        feature_importance=pd.concat(importances, ignore_index=True) if importances else None,
        last_estimator=last_estimator,
        last_test_x=last_test_x,
    )


def _component_set_array(panel: pd.DataFrame, features: list[str], medians: pd.Series, scales: pd.Series) -> np.ndarray:
    values = panel[features].copy().fillna(medians)
    scaled = (values - medians) / scales.replace(0, 1.0)
    tokens = []
    ids = np.linspace(0.0, 1.0, len(features), dtype=np.float32)
    for idx, feature in enumerate(features):
        tokens.append(np.column_stack([scaled[feature].to_numpy(dtype=np.float32), np.full(len(panel), ids[idx], dtype=np.float32)]))
    return np.stack(tokens, axis=1)


def _torch_predict(model, loader, device) -> np.ndarray:
    import torch

    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _yb, mask in loader:
            out = model(xb.to(device), mask.to(device))
            preds.append(out[:, 0].detach().cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


def _neural_component_set(
    panel: pd.DataFrame,
    test_start_year: int,
    epochs: int,
    batch_size: int,
    use_cuda: bool,
    seed: int,
) -> PredictionBlock:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return PredictionBlock(model="deep_sets_component_proxy_unavailable", frame=pd.DataFrame())

    features = [c for c in COMPONENT_SET_FEATURES if c in panel and panel[c].notna().sum() > 100]
    if len(features) < 3:
        return PredictionBlock(model="deep_sets_component_proxy_no_features", frame=pd.DataFrame())
    train = panel[panel["date"].dt.year < test_start_year].copy()
    test = panel[panel["date"].dt.year >= test_start_year].copy()
    if len(train) < 1000 or len(test) < 100:
        return PredictionBlock(model="deep_sets_component_proxy_no_sample", frame=pd.DataFrame())
    med = train[features].median()
    scale = train[features].std().replace(0, 1.0)
    x_train = _component_set_array(train, features, med, scale)
    x_test = _component_set_array(test, features, med, scale)
    y_train_raw = train["ret_excess_fwd1m"].to_numpy(dtype=np.float32)
    y_test = test["ret_excess_fwd1m"].to_numpy(dtype=np.float32)
    y_mean = float(np.nanmean(y_train_raw))
    y_std = float(np.nanstd(y_train_raw))
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    y_train = np.clip((y_train_raw - y_mean) / y_std, -10.0, 10.0).astype(np.float32)
    mask_train = np.ones(x_train.shape[:2], dtype=np.float32)
    mask_test = np.ones(x_test.shape[:2], dtype=np.float32)
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = build_deep_set_model(DeepSetConfig(n_features=2, hidden_dim=128, dropout=0.10)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(mask_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb, mask in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb.to(device), mask.to(device))[:, 0]
            loss = loss_fn(pred, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(mask_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    pred = _torch_predict(model, test_loader, device) * y_std + y_mean
    frame = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "permno": test["permno"].to_numpy(),
            "score": pred,
            "ret_excess_fwd1m": test["ret_excess_fwd1m"].to_numpy(dtype=float),
            "mktcap": test.get("mktcap", pd.Series(np.nan, index=test.index)).to_numpy(dtype=float),
            "test_year": test["date"].dt.year.to_numpy(),
        }
    )
    return PredictionBlock(model=f"deep_sets_component_proxy_{device.type}", frame=frame)


def _neural_bottleneck_sdf(
    panel: pd.DataFrame,
    features: list[str],
    test_start_year: int,
    epochs: int,
    batch_size: int,
    use_cuda: bool,
    seed: int,
) -> PredictionBlock:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return PredictionBlock(model="conditional_autoencoder_sdf_unavailable", frame=pd.DataFrame())

    train = panel[panel["date"].dt.year < test_start_year].copy()
    test = panel[panel["date"].dt.year >= test_start_year].copy()
    if len(train) < 1000 or len(test) < 100:
        return PredictionBlock(model="conditional_autoencoder_sdf_no_sample", frame=pd.DataFrame())
    med = train[features].median()
    scale = train[features].std().replace(0, 1.0)
    x_train = ((train[features].fillna(med) - med) / scale).to_numpy(dtype=np.float32)
    x_test = ((test[features].fillna(med) - med) / scale).to_numpy(dtype=np.float32)
    y_train_raw = train["ret_excess_fwd1m"].to_numpy(dtype=np.float32)
    y_test = test["ret_excess_fwd1m"].to_numpy(dtype=np.float32)
    y_mean = float(np.nanmean(y_train_raw))
    y_std = float(np.nanstd(y_train_raw))
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    y_train = np.clip((y_train_raw - y_mean) / y_std, -10.0, 10.0).astype(np.float32)
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    class BottleneckSDF(nn.Module):
        def __init__(self, n_features: int, latent_dim: int = 4) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = BottleneckSDF(len(features)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            preds.append(model(torch.from_numpy(x_test[start : start + batch_size]).to(device)).cpu().numpy())
    pred = np.concatenate(preds) * y_std + y_mean
    frame = pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "permno": test["permno"].to_numpy(),
            "score": pred,
            "ret_excess_fwd1m": y_test.astype(float),
            "mktcap": test.get("mktcap", pd.Series(np.nan, index=test.index)).to_numpy(dtype=float),
            "test_year": test["date"].dt.year.to_numpy(),
        }
    )
    return PredictionBlock(model=f"conditional_autoencoder_sdf_{device.type}", frame=frame)


def _metrics_for_block(block: PredictionBlock) -> pd.DataFrame:
    if block.frame.empty:
        return pd.DataFrame()
    rows = []
    for year, group in block.frame.groupby("test_year", observed=True):
        y = group["ret_excess_fwd1m"].to_numpy(dtype=float)
        score = group["score"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(score)
        if mask.sum() < 5:
            continue
        rows.append(
            {
                "model": block.model,
                "test_year": int(year),
                "n_obs": int(mask.sum()),
                "rank_ic": _rank_ic(y[mask], score[mask]),
                "oos_r2": _oos_r2(y[mask], score[mask], benchmark=0.0),
                "mse": float(mean_squared_error(y[mask], score[mask])),
            }
        )
    return pd.DataFrame(rows)


def _prediction_portfolio(block: PredictionBlock) -> pd.DataFrame:
    if block.frame.empty:
        return pd.DataFrame()
    return long_short_returns(block.frame, signal="score", ret_col="ret_excess_fwd1m", q=5)


def _partial_dependence(block: PredictionBlock, features: list[str]) -> pd.DataFrame:
    if block.last_estimator is None or block.last_test_x is None or "ocf" not in features:
        return pd.DataFrame()
    sample = block.last_test_x.sample(min(5000, len(block.last_test_x)), random_state=20260601).copy()
    qs = np.linspace(0.05, 0.95, 15)
    grid = sample["ocf"].quantile(qs).drop_duplicates()
    rows = []
    for value in grid:
        tmp = sample.copy()
        tmp["ocf"] = value
        rows.append({"ocf_bin_mid": float(value), "mean_prediction": float(np.mean(block.last_estimator.predict(tmp)))})
    return pd.DataFrame(rows)


def _shap_or_contribution(block: PredictionBlock, features: list[str]) -> pd.DataFrame:
    if block.last_estimator is None or block.last_test_x is None:
        return pd.DataFrame()
    sample = block.last_test_x.sample(min(3000, len(block.last_test_x)), random_state=20260601).copy()
    try:
        import shap

        model = block.last_estimator.named_steps["model"]
        transformed = block.last_estimator[:-1].transform(sample)
        explainer = shap.Explainer(model)
        values = explainer(transformed)
        arr = np.asarray(values.values)
        return pd.DataFrame(
            {
                "feature": features,
                "mean_abs_contribution": np.mean(np.abs(arr), axis=0),
                "mean_signed_contribution": np.mean(arr, axis=0),
                "source": "shap",
            }
        )
    except Exception:
        baseline = sample.copy()
        pred = block.last_estimator.predict(baseline)
        rows = []
        for feature in features:
            shuffled = baseline.copy()
            shuffled[feature] = shuffled[feature].sample(frac=1, random_state=20260601).to_numpy()
            delta = pred - block.last_estimator.predict(shuffled)
            rows.append(
                {
                    "feature": feature,
                    "mean_abs_contribution": float(np.mean(np.abs(delta))),
                    "mean_signed_contribution": float(np.mean(delta)),
                    "source": "permutation_contribution",
                }
            )
        return pd.DataFrame(rows)


def run_model_ladder(
    panel_path: Path,
    output_dir: Path,
    test_start_year: int = 2015,
    neural_test_start_year: int = 2018,
    epochs_deep: int = 8,
    epochs_sdf: int = 8,
    batch_size: int = 4096,
    use_cuda: bool = True,
    n_jobs: int = 8,
    seed: int = 20260601,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, figure_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(panel_path)
    features = _available_features(panel)
    elastic = ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=5000, random_state=seed)
    lgbm_estimator, lgbm_name = _lightgbm_or_fallback(seed, n_jobs)
    blocks = [
        _walk_forward_tabular(panel, features, "elastic_net", elastic, test_start_year),
        _walk_forward_tabular(panel, features, lgbm_name, lgbm_estimator, test_start_year),
        _neural_component_set(panel, neural_test_start_year, epochs_deep, batch_size, use_cuda, seed),
        _neural_bottleneck_sdf(panel, features, neural_test_start_year, epochs_sdf, batch_size, use_cuda, seed),
    ]
    metrics = pd.concat([_metrics_for_block(block) for block in blocks], ignore_index=True)
    metrics.to_csv(table_dir / "walk_forward_metrics_by_year.csv", index=False)
    comparison = (
        metrics.groupby("model", observed=True)
        .agg(
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_t=("rank_ic", lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std(ddof=1) else np.nan),
            oos_r2_mean=("oos_r2", "mean"),
            mse_mean=("mse", "mean"),
            n_test_years=("test_year", "nunique"),
            n_obs=("n_obs", "sum"),
        )
        .reset_index()
    )
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)
    if not comparison.empty:
        plot_model_comparison(comparison, figure_dir / "model_comparison_rank_ic.png")

    port_rows = []
    port_series = []
    for block in blocks:
        ls = _prediction_portfolio(block)
        if ls.empty:
            continue
        perf = performance_summary(ls["long_short"])
        port_rows.append({"model": block.model, **perf})
        port_series.append(ls.assign(model=block.model))
    pd.DataFrame(port_rows).to_csv(table_dir / "prediction_portfolio_performance.csv", index=False)
    if port_series:
        pd.concat(port_series, ignore_index=True).to_csv(table_dir / "prediction_long_short_returns.csv", index=False)

    importances = pd.concat(
        [block.feature_importance.assign(model=block.model) for block in blocks if block.feature_importance is not None],
        ignore_index=True,
    ) if any(block.feature_importance is not None for block in blocks) else pd.DataFrame()
    importances.to_csv(table_dir / "feature_importance.csv", index=False)
    if not importances.empty:
        plot_feature_importance(importances, figure_dir / "feature_importance.png")

    best_block = next((b for b in blocks if b.model == lgbm_name and b.last_estimator is not None), blocks[0])
    pd_frame = _partial_dependence(best_block, features)
    pd_frame.to_csv(table_dir / "partial_dependence_ocf.csv", index=False)
    if not pd_frame.empty:
        plot_partial_dependence(pd_frame, figure_dir / "partial_dependence_ocf.png")
    contrib = _shap_or_contribution(best_block, features)
    contrib.to_csv(table_dir / "shap_or_contribution_summary.csv", index=False)
    if not contrib.empty:
        plot_shap_like_summary(contrib, figure_dir / "shap_summary.png")

    qa = inspect_figure_dir(figure_dir)
    manifest = {
        "kind": "model_ladder",
        "status": "ok",
        "panel_path": str(panel_path),
        "features": features,
        "test_start_year": test_start_year,
        "neural_test_start_year": neural_test_start_year,
        "models": [block.model for block in blocks],
        "n_panel_rows": int(len(panel)),
        "comparison_rows": int(len(comparison)),
        "public_outputs_aggregate_only": True,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "model_ladder.json", manifest)
    return manifest
