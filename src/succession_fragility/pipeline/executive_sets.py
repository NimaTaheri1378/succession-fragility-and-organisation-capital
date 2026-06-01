from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from succession_fragility.backtest.portfolio import long_short_returns, performance_summary
from succession_fragility.features.ocf import role_weight
from succession_fragility.models.deep_sets import DeepSetConfig, build_deep_set_model
from succession_fragility.pipeline.event_study import _read_year_shards
from succession_fragility.plots.figures import plot_model_comparison
from succession_fragility.plots.visual_qa import inspect_figure_dir
from succession_fragility.utils.manifest import write_manifest

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None


EXEC_FEATURES = [
    "role_weight",
    "log_tenure",
    "internal_candidate",
    "outside_roles",
    "prior_employers",
]


def _rank_ic(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 5:
        return np.nan
    if spearmanr is not None:
        return float(spearmanr(y[mask], yhat[mask]).statistic)
    return float(pd.Series(y[mask]).corr(pd.Series(yhat[mask]), method="spearman"))


def _load_panel(panel_path: Path) -> pd.DataFrame:
    cols = ["boardex_companyid", "date", "permno", "ret_excess_fwd1m", "mktcap"]
    panel = pd.read_parquet(panel_path, columns=cols)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["month"] = panel["date"].dt.to_period("M")
    panel["boardex_companyid"] = panel["boardex_companyid"].astype(str)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_excess_fwd1m"])
    return panel


def _prepare_roles(raw_root: Path, years: list[int]) -> pd.DataFrame:
    roles = _read_year_shards(
        raw_root,
        "boardex_roles",
        years,
        columns=[
            "directorid",
            "companyid",
            "rolename",
            "brdposition",
            "datestartrole",
            "dateendrole",
            "leadershipteam",
        ],
    )
    roles = roles.drop_duplicates(
        ["directorid", "companyid", "rolename", "brdposition", "datestartrole", "dateendrole"]
    ).copy()
    roles["person_id"] = roles["directorid"].astype(str)
    roles["boardex_companyid"] = roles["companyid"].astype(str)
    roles["title"] = (roles["rolename"].fillna("").astype(str) + " " + roles["brdposition"].fillna("").astype(str)).str.strip()
    roles["role_weight"] = roles["title"].map(role_weight)
    roles = roles[roles["role_weight"] >= 0.40].copy()
    roles["role_start_date"] = pd.to_datetime(roles["datestartrole"], errors="coerce", format="mixed")
    roles["role_end_date"] = pd.to_datetime(roles["dateendrole"], errors="coerce", format="mixed")
    roles["leadership_flag"] = roles["leadershipteam"].astype(str).str.lower().isin({"1", "y", "yes", "true", "t"})
    first_starts = (
        roles.dropna(subset=["role_start_date"])
        .groupby(["person_id", "boardex_companyid"], observed=True)["role_start_date"]
        .min()
        .reset_index()
    )
    history: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first_start_lookup: dict[tuple[str, str], np.datetime64] = {}
    for person_id, group in first_starts.groupby("person_id", observed=True):
        order = np.argsort(group["role_start_date"].to_numpy(dtype="datetime64[ns]"))
        history[str(person_id)] = (
            group["role_start_date"].to_numpy(dtype="datetime64[ns]")[order],
            group["boardex_companyid"].astype(str).to_numpy()[order],
        )
    for row in first_starts.itertuples(index=False):
        first_start_lookup[(str(row.person_id), str(row.boardex_companyid))] = np.datetime64(row.role_start_date)
    roles.attrs["person_company_first_starts"] = history
    roles.attrs["person_company_first_start_lookup"] = first_start_lookup
    return roles


def _expand_roles_for_year(roles: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    active = roles[
        roles["role_start_date"].fillna(start).le(end)
        & roles["role_end_date"].fillna(pd.Timestamp("2099-12-31")).ge(start)
    ].copy()
    history = roles.attrs.get("person_company_first_starts", {})
    first_start_lookup = roles.attrs.get("person_company_first_start_lookup", {})
    rows: list[tuple[object, ...]] = []
    for row in active.itertuples(index=False):
        role_start = row.role_start_date if pd.notna(row.role_start_date) else start
        role_end = row.role_end_date if pd.notna(row.role_end_date) else end
        active_start = max(start, role_start)
        active_end = min(end, role_end)
        if active_start > active_end:
            continue
        for date in pd.date_range(active_start, active_end, freq="ME"):
            tenure = max((date - role_start).days / 365.25, 0.0)
            person_history = history.get(str(row.person_id))
            if person_history is None:
                prior_employers = 0.0
            else:
                first_dates, companies = person_history
                current_company = str(row.boardex_companyid)
                cutoff = np.datetime64(date)
                prior_count = int(np.searchsorted(first_dates, cutoff, side="left"))
                current_first = first_start_lookup.get((str(row.person_id), current_company))
                if current_first is not None and current_first < cutoff:
                    prior_count -= 1
                elif current_first is None:
                    current_dates = first_dates[companies == current_company]
                    if len(current_dates) and current_dates[0] < cutoff:
                        prior_count -= 1
                prior_employers = float(max(prior_count, 0))
            rows.append(
                (
                    row.boardex_companyid,
                    row.person_id,
                    date.to_period("M"),
                    float(row.role_weight),
                    float(np.log1p(tenure)),
                    float(row.leadership_flag and "ceo" not in str(row.title).lower()),
                    prior_employers,
                )
            )
    cols = [
        "boardex_companyid",
        "person_id",
        "month",
        "role_weight",
        "log_tenure",
        "internal_candidate",
        "prior_employers",
    ]
    expanded = pd.DataFrame(rows, columns=cols)
    if expanded.empty:
        return expanded
    company_counts = expanded.groupby(["person_id", "month"], observed=True)["boardex_companyid"].transform("nunique")
    expanded["outside_roles"] = (company_counts - 1).clip(lower=0)
    return expanded


def _arrays_for_year(
    panel: pd.DataFrame,
    roles: pd.DataFrame,
    year: int,
    max_exec: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    p = panel[panel["date"].dt.year.eq(year)].copy().reset_index(drop=True)
    if p.empty:
        return (
            np.empty((0, max_exec, len(EXEC_FEATURES)), dtype=np.float32),
            np.empty((0, max_exec), dtype=np.float32),
            p,
            0,
        )
    p["obs_idx"] = np.arange(len(p))
    expanded = _expand_roles_for_year(roles, year)
    if expanded.empty:
        return (
            np.zeros((len(p), max_exec, len(EXEC_FEATURES)), dtype=np.float32),
            np.zeros((len(p), max_exec), dtype=np.float32),
            p,
            0,
        )
    merged = expanded.merge(p[["boardex_companyid", "month", "obs_idx"]], on=["boardex_companyid", "month"], how="inner")
    if merged.empty:
        return (
            np.zeros((len(p), max_exec, len(EXEC_FEATURES)), dtype=np.float32),
            np.zeros((len(p), max_exec), dtype=np.float32),
            p,
            0,
        )
    merged["_priority"] = merged["role_weight"] * (1.0 + merged["log_tenure"])
    merged = merged.sort_values(["obs_idx", "_priority"], ascending=[True, False])
    merged["slot"] = merged.groupby("obs_idx", observed=True).cumcount()
    merged = merged[merged["slot"] < max_exec].copy()
    x = np.zeros((len(p), max_exec, len(EXEC_FEATURES)), dtype=np.float32)
    mask = np.zeros((len(p), max_exec), dtype=np.float32)
    obs = merged["obs_idx"].to_numpy(dtype=int)
    slot = merged["slot"].to_numpy(dtype=int)
    x[obs, slot, :] = merged[EXEC_FEATURES].to_numpy(dtype=np.float32)
    mask[obs, slot] = 1.0
    return x, mask, p, int(len(merged))


def _standardize_sets(x_train: np.ndarray, x_test: np.ndarray, mask_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_train[mask_train.astype(bool)]
    med = np.nanmedian(flat, axis=0)
    scale = np.nanstd(flat, axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale == 0), 1.0, scale)
    return ((x_train - med) / scale).astype(np.float32), ((x_test - med) / scale).astype(np.float32)


def _fit_deep_sets_score(
    x_train_raw: np.ndarray,
    mask_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    mask_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    use_cuda: bool,
    seed: int,
) -> tuple[np.ndarray, str]:
    x_train, x_test = _standardize_sets(x_train_raw, x_test_raw, mask_train)
    y_mean = float(np.nanmean(y_train_raw))
    y_std = float(np.nanstd(y_train_raw))
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    y_train = np.clip((y_train_raw - y_mean) / y_std, -10.0, 10.0).astype(np.float32)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = build_deep_set_model(DeepSetConfig(n_features=len(EXEC_FEATURES), hidden_dim=128, dropout=0.10)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(mask_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb, mb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb.to(device), mb.to(device))[:, 0]
            loss = loss_fn(pred, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            xb = torch.from_numpy(x_test[start : start + batch_size]).to(device)
            mb = torch.from_numpy(mask_test[start : start + batch_size]).to(device)
            preds.append(model(xb, mb)[:, 0].cpu().numpy())
    scores = np.concatenate(preds) * y_std + y_mean
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, device.type


def run_executive_deep_sets(
    panel_path: Path,
    raw_root: Path,
    output_dir: Path,
    years: list[int],
    test_start_year: int = 2018,
    max_exec: int = 32,
    epochs: int = 8,
    batch_size: int = 8192,
    use_cuda: bool = True,
    seed: int = 20260601,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    manifest_dir = output_dir / "manifests"
    for path in [table_dir, figure_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(panel_path)
    roles = _prepare_roles(raw_root, years)
    print(
        f"[executive_sets] loaded panel_rows={len(panel)} role_rows={len(roles)} "
        f"years={min(years)}-{max(years)}",
        flush=True,
    )
    year_data: dict[int, dict[str, object]] = {}
    retained_exec_rows = 0
    for year in years:
        x, mask, meta, retained = _arrays_for_year(panel, roles, year, max_exec)
        if len(meta) == 0:
            print(f"[executive_sets] expanded year={year} obs=0 valid=0 retained_slots=0", flush=True)
            continue
        retained_exec_rows += retained
        y = meta["ret_excess_fwd1m"].to_numpy(dtype=np.float32)
        valid = mask.sum(axis=1) > 0
        print(
            f"[executive_sets] expanded year={year} obs={len(meta)} "
            f"valid={int(valid.sum())} retained_slots={retained}",
            flush=True,
        )
        if not valid.any():
            continue
        year_data[year] = {
            "x": x[valid],
            "mask": mask[valid],
            "y": y[valid],
            "meta": meta.loc[valid, ["date", "permno", "ret_excess_fwd1m", "mktcap"]].copy(),
        }

    test_years = [year for year in years if year >= test_start_year and year in year_data]
    if not test_years or not any(year < test_start_year for year in year_data):
        raise RuntimeError("Executive set tensors are empty after linking roles to the panel.")

    prediction_frames: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    device_type = "cpu"
    for target_year in test_years:
        train_years = [year for year in sorted(year_data) if year < target_year]
        if not train_years:
            continue
        x_train = np.concatenate([year_data[year]["x"] for year in train_years])
        mask_train = np.concatenate([year_data[year]["mask"] for year in train_years])
        y_train_raw = np.concatenate([year_data[year]["y"] for year in train_years])
        x_test = year_data[target_year]["x"]
        mask_test = year_data[target_year]["mask"]
        meta_test = year_data[target_year]["meta"].copy()
        print(
            f"[executive_sets] fitting target_year={target_year} "
            f"train={min(train_years)}-{max(train_years)} n_train={len(x_train)} n_test={len(x_test)}",
            flush=True,
        )
        scores, device_type = _fit_deep_sets_score(
            x_train,
            mask_train,
            y_train_raw,
            x_test,
            mask_test,
            epochs=epochs,
            batch_size=batch_size,
            use_cuda=use_cuda,
            seed=seed + target_year,
        )
        year_ic = _rank_ic(meta_test["ret_excess_fwd1m"].to_numpy(dtype=float), scores.astype(float))
        print(
            f"[executive_sets] predicted target_year={target_year} device={device_type} rank_ic={year_ic:.6f}",
            flush=True,
        )
        prediction_frames.append(meta_test.assign(score=scores, test_year=target_year))
        training_rows.append(
            {
                "target_year": int(target_year),
                "train_start_year": int(min(train_years)),
                "train_end_year": int(max(train_years)),
                "n_train_obs": int(len(x_train)),
                "n_test_obs": int(len(x_test)),
            }
        )

    if not prediction_frames:
        raise RuntimeError("No executive-set target-year predictions were produced.")

    pred_frame = pd.concat(prediction_frames, ignore_index=True)
    rows = []
    for year, group in pred_frame.groupby("test_year", observed=True):
        y = group["ret_excess_fwd1m"].to_numpy(dtype=float)
        score = group["score"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(score)
        if mask.sum() < 5:
            continue
        rows.append({"model": "executive_deep_sets", "test_year": int(year), "n_obs": int(mask.sum()), "rank_ic": _rank_ic(y[mask], score[mask])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(table_dir / "executive_deep_sets_metrics_by_year.csv", index=False)
    comparison = (
        metrics.groupby("model", observed=True)
        .agg(
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_t=("rank_ic", lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std(ddof=1) else np.nan),
            n_test_years=("test_year", "nunique"),
            n_obs=("n_obs", "sum"),
        )
        .reset_index()
    )
    comparison.to_csv(table_dir / "executive_deep_sets_comparison.csv", index=False)
    ls = long_short_returns(pred_frame, signal="score", ret_col="ret_excess_fwd1m", q=5)
    ls.to_csv(table_dir / "executive_deep_sets_long_short.csv", index=False)
    perf = performance_summary(ls["long_short"]) if not ls.empty else {}
    pd.DataFrame([{"model": "executive_deep_sets", **perf}]).to_csv(table_dir / "executive_deep_sets_performance.csv", index=False)
    if not comparison.empty:
        plot_model_comparison(comparison, figure_dir / "executive_deep_sets_rank_ic.png")
    qa = inspect_figure_dir(figure_dir)
    manifest = {
        "kind": "executive_deep_sets",
        "status": "ok",
        "panel_path": str(panel_path),
        "raw_root": str(raw_root),
        "years": years,
        "test_start_year": test_start_year,
        "max_exec": max_exec,
        "features": EXEC_FEATURES,
        "device": device_type,
        "n_model_fits": int(len(training_rows)),
        "n_train_obs_total_across_fits": int(sum(row["n_train_obs"] for row in training_rows)),
        "n_test_obs": int(len(pred_frame)),
        "training_windows": training_rows,
        "retained_executive_slots": retained_exec_rows,
        "comparison": comparison.to_dict(orient="records"),
        "performance": perf,
        "timing_audit": {
            "removed_forward_exit_feature": True,
            "prior_employers_uses_only_roles_started_before_formation_month": True,
            "expanding_walk_forward_by_target_year": True,
            "target_year_model_uses_only_prior_years": True,
            "row_level_predictions_public": False,
        },
        "public_outputs_aggregate_only": True,
        "visual_qa": qa,
    }
    write_manifest(manifest_dir / "executive_deep_sets.json", manifest)
    print(f"[executive_sets] wrote manifest={manifest_dir / 'executive_deep_sets.json'}", flush=True)
    return manifest
