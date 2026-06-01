# Succession Fragility

This documentation describes the code and aggregate research artifacts for the succession-fragility asset-pricing project.

The headline result is a walk-forward LightGBM model-sort on executive-team succession and organisation-capital state variables: 13.35% annualized excess return, Sharpe 1.24, and FF5+UMD alpha t-stat 3.74 in tangible-heavy firms from 2015-2024.

## Public Artifacts

- `reports/full_1995_2025/`: baseline full-sample tables, manifests, and figures.
- `reports/robustness/`: robustness, regimes, turnover, costs, double sorts, and factor alpha.
- `reports/event_study/`: aggregate BoardEx leadership-change event paths.
- `reports/event_study_sec_audit/`: aggregate SEC timing-audit coverage.
- `reports/model_ladder/`: walk-forward model metrics and interpretation figures.
- `reports/sharp_strategy/`: sharpened walk-forward LightGBM strategy tables and figures.
- `reports/executive_deep_sets_timing_safe/`: leakage-safe BoardEx executive-set GPU Deep Sets metrics.
- `reports/dashboard/public_aggregate_dashboard.html`: public aggregate dashboard.
