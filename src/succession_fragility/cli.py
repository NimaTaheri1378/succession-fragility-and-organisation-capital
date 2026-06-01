from __future__ import annotations

import argparse
import json
from pathlib import Path

from succession_fragility.extract.wrds_schema import describe_tables, run_schema_audit
from succession_fragility.extract.coverage import run_boardex_coverage
from succession_fragility.extract.wrds_full import extract_many
from succession_fragility.extract.wrds_smoke import run_wrds_smoke
from succession_fragility.pipeline.dashboard import build_public_dashboard
from succession_fragility.pipeline.event_study import run_event_study
from succession_fragility.pipeline.executive_sets import run_executive_deep_sets
from succession_fragility.pipeline.synthetic import run_synthetic
from succession_fragility.pipeline.full_panel import analyze_panel_file, run_full_panel
from succession_fragility.pipeline.model_ladder import run_model_ladder
from succession_fragility.pipeline.regimes import build_regime_controls
from succession_fragility.pipeline.robustness import run_robustness
from succession_fragility.pipeline.sharp_strategy import run_sharp_strategy
from succession_fragility.pipeline.wrds_smoke_panel import run_wrds_smoke_panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ocf")
    sub = parser.add_subparsers(dest="command", required=True)

    p_syn = sub.add_parser("run-synthetic", help="Run synthetic end-to-end smoke pipeline.")
    p_syn.add_argument("--output-dir", type=Path, default=Path("reports"))

    p_schema = sub.add_parser("audit-schema", help="Run WRDS metadata-only schema audit.")
    p_schema.add_argument("--output", type=Path, default=Path("configs/schema_map.json"))

    p_desc = sub.add_parser("describe-tables", help="Describe selected WRDS tables.")
    p_desc.add_argument("--library", required=True)
    p_desc.add_argument("--tables", nargs="+", required=True)
    p_desc.add_argument("--output", type=Path, required=True)

    p_wrds_smoke = sub.add_parser("wrds-smoke", help="Run small WRDS row-count smoke pulls.")
    p_wrds_smoke.add_argument("--output-dir", type=Path, default=Path("data/smoke/wrds"))
    p_wrds_smoke.add_argument("--start-date", default="2020-01-01")
    p_wrds_smoke.add_argument("--end-date", default="2020-12-31")

    p_wrds_panel = sub.add_parser("run-wrds-smoke-panel", help="Build linked panel from cached WRDS smoke shards.")
    p_wrds_panel.add_argument("--smoke-dir", type=Path, default=Path("data/smoke/wrds"))
    p_wrds_panel.add_argument("--output-dir", type=Path, default=Path("reports/wrds_smoke"))
    p_wrds_panel.add_argument("--start-date", default="2020-01-31")
    p_wrds_panel.add_argument("--end-date", default="2020-12-31")

    p_extract = sub.add_parser("extract-wrds", help="Run resumable WRDS full/sharded extracts.")
    p_extract.add_argument("--output-root", type=Path, default=Path("data/raw/wrds"))
    p_extract.add_argument("--datasets", nargs="+", required=True)
    p_extract.add_argument("--years", nargs="*", type=int, default=[])
    p_extract.add_argument("--force", action="store_true")

    p_cov = sub.add_parser("boardex-coverage", help="Run metadata-style BoardEx active coverage diagnostics.")
    p_cov.add_argument("--output-dir", type=Path, default=Path("reports/coverage"))
    p_cov.add_argument("--start-year", type=int, default=1995)
    p_cov.add_argument("--end-year", type=int, default=2025)

    p_full_panel = sub.add_parser("run-full-panel", help="Build panel and aggregate outputs from raw WRDS shards.")
    p_full_panel.add_argument("--raw-root", type=Path, default=Path("data/raw/wrds"))
    p_full_panel.add_argument("--output-dir", type=Path, default=Path("reports/full_panel"))
    p_full_panel.add_argument("--years", nargs="+", type=int, required=True)

    p_analyze_panel = sub.add_parser("analyze-panel", help="Generate aggregate outputs from a cached private panel.")
    p_analyze_panel.add_argument("--panel", type=Path, required=True)
    p_analyze_panel.add_argument("--output-dir", type=Path, required=True)
    p_analyze_panel.add_argument("--years", nargs="+", type=int, required=True)

    p_regimes = sub.add_parser("build-regimes", help="Pull public FRED macro-regime controls.")
    p_regimes.add_argument("--output-dir", type=Path, default=Path("reports/regimes"))
    p_regimes.add_argument("--start-date", default="1995-01-01")
    p_regimes.add_argument("--end-date", default=None)

    p_robust = sub.add_parser("run-robustness", help="Run robustness, factor alpha, turnover, and regime tests.")
    p_robust.add_argument("--panel", type=Path, required=True)
    p_robust.add_argument("--output-dir", type=Path, default=Path("reports/robustness"))
    p_robust.add_argument("--regimes", type=Path, default=None)
    p_robust.add_argument("--ff5", type=Path, default=Path("data/raw/wrds/ff5/all.parquet"))
    p_robust.add_argument("--q", type=int, default=5)

    p_events = sub.add_parser("run-event-study", help="Run leadership-change event study from BoardEx roles and CRSP daily returns.")
    p_events.add_argument("--panel", type=Path, required=True)
    p_events.add_argument("--raw-root", type=Path, default=Path("data/raw/wrds"))
    p_events.add_argument("--output-dir", type=Path, default=Path("reports/event_study"))
    p_events.add_argument("--years", nargs="+", type=int, required=True)
    p_events.add_argument("--sec-overlay", action="store_true")
    p_events.add_argument("--sec-max-ciks", type=int, default=0, help="Cap SEC CIK fetches; 0 means no cap.")

    p_models = sub.add_parser("run-model-ladder", help="Run strict walk-forward Elastic Net, LightGBM, Deep Sets, and SDF models.")
    p_models.add_argument("--panel", type=Path, required=True)
    p_models.add_argument("--output-dir", type=Path, default=Path("reports/model_ladder"))
    p_models.add_argument("--test-start-year", type=int, default=2015)
    p_models.add_argument("--neural-test-start-year", type=int, default=2018)
    p_models.add_argument("--epochs-deep", type=int, default=8)
    p_models.add_argument("--epochs-sdf", type=int, default=8)
    p_models.add_argument("--batch-size", type=int, default=4096)
    p_models.add_argument("--no-cuda", action="store_true")
    p_models.add_argument("--n-jobs", type=int, default=8)

    p_dashboard = sub.add_parser("build-dashboard", help="Build public aggregate Plotly dashboard.")
    p_dashboard.add_argument("--panel", type=Path, required=True)
    p_dashboard.add_argument("--source-root", type=Path, default=Path("reports/full_1995_2025"))
    p_dashboard.add_argument("--output-dir", type=Path, default=Path("reports/dashboard"))
    p_dashboard.add_argument("--event-root", type=Path, default=None)
    p_dashboard.add_argument("--model-root", type=Path, default=None)
    p_dashboard.add_argument("--executive-root", type=Path, default=None)

    p_exec_sets = sub.add_parser("run-executive-deep-sets", help="Train a GPU Deep Sets model on real BoardEx executive sets.")
    p_exec_sets.add_argument("--panel", type=Path, required=True)
    p_exec_sets.add_argument("--raw-root", type=Path, default=Path("data/raw/wrds"))
    p_exec_sets.add_argument("--output-dir", type=Path, default=Path("reports/executive_deep_sets"))
    p_exec_sets.add_argument("--years", nargs="+", type=int, required=True)
    p_exec_sets.add_argument("--test-start-year", type=int, default=2018)
    p_exec_sets.add_argument("--max-exec", type=int, default=32)
    p_exec_sets.add_argument("--epochs", type=int, default=8)
    p_exec_sets.add_argument("--batch-size", type=int, default=8192)
    p_exec_sets.add_argument("--no-cuda", action="store_true")

    p_sharp = sub.add_parser("run-sharp-strategy", help="Run leakage-safe strategy sharpening around the walk-forward LightGBM score.")
    p_sharp.add_argument("--panel", type=Path, required=True)
    p_sharp.add_argument("--output-dir", type=Path, default=Path("reports/sharp_strategy"))
    p_sharp.add_argument("--ff5", type=Path, default=Path("data/raw/wrds/ff5/all.parquet"))
    p_sharp.add_argument("--test-start-year", type=int, default=2015)
    p_sharp.add_argument("--n-jobs", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "run-synthetic":
        result = run_synthetic(args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "audit-schema":
        result = run_schema_audit(args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "describe-tables":
        result = describe_tables(args.output, args.library, args.tables)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "wrds-smoke":
        result = run_wrds_smoke(args.output_dir, args.start_date, args.end_date)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-wrds-smoke-panel":
        result = run_wrds_smoke_panel(args.smoke_dir, args.output_dir, args.start_date, args.end_date)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "extract-wrds":
        result = [r.__dict__ for r in extract_many(args.datasets, args.output_root, args.years, force=args.force)]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "boardex-coverage":
        result = run_boardex_coverage(args.output_dir, args.start_year, args.end_year)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-full-panel":
        result = run_full_panel(args.raw_root, args.output_dir, args.years)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze-panel":
        result = analyze_panel_file(args.panel, args.output_dir, args.years)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-regimes":
        result = build_regime_controls(args.output_dir, args.start_date, args.end_date)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-robustness":
        result = run_robustness(args.panel, args.output_dir, args.regimes, ff5_path=args.ff5, q=args.q)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-event-study":
        result = run_event_study(
            args.panel,
            args.raw_root,
            args.output_dir,
            args.years,
            sec_overlay=args.sec_overlay,
            sec_max_ciks=args.sec_max_ciks or None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-model-ladder":
        result = run_model_ladder(
            args.panel,
            args.output_dir,
            test_start_year=args.test_start_year,
            neural_test_start_year=args.neural_test_start_year,
            epochs_deep=args.epochs_deep,
            epochs_sdf=args.epochs_sdf,
            batch_size=args.batch_size,
            use_cuda=not args.no_cuda,
            n_jobs=args.n_jobs,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-dashboard":
        result = build_public_dashboard(
            args.panel, args.source_root, args.output_dir, args.event_root, args.model_root, args.executive_root
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-executive-deep-sets":
        result = run_executive_deep_sets(
            args.panel,
            args.raw_root,
            args.output_dir,
            args.years,
            test_start_year=args.test_start_year,
            max_exec=args.max_exec,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_cuda=not args.no_cuda,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-sharp-strategy":
        result = run_sharp_strategy(
            args.panel,
            args.output_dir,
            ff5_path=args.ff5,
            test_start_year=args.test_start_year,
            n_jobs=args.n_jobs,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
