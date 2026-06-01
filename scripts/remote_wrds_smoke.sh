#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export WRDS_USERNAME="${WRDS_USERNAME:?Set WRDS_USERNAME in the job environment}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON_BIN" -m succession_fragility.cli wrds-smoke \
  --output-dir data/smoke/wrds \
  --start-date "${OCF_SMOKE_START_DATE:-2020-01-01}" \
  --end-date "${OCF_SMOKE_END_DATE:-2020-12-31}"
