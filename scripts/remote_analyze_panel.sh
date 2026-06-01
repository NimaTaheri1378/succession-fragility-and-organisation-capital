#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON_BIN" -m succession_fragility.cli analyze-panel "$@"
