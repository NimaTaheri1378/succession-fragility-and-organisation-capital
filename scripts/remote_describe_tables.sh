#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: remote_describe_tables.sh LIBRARY OUTPUT TABLE [TABLE ...]" >&2
  exit 2
fi

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"
LIBRARY="$1"
OUTPUT="$2"
shift 2

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export WRDS_USERNAME="${WRDS_USERNAME:?Set WRDS_USERNAME in the job environment}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON_BIN" -m succession_fragility.cli describe-tables \
  --library "$LIBRARY" \
  --output "$OUTPUT" \
  --tables "$@"
