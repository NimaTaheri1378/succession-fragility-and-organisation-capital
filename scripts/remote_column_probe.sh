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

if [[ "$#" -gt 0 ]]; then
  "$PYTHON_BIN" scripts/probe_wrds_columns.py "$@"
else
  "$PYTHON_BIN" scripts/probe_wrds_columns.py \
    --output reports/manifests/wrds_core_columns.json \
    boardex_na.na_wrds_company_profile \
    boardex_na.na_wrds_dir_profile_emp \
    boardex_na.na_wrds_dir_profile_all \
    boardex_na.na_wrds_org_summary \
    boardex_na.na_wrds_org_composition \
    boardex_na.na_company_profile_stocks \
    boardex_na.na_lookuproles \
    crsp.crsp_monthly_data \
    crsp.crsp_daily_data \
    crsp.ccmxpf_lnkhist \
    comp.funda \
    ff.factors_monthly \
    ff.fivefactors_monthly \
    ibes.statsum_epsus \
    ibes.nstatsum_epsus
fi
