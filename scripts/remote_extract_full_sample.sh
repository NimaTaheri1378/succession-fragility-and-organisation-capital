#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
cd "$PROJECT_ROOT"

YEARS=()
for year in $(seq "${OCF_SAMPLE_START_YEAR:-1995}" "${OCF_SAMPLE_END_YEAR:-2025}"); do
  YEARS+=("$year")
done

bash scripts/remote_extract_wrds.sh \
  --output-root data/raw/wrds \
  --datasets boardex_company ccm_links ff5 crsp_monthly crsp_daily crsp_delist comp_funda boardex_roles ibes_attention \
  --years "${YEARS[@]}"
