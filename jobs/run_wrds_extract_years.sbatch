#!/usr/bin/env bash
#SBATCH --job-name=ocf_wrds_extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=08:00:00
#SBATCH --output=reports/logs/%x_%j.out
#SBATCH --error=reports/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
cd "$PROJECT_ROOT"

bash scripts/remote_extract_wrds.sh \
  --output-root data/raw/wrds \
  --datasets boardex_company ccm_links ff5 crsp_monthly crsp_daily crsp_delist comp_funda boardex_roles ibes_attention \
  --years "$@"
