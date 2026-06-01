#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
cd "$PROJECT_ROOT"

echo "--- process"
ps -u "$USER" -f | egrep 'remote_extract_full_sample|remote_extract_wrds|extract-wrds' | grep -v grep || true
echo "--- out tail"
LOG_BASENAME="${LOG_BASENAME:-wrds_extract_1995_2025_resume}"
tail -n "${TAIL_LINES:-50}" "reports/logs/${LOG_BASENAME}.out" 2>/dev/null || true
echo "--- err tail"
tail -n "${TAIL_LINES:-30}" "reports/logs/${LOG_BASENAME}.err" 2>/dev/null || true
echo "--- manifest count"
find data/raw/wrds -name '*.manifest.json' | wc -l
echo "--- latest manifests"
find data/raw/wrds -name '*.manifest.json' -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -n 12
