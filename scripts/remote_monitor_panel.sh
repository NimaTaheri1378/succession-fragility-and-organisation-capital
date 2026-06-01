#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCF_PROJECT_ROOT:-${SCRATCH:-/scratch/$USER}/Github/succession-fragility-and-organisation-capital}"
cd "$PROJECT_ROOT"

LOG_BASENAME="${LOG_BASENAME:-full_panel_1995_2025}"
echo "--- process"
ps -u "$USER" -o pid,ppid,pcpu,pmem,etime,rss,args | egrep 'remote_full_panel|run-full-panel|full_panel.py' | grep -v grep || true
echo "--- out tail"
tail -n "${TAIL_LINES:-40}" "reports/logs/${LOG_BASENAME}.out" 2>/dev/null || true
echo "--- err tail"
tail -n "${TAIL_LINES:-30}" "reports/logs/${LOG_BASENAME}.err" 2>/dev/null || true
echo "--- output files"
find "reports/full_1995_2025" -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -n 20
