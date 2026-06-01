#!/usr/bin/env bash
set -euo pipefail

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "PATH=$PATH"
command -v nvidia-smi || true
for candidate in /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi /usr/local/cuda/bin/nvidia-smi; do
  if [[ -x "$candidate" ]]; then
    "$candidate" -L || true
  fi
done

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"
"$PYTHON_BIN" - <<'PY'
import torch
print("torch_version", torch.__version__)
print("torch_cuda_version", torch.version.cuda)
print("torch_cuda_available", torch.cuda.is_available())
print("torch_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch_device_0", torch.cuda.get_device_name(0))
PY
