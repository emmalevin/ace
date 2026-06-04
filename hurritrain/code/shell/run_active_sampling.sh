#!/bin/bash
# Allocate one node and run the active sampling driver with active_sampling.yaml.
# Usage:
#   ./run_active_sampling.sh
#   ./run_active_sampling.sh /path/to/active_sampling.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_DIR="${CODE_DIR}/python"
CONFIG="${1:-${CODE_DIR}/yaml/active_sampling.yaml}"

salloc \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=64G \
  --time=0-08:00:00 \
  --account=gvecchi \
  bash -lc "
    module purge
    module load anaconda3/2025.6
    conda activate geoclim
    cd '${PYTHON_DIR}'
    python loop_point_selection.py --config '${CONFIG}'
  "
