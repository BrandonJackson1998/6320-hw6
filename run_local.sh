#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs
LOG="logs/run_local.log"

{
  echo "=== CS6320 Assignment 6 local run $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
  python3 scripts/prepare_bgg_data.py
  python3 scripts/run_split_audit.py
  python3 scripts/train_model_comparison.py
  python3 scripts/plot_mlp_training.py
  echo "Done."
} 2>&1 | tee "${LOG}"

echo "Local BGG model comparison complete. See prep/, outputs/, outputs/plots/"
