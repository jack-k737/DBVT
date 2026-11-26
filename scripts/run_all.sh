#!/bin/bash
# Reproduce all paper experiments sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "### 1/4 Main comparison + plug-in (Table II)"
bash "$SCRIPT_DIR/run_main_comparison.sh"

echo "### 2/4 Component ablation (Table IV)"
bash "$SCRIPT_DIR/run_ablation.sh"

echo "### 3/4 Leave-variables-out robustness (Table III)"
bash "$SCRIPT_DIR/run_lvo.sh"

echo "### 4/4 Optuna hyperparameter tuning"
bash "$SCRIPT_DIR/run_optuna.sh"
