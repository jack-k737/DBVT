#!/bin/bash
# Table III: leave-variables-out robustness on MIMIC-III.
# Removes 10%, 30%, and 50% of variable types and evaluates all paper methods.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DROP_RATES="${DROP_RATES:-0.1,0.3,0.5}"
for rate in ${DROP_RATES//,/ }; do
    echo ">>> Variable drop rate: $rate"
    AGGREGATE=0 \
    DATASETS="${DATASETS:-mimic_iii}" \
    MODELS="${MODELS:-tcn,sand,grud,raindrop,hipatch,kedgn,strats,warpformer,mtm,dbvt}" \
    EXTRA_ARGS="--variable_drop_rate $rate" \
    bash "$SCRIPT_DIR/run_experiments.sh"
done

"${PYTHON_BIN:-python3}" "$PROJECT_ROOT/scripts/aggregate_test_metrics.py" \
    --outputs_dir "$PROJECT_ROOT/outputs"
