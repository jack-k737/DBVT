#!/bin/bash
# Table IV: component ablation on all three datasets.
# full / no_gru (freeze GRU branch) / no_transformer (freeze Transformer branch)
# / no_fusion (concatenation instead of gated sum).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODES="${ABLATION_MODES:-full,no_gru,no_transformer,no_fusion}"
for mode in ${MODES//,/ }; do
    echo ">>> Ablation mode: $mode"
    AGGREGATE=0 \
    MODELS=dbvt \
    EXTRA_ARGS="--ablation_mode $mode" \
    bash "$SCRIPT_DIR/run_experiments.sh"
done

"${PYTHON_BIN:-python3}" "$PROJECT_ROOT/scripts/aggregate_test_metrics.py" \
    --outputs_dir "$PROJECT_ROOT/outputs"
