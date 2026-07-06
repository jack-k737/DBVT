#!/bin/bash
# Sequential master entrypoint with optional stages.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

run_stage() {
    local label="$1"
    local target="$2"

    echo "============================================"
    echo "$label"
    echo "============================================"

    if [ -x "$target" ]; then
        bash "$target"
        return
    fi

    if [ -f "$target" ]; then
        bash "$target"
        return
    fi

    echo "Skipping missing script: $target"
}

run_stage "Stage 1: main model group A" "$SCRIPT_DIR/run_models.sh"
run_stage "Stage 2: main model group B" "$SCRIPT_DIR/run_models1.sh"
run_stage "Stage 3: ablation study" "$PROJECT_ROOT/lab-ablation/run_models.sh"
run_stage "Stage 4: variable-dropout study" "$PROJECT_ROOT/lab-dropout/run_models.sh"
