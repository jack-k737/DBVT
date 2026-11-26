#!/bin/bash
# Generic multi-GPU / multi-fold experiment runner.
#
# Environment variables (all optional):
#   PYTHON_BIN   Python interpreter (default: python3)
#   DATASETS     Comma-separated datasets (default: all three)
#   MODELS       Comma-separated model types (default: dbvt)
#   FOLDS        Comma-separated folds (default: 1,2,3,4,5)
#   GPUS         Comma-separated GPU ids (default: 0)
#   DEVICE       If set (e.g. cpu), run sequentially on this device and ignore GPUS.
#   MAX_EPOCHS   Default: 20
#   PATIENCE     Default: 5
#   SEED         Default: 2026
#   EXTRA_ARGS   Extra CLI args, e.g. "--ablation_mode no_gru"
#   AGGREGATE    Set to 0 to skip result aggregation (default: 1)
#
# Example:
#   DATASETS=physionet_2012 MODELS=dbvt,tcn GPUS=0,1 bash scripts/run_experiments.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RAW_DATASETS="${DATASETS:-physionet_2012,physionet_2019,mimic_iii}"
RAW_MODELS="${MODELS:-dbvt}"
RAW_FOLDS="${FOLDS:-1,2,3,4,5}"
RAW_GPUS="${GPUS:-0,1}"
DEVICE="${DEVICE:-}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-2026}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
AGGREGATE="${AGGREGATE:-1}"

parse_list() {
    local raw="$1"
    local -n out="$2"
    out=()
    local token
    IFS=',' read -r -a tokens <<< "$raw"
    for token in "${tokens[@]}"; do
        token="${token// /}"
        [ -n "$token" ] && out+=("$token")
    done
}

declare -a DATASETS=() MODELS=() FOLDS=() GPUS=()
parse_list "$RAW_DATASETS" DATASETS
parse_list "$RAW_MODELS" MODELS
parse_list "$RAW_FOLDS" FOLDS
parse_list "$RAW_GPUS" GPUS

[ ${#DATASETS[@]} -eq 0 ] && { echo "ERROR: empty DATASETS" >&2; exit 1; }
[ ${#MODELS[@]} -eq 0 ] && { echo "ERROR: empty MODELS" >&2; exit 1; }
[ ${#FOLDS[@]} -eq 0 ] && { echo "ERROR: empty FOLDS" >&2; exit 1; }
[ ${#GPUS[@]} -eq 0 ] && GPUS=(0)

if [ -n "$DEVICE" ]; then
    GPUS=(cpu)
fi
for gpu in "${GPUS[@]}"; do
    [ "$gpu" = "cpu" ] || [[ "$gpu" =~ ^[0-9]+$ ]] || {
        echo "ERROR: GPUS must be comma-separated integer ids (got: $gpu)" >&2
        exit 1
    }
done

LOG_DIR="$PROJECT_ROOT/logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
TASK_FILE="$LOG_DIR/tasks.txt"
: > "$TASK_FILE"
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for fold in "${FOLDS[@]}"; do
            printf '%s|%s|%s\n' "$dataset" "$model" "$fold" >> "$TASK_FILE"
        done
    done
done

TOTAL=$(wc -l < "$TASK_FILE")
echo "============================================"
echo "DBVT experiments"
echo "Python:      $PYTHON_BIN"
echo "Datasets:    ${DATASETS[*]}"
echo "Models:      ${MODELS[*]}"
echo "Folds:       ${FOLDS[*]}"
echo "Devices:     ${GPUS[*]}"
echo "Extra args:  ${EXTRA_ARGS:-<none>}"
echo "Total tasks: $TOTAL"
echo "Logs:        $LOG_DIR"
echo "============================================"

run_task() {
    local task="$1" slot="$2" gpu="$3"
    local dataset model fold log_file
    IFS='|' read -r dataset model fold <<< "$task"
    log_file="$LOG_DIR/${dataset}_${model}_fold${fold}.log"

    if [ "$gpu" = "cpu" ]; then
        device="cpu"
    else
        device="cuda:$gpu"
    fi

    echo "[slot $slot | $device] start  $dataset / $model / fold $fold"
    cd "$PROJECT_ROOT/src" || exit 1
    # shellcheck disable=SC2086
    if "$PYTHON_BIN" main.py \
        --dataset "$dataset" \
        --model_type "$model" \
        --fold "$fold" \
        --max_epochs "$MAX_EPOCHS" \
        --patience "$PATIENCE" \
        --seed "$SEED" \
        --device "$device" \
        $EXTRA_ARGS > "$log_file" 2>&1; then
        echo "[slot $slot | $device] done   $dataset / $model / fold $fold"
    else
        echo "[slot $slot | $device] FAILED $dataset / $model / fold $fold (see $log_file)" >&2
    fi
}

declare -a PIDS=()
slot=0
num_slots=${#GPUS[@]}

while IFS= read -r task; do
    while :; do
        slot=$((slot % num_slots))
        pid="${PIDS[$slot]:-}"
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        slot=$(((slot + 1) % num_slots))
        sleep 1
    done
    run_task "$task" "$slot" "${GPUS[$slot]}" &
    PIDS[$slot]=$!
done < "$TASK_FILE"

if [ ${#PIDS[@]} -gt 0 ]; then
    for pid in "${PIDS[@]}"; do
        [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
    done
fi

echo "============================================"
echo "All tasks finished. Logs: $LOG_DIR"
echo "============================================"

if [ "$AGGREGATE" = "1" ]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/aggregate_test_metrics.py" \
        --outputs_dir "$PROJECT_ROOT/outputs" | tee "$LOG_DIR/summary.txt"
fi
