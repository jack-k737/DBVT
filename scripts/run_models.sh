#!/bin/bash
# 8-slot runner: all target models except hipatch/strats and excluding dbvt/gru/grud.

set -u

PYTHON_BIN="/home/xiejun/miniconda3/envs/ehrs/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: python not found at $PYTHON_BIN"
    exit 1
fi

NUM_FOLDS="${NUM_FOLDS:-5}"
RAW_DATASETS="${DATASETS:-physionet_2012}"
RAW_MODELS="${MODELS:-tcn,sand}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-2026}"

unset DATASETS MODELS
declare -a DATASETS=()
declare -a MODELS=()

GPUS=(0 1 2 3 0 1 2 3 0 1)
CPU_SETS=("0-15" "32-47" "64-79" "96-111" "16-31" "48-63" "80-95" "112-127" "0-31" "32-63")
NUM_SLOTS=${#GPUS[@]}

if [ ${#CPU_SETS[@]} -ne $NUM_SLOTS ]; then
    echo "Error: CPU_SETS and GPUS must have the same length."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

parse_list() {
    local raw_value="$1"
    local -n output_ref="$2"
    local token

    output_ref=()
    IFS=, read -r -a raw_tokens <<< "$raw_value"
    for token in "${raw_tokens[@]}"; do
        token="${token// /}"
        [ -z "$token" ] && continue
        output_ref+=("$token")
    done
}

parse_list "$RAW_DATASETS" DATASETS
parse_list "$RAW_MODELS" MODELS

if [ ${#DATASETS[@]} -eq 0 ] || [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: DATASETS and MODELS must each contain at least one value."
    exit 1
fi

TASK_FILE="${LOG_DIR}/tasks.txt"
> "$TASK_FILE"
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for ((fold=1; fold<=NUM_FOLDS; fold++)); do
            echo "${dataset}|${model}|${fold}" >> "$TASK_FILE"
        done
    done
done

TOTAL_TASKS=$(wc -l < "$TASK_FILE")
echo "============================================"
echo "Training Model Group A"
echo "Python: $PYTHON_BIN"
echo "GPUs: ${GPUS[*]}"
echo "CPU sets: ${CPU_SETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Folds: $NUM_FOLDS"
echo "Total tasks: $TOTAL_TASKS"
echo "Log dir: $LOG_DIR"
echo "============================================"

run_task() {
    local task="$1"
    local gpu_id="$2"
    local cpu_set="$3"
    IFS='|' read -r dataset model fold <<< "$task"
    local log_file="${LOG_DIR}/${dataset}_${model}_fold${fold}.log"

    echo "[GPU:$gpu_id|CPU:$cpu_set] Starting: ${dataset}/${model} (fold ${fold})"
    cd "${PROJECT_ROOT}/src" || exit 1

    if OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8         taskset -c "$cpu_set"         "$PYTHON_BIN" main.py             --dataset "$dataset"             --model_type "$model"             --fold "$fold"             --max_epochs "$MAX_EPOCHS"             --patience "$PATIENCE"             --seed "$SEED"             --device "cuda:$gpu_id" > "$log_file" 2>&1; then
        echo "[GPU:$gpu_id|CPU:$cpu_set] Completed: ${dataset}/${model} (fold ${fold})"
    else
        echo "[GPU:$gpu_id|CPU:$cpu_set] Failed: ${dataset}/${model} (fold ${fold})"
    fi
}

declare -a SLOT_PIDS=()

while IFS= read -r task; do
    slot_idx=-1
    while [ "$slot_idx" -lt 0 ]; do
        for idx in "${!GPUS[@]}"; do
            pid="${SLOT_PIDS[$idx]:-}"
            if [ -z "$pid" ]; then
                slot_idx=$idx
                break
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || true
                SLOT_PIDS[$idx]=""
                slot_idx=$idx
                break
            fi
        done
        if [ "$slot_idx" -lt 0 ]; then
            sleep 1
        fi
    done

    gpu_id=${GPUS[$slot_idx]}
    cpu_set=${CPU_SETS[$slot_idx]}
    run_task "$task" "$gpu_id" "$cpu_set" &
    SLOT_PIDS[$slot_idx]=$!
done < "$TASK_FILE"

for pid in "${SLOT_PIDS[@]}"; do
    if [ -n "$pid" ]; then
        wait "$pid" || true
    fi
done

echo ""
echo "============================================"
echo "Training completed! Running aggregate..."
echo "============================================"

cd "${PROJECT_ROOT}/scripts" || exit 1
"$PYTHON_BIN" aggregate_test_metrics.py --outputs_dir "${PROJECT_ROOT}/outputs"

echo ""
echo "All done! Logs: $LOG_DIR"
