#!/bin/bash
# =============================================================================
# 所有模型训练脚本 - 并行运行 + 自动汇总
# =============================================================================

# 激活虚拟环境


# 配置
NUM_FOLDS=5  # 5-fold cross validation
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# 获取项目目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 日志目录
LOG_DIR="${PROJECT_ROOT}/logs/training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 模型和数据集
MODELS=("dbvt")
DATASETS=("physionet_2012" "physionet_2019" "mimic_iii")

# 生成任务列表（使用 fold 而不是 run）
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
echo "Training All Models"
echo "GPUs: ${GPUS[@]}"
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DATASETS[@]}"
echo "Folds: $NUM_FOLDS (K-fold cross validation)"
echo "Total tasks: $TOTAL_TASKS"
echo "Log dir: $LOG_DIR"
echo "============================================"

# 运行单个任务
run_task() {
    local task="$1"
    local gpu_id="$2"
    IFS='|' read -r dataset model fold <<< "$task"
    local log_file="${LOG_DIR}/${dataset}_${model}_fold${fold}.log"
    
    echo "[GPU:$gpu_id] Starting: ${dataset}/${model} (fold ${fold})"
    cd "${PROJECT_ROOT}/src"
    if python main.py --dataset "$dataset" --model_type "$model" --fold "$fold" --device "cuda:$gpu_id" > "$log_file" 2>&1; then
        echo "[GPU:$gpu_id] ✓ Completed: ${dataset}/${model} (fold ${fold})"
    else
        echo "[GPU:$gpu_id] ✗ Failed: ${dataset}/${model} (fold ${fold})"
    fi
}
export -f run_task
export LOG_DIR PROJECT_ROOT

# 并行执行
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel..."
    cat "$TASK_FILE" | parallel -j $NUM_GPUS --joblog "${LOG_DIR}/joblog.txt" \
        'gpus_arr=('"${GPUS[*]}"'); gpu_id=${gpus_arr[$(( ({%} - 1) % '"$NUM_GPUS"' ))]}; run_task {} $gpu_id'
else
    echo "GNU parallel not found, using simple parallel..."
    task_idx=0
    while IFS= read -r task; do
        while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do sleep 1; done
        gpu_id=${GPUS[$((task_idx % NUM_GPUS))]}
        run_task "$task" "$gpu_id" &
        task_idx=$((task_idx + 1))
    done < "$TASK_FILE"
    wait
fi

echo ""
echo "============================================"
echo "Training completed! Running aggregate..."
echo "============================================"

# 汇总结果
cd "${PROJECT_ROOT}/scripts"
python aggregate_test_metrics.py --outputs_dir "${PROJECT_ROOT}/outputs"

echo ""
echo "All done! Logs: $LOG_DIR"
