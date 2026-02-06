#!/bin/bash
# =============================================================================
# Optuna批量调参脚本 - 为所有模型和数据集组合运行超参数优化
# =============================================================================

# 激活虚拟环境
source your_path/.venv/bin/activate

# 配置
N_TRIALS=100  # 每个模型的试验次数
GPUS=(0 1 2 3)  # 可用的GPU
NUM_GPUS=${#GPUS[@]}

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================"
echo "Optuna Hyperparameter Tuning - All Models"
echo "Trials per model: $N_TRIALS"
echo "GPUs: ${GPUS[@]}"
echo "============================================"
echo ""

# 定义要调参的模型和数据集 (9个模型 x 3个数据集)
MODELS=("dbvt")
# MODELS=("strats" "istrats" "grud" "kedgn" "hipatch" "raindrop" "warpformer" "mtm" "dbvt")
DATASETS=("physionet_2019" "mimic_iii")

# 生成任务列表
declare -a TASKS=()
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        TASKS+=("${dataset}|${model}")
    done
done

TOTAL_TASKS=${#TASKS[@]}
echo -e "${GREEN}Total tuning tasks: $TOTAL_TASKS${NC}"
echo ""

# 运行单个调参任务
run_tuning_task() {
    local task="$1"
    local gpu_id="$2"
    
    IFS='|' read -r dataset model <<< "$task"
    
    local log_file="${PROJECT_ROOT}/optuna_results/${dataset}_${model}_tuning.log"
    mkdir -p "${PROJECT_ROOT}/optuna_results"
    
    echo -e "${BLUE}[GPU:$gpu_id]${NC} Starting tuning: ${dataset}/${model}"
    
    cd "${PROJECT_ROOT}/scripts"
    if python optuna_tune.py \
        --model_type "$model" \
        --dataset "$dataset" \
        --n_trials "$N_TRIALS" \
        --gpu "$gpu_id" \
        --max_epochs 20 \
        --patience 5 > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ [GPU:$gpu_id]${NC} Completed tuning: ${dataset}/${model}"
        return 0
    else
        echo -e "${RED}✗ [GPU:$gpu_id]${NC} Failed tuning: ${dataset}/${model}"
        return 1
    fi
}

export -f run_tuning_task
export PROJECT_ROOT RED GREEN YELLOW BLUE NC N_TRIALS

# 检查是否安装了GNU parallel
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for parallel tuning..."
    echo ""
    
    # 使用GNU parallel并行调参（正确映射到GPUS数组）
    printf '%s\n' "${TASKS[@]}" | parallel -j $NUM_GPUS \
        'gpus_arr=('"${GPUS[*]}"'); gpu_id=${gpus_arr[$(( ({%} - 1) % '"$NUM_GPUS"' ))]}; run_tuning_task {} $gpu_id'
else
    echo -e "${YELLOW}GNU parallel not found. Running sequentially...${NC}"
    echo ""
    
    # 顺序执行
    task_index=0
    for task in "${TASKS[@]}"; do
        gpu_id=${GPUS[$((task_index % NUM_GPUS))]}
        
        # 等待GPU空闲
        while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
            sleep 5
        done
        
        run_tuning_task "$task" "$gpu_id" &
        task_index=$((task_index + 1))
    done
    
    wait
fi

echo ""
echo "============================================"
echo "Tuning Completed"
echo "============================================"
echo "Results saved to: ${PROJECT_ROOT}/optuna_results/"
echo ""
echo "Best parameters for each model:"
echo "============================================"

# 显示所有最佳参数
for result_file in "${PROJECT_ROOT}/optuna_results"/*_best_params.txt; do
    if [ -f "$result_file" ]; then
        echo ""
        cat "$result_file"
        echo "--------------------------------------------"
    fi
done

echo ""
echo "Done!"
