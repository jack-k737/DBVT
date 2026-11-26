#!/bin/bash
# Hyperparameter tuning with Optuna (paper setting: 100 trials per model/dataset).
#
# Environment variables:
#   PYTHON_BIN  Python interpreter (default: python3)
#   MODELS      Comma-separated model types (default: paper methods)
#   DATASETS    Comma-separated datasets (default: all three)
#   N_TRIALS    Trials per model/dataset (default: 100)
#   TUNE_FOLDS  --folds value for Optuna (default: 1)
#   GPU         GPU id used by the Optuna process (default: 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODELS="${MODELS:-tcn,sand,grud,raindrop,hipatch,kedgn,strats,warpformer,mtm,dbvt}"
DATASETS="${DATASETS:-physionet_2012,physionet_2019,mimic_iii}"
N_TRIALS="${N_TRIALS:-100}"
TUNE_FOLDS="${TUNE_FOLDS:-1}"
GPU="${GPU:-0}"

mkdir -p "$PROJECT_ROOT/optuna_results"
for dataset in ${DATASETS//,/ }; do
    for model in ${MODELS//,/ }; do
        log_file="$PROJECT_ROOT/optuna_results/${dataset}_${model}_tuning.log"
        echo ">>> Tuning $model on $dataset (trials=$N_TRIALS, folds=$TUNE_FOLDS)"
        cd "$PROJECT_ROOT/scripts"
        "$PYTHON_BIN" optuna_tune.py \
            --model_type "$model" \
            --dataset "$dataset" \
            --n_trials "$N_TRIALS" \
            --folds "$TUNE_FOLDS" \
            --gpu "$GPU" \
            --max_epochs 20 \
            --patience 5 2>&1 | tee "$log_file"
    done
done

echo "Best parameters are in $PROJECT_ROOT/optuna_results/"
