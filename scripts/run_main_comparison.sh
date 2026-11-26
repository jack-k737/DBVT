#!/bin/bash
# Table II: main comparison + plug-in evaluation.
# Reproduces the 9 baselines, their 6 plug-in variants, and standalone DBVT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODELS="${MODELS:-tcn,sand,grud,raindrop,raindrop+,hipatch,hipatch+,kedgn,kedgn+,strats,strats+,warpformer,warpformer+,mtm,mtm+,dbvt}"
export DATASETS="${DATASETS:-physionet_2012,physionet_2019,mimic_iii}"
export FOLDS="${FOLDS:-1,2,3,4,5}"
export GPUS="${GPUS:-0}"

exec bash "$SCRIPT_DIR/run_experiments.sh"
