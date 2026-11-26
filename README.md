# DBVT

Official code for **DBVT: Dual-Branch Variable Temporal Modeling with Decoupled Representations for Irregular Clinical Time Series** (BIBM 2026).

DBVT is a model-agnostic dual-branch framework. A trajectory-preserving GRU branch models variable-specific temporal evolution, a factorized-attention Transformer branch models asynchronous cross-variable correlations, and a gated sum fusion adaptively combines the two representations. The same framework can be plugged into existing backbones by adding the GRU branch and gated fusion on top of the original model.

This repository contains:

- the full training pipeline for DBVT and all paper baselines;
- the plug-in variants for six backbones;
- the leave-variables-out robustness protocol;
- the branch-freezing / fusion ablation;
- the Optuna hyperparameter-tuning pipeline;
- the exact data splits and precomputed KEDGN variable embeddings used in the paper.

The qualitative t-SNE visualization (Fig. 4 in the paper) is intentionally not included.

## Repository layout

```
configs/                     YAML hyperparameters (final paper values)
data/process_scripts/        Dataset preprocessing and split generation
data/splits/                 Fixed 5-fold patient-level splits used in the paper
data/kedgn_embeddings/       Bio_ClinicalBERT variable embeddings for KEDGN
src/main.py                  Training / evaluation entry point
src/models/                  DBVT, baselines, and dual-branch wrapper
src/dataset/                 Dataset loading and model-specific adapters
scripts/run_main_comparison.sh   Table II: baselines, plug-ins, DBVT
scripts/run_lvo.sh               Table III: leave-variables-out on MIMIC-III
scripts/run_ablation.sh          Table IV: component ablation
scripts/run_optuna.sh            Hyperparameter tuning
scripts/run_all.sh               Run all of the above
scripts/aggregate_test_metrics.py
scripts/paired_ttest_plus_models.py
```

## Environment

The experiments in the paper used Python 3.10, PyTorch 2.4.0, CUDA 11.8, cuDNN 9.9.0, and four NVIDIA RTX 6000 Ada GPUs. A single GPU is sufficient to run every command; increase `GPUS` to use more.

```bash
conda create -n dbvt python=3.10 -y
conda activate dbvt

# Install the PyTorch build matching your CUDA toolkit first, e.g. CUDA 11.8:
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
python scripts/check_env.py
```

`scripts/check_env.py` verifies packages, CUDA availability, splits, and processed data. To use the CPU, pass `--device cpu` to `src/main.py` or set `DEVICE=cpu` when using the shell runners (slow but functional).

## Data preparation

Processed data is not shipped with the repository. Download the raw data from PhysioNet, then run the scripts below. The directory containing each processed pickle is `data/processed/`, and the fixed paper splits are already in `data/splits/`.

### PhysioNet 2012 (P12)

Download the [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) dataset and place `set-a/`, `set-b/`, `set-c/`, and `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt` under `data/raw/challenge-2012/`.

```bash
python data/process_scripts/preprocess_physionet_2012.py
```

### PhysioNet 2019 (P19)

Download the [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) dataset and place `training_setA/`, `training_setB/`, and `training_setC/` under `data/raw/challenge-2019/`.

```bash
python data/process_scripts/preprocess_physionet_2019.py
```

### MIMIC-III

MIMIC-III requires credentialed access: [https://physionet.org/content/mimiciii/1.4/](https://physionet.org/content/mimiciii/1.4/). Place the following CSV files under `data/raw/mimic-iii/`:

`ICUSTAYS.csv`, `PATIENTS.csv`, `CHARTEVENTS.csv`, `LABEVENTS.csv`, `OUTPUTEVENTS.csv`, `INPUTEVENTS_CV.csv`, `INPUTEVENTS_MV.csv`, `D_ITEMS.csv`, `ADMISSIONS.csv`.

```bash
python data/process_scripts/preprocess_mimic_iii.py
```

### Splits

The exact paper splits are included, so split generation is optional. To regenerate them:

```bash
python data/process_scripts/generate_splits.py --dataset physionet_2012 --val_ratio 0.2
python data/process_scripts/generate_splits.py --dataset physionet_2019 --val_ratio 0.2
python data/process_scripts/generate_splits.py --dataset mimic_iii --val_ratio 0.2
```

Raw-data locations can also be set with `P12_RAW_DATA`, `P19_RAW_DATA`, and `MIMIC_III_RAW_DATA`.

### KEDGN embeddings

The Bio_ClinicalBERT embeddings used by KEDGN are included in `data/kedgn_embeddings/` and are loaded automatically. To regenerate them:

```bash
python data/kedgn_embeddings/generate_embeddings.py --model_path /path/to/Bio_ClinicalBERT
```

## Quick start

After data preparation, run a single DBVT fold:

```bash
cd src
python main.py --dataset physionet_2012 --model_type dbvt --fold 1 --device cuda:0
```

Or run a one-epoch smoke test from the repository root:

```bash
PYTHON_BIN=python CUDA_VISIBLE_DEVICES=0 \
  DATASETS=physionet_2012 MODELS=dbvt FOLDS=1 MAX_EPOCHS=1 PATIENCE=1 \
  bash scripts/run_experiments.sh
```

`dbvt` is the paper name; internally it resolves to the historical `transformer+` variant and `configs/transformer+.yaml`.

## Reproducing the paper

All runners accept environment overrides: `PYTHON_BIN`, `DATASETS`, `MODELS`, `FOLDS`, `GPUS`, `MAX_EPOCHS`, `PATIENCE`, `SEED`, and, where relevant, `EXTRA_ARGS`. Results are written under `outputs/<dataset>/<model>|fold:<k>|...`, logs under `logs/`, and the aggregated mean/std table is printed at the end.

### Table II: main comparison and plug-in evaluation

Nine baselines (`tcn`, `sand`, `grud`, `raindrop`, `hipatch`, `kedgn`, `strats`, `warpformer`, `mtm`), their six plug-in variants (`raindrop+`, `hipatch+`, `kedgn+`, `strats+`, `warpformer+`, `mtm+`), and standalone `dbvt` on all three datasets:

```bash
GPUS=0,1,2,3 bash scripts/run_main_comparison.sh
```

The result summary is `outputs/` aggregated by `scripts/aggregate_test_metrics.py`. `scripts/paired_ttest_plus_models.py` can be used for paired comparisons between a backbone and its `+` variant.

### Table III: leave-variables-out robustness

Removes 10%, 30%, and 50% of variable types on MIMIC-III:

```bash
GPUS=0,1 bash scripts/run_lvo.sh
```

Each run is tagged as `<model>|drop:<rate>`; the aggregation groups them automatically.

### Table IV: component ablation

Runs four variants on all datasets: `full`, `no_gru` (freeze the GRU branch), `no_transformer` (freeze the Transformer branch), and `no_fusion` (concatenation instead of gated sum):

```bash
GPUS=0,1,2,3 bash scripts/run_ablation.sh
```

Runs are tagged as `dbvt|ablation:<mode>`.

### Hyperparameter tuning

The paper tuned DBVT and each baseline with Optuna for 100 trials per dataset on fold 1. The tuned values used for the reported tables are already in `configs/`; tuning is only needed if you want to re-run the search.

```bash
GPUS=0 bash scripts/run_optuna.sh
# or a single study:
python scripts/optuna_tune.py --model_type dbvt --dataset physionet_2012 \
  --n_trials 100 --folds 1 --gpu 0
```

Studies are stored in `optuna_studies/` and best parameters in `optuna_results/`.

### Everything

```bash
GPUS=0,1,2,3 bash scripts/run_all.sh
```

This runs Table II, Table IV, Table III, and Optuna sequentially. Full five-fold runs with 100 Optuna trials take a long time; use `DATASETS`, `MODELS`, `FOLDS`, and `N_TRIALS` to scope a run.

## Evaluated metrics

`AUROC`, `AUPRC`, `MinRP`, and `Accuracy` are reported. AUPRC is the primary metric for model selection and early stopping. The evaluation code is in `src/utils/evaluator.py`.

## Hardware notes

The paper used four 48 GB RTX 6000 Ada GPUs. Every experiment can be run on one GPU; reduce `GPUS` to `0` and, if memory is limited, lower `train_batch_size` in the corresponding YAML config. The default runner does not use CPU affinity pinning, so it works on ordinary workstations as well as multi-GPU servers.

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{dbvt2026,
  title   = {DBVT: Dual-Branch Variable Temporal Modeling with Decoupled Representations for Irregular Clinical Time Series},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year    = {2026}
}
```
