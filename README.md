# DBVT: Dual-Branch Variable-Temporal Network for Irregular Medical Time Series

This is the official implementation of the paper:

**DBVT: A Dual-Branch Variable-Temporal Network for Irregular Clinical Time Series**

*Submitted to MICCAI 2026*

> The DBVT model implementation will be released upon paper acceptance. Currently, this repository includes the training framework, all baseline implementations, and experiment results for reproducibility.

## Overview

We propose **DBVT**, a dual-branch architecture that jointly captures **variable correlation** and **temporal dependency** in irregularly sampled clinical time series. The two branches (GRU branch for temporal modeling and Transformer branch for cross-variable attention) run in parallel and are fused for downstream classification.

We evaluate DBVT and 9 baselines on three clinical benchmarks under 5-fold cross-validation.

## Datasets

| Dataset | Task | # Samples | # Variables | Source |
|---------|------|-----------|-------------|--------|
| PhysioNet 2012 | In-hospital mortality | ~11,988 | 36 | [link](https://physionet.org/content/challenge-2012/) |
| PhysioNet 2019 | Sepsis early prediction | ~38575 | 34 | [link](https://physionet.org/content/challenge-2019/) |
| MIMIC-III | In-hospital mortality | ~52,871 | 127 | [link](https://physionet.org/content/mimiciii/)  |

## Experiment Results

### Viewing Results

Training logs for all experiments are provided in [`lab1/outputs/`](lab1/outputs/). Each `log.txt` records the full training process and final test metrics.

```
lab1/outputs/
├── physionet_2012/
│   ├── ours1|fold:{1-5}/log.txt                      # DBVT (ours)
│   ├── ablation_gru_only|fold:{1-5}/log.txt           # Ablation: GRU branch only
│   ├── ablation_transformer_only|fold:{1-5}/log.txt   # Ablation: Transformer branch only
│   └── ablation_no_aux_loss|fold:{1-5}/log.txt        # Ablation: without auxiliary loss
├── physionet_2019/
│   └── ours1|fold:{1-5}/log.txt
└── mimic_iii/
    └── ours1|fold:{1-5}/log.txt
```

To inspect the final test result of a single run, check the end of any `log.txt`:

```
Final Test Results:
  AUROC: 0.8761 | AUPRC: 0.6274 | MinRP: 0.5845 | Accuracy: 0.7797
```

### Aggregating Results

We provide a script to compute mean and standard deviation across all folds:

```bash
python scripts/aggregate_test_metrics.py --outputs_dir lab1/outputs
python scripts/aggregate_test_metrics.py --outputs_dir outputs
```

Example output:

```
================================================================================
DATASET: mimic_iii
================================================================================
model  n   auroc            auprc            minrp
-----  --  ---------------  ---------------  ---------------
ours1  10  0.9153 ± 0.0027  0.6961 ± 0.0092  0.6159 ± 0.0104

================================================================================
DATASET: physionet_2012
================================================================================
model                      n  auroc            auprc            minrp
-------------------------  -  ---------------  ---------------  ---------------
ablation_gru_only          5  0.8547 ± 0.0157  0.5320 ± 0.0461  0.5179 ± 0.0289
ablation_no_aux_loss       5  0.8627 ± 0.0155  0.5510 ± 0.0408  0.5286 ± 0.0368
ablation_transformer_only  5  0.8612 ± 0.0132  0.5343 ± 0.0348  0.5215 ± 0.0401
ours1                      5  0.8690 ± 0.0138  0.5608 ± 0.0450  0.5342 ± 0.0348

================================================================================
DATASET: physionet_2019
================================================================================
model  n  auroc            auprc            minrp
-----  -  ---------------  ---------------  ---------------
ours1  5  0.8842 ± 0.0154  0.4881 ± 0.0454  0.4822 ± 0.0292
```

### Evaluation Metrics

- **AUROC**: Area Under Receiver Operating Characteristic Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **MinRP**: Minimum of Recall and Precision (at optimal threshold)

## Getting Started

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, pandas, scikit-learn

### Data Preparation

1. Download the raw datasets from the links in the [Datasets](#datasets) section.

   > MIMIC-III requires credentialed access via [PhysioNet](https://physionet.org/content/mimiciii/).

2. Run the preprocessing scripts:

```bash
cd data/process_scripts
python preprocess_physionet_2012.py
python preprocess_physionet_2019.py
python preprocess_mimic_iii.py
```

3. Generate cross-validation splits (pre-generated splits are already provided in `data/splits/`):

```bash
python generate_splits.py
```

### Training

**Single model on a single fold:**

```bash
cd src
python main.py \
    --dataset physionet_2012 \
    --model_type grud \
    --fold 1 \
    --device cuda:0
```

**Run all baselines (5-fold CV, multi-GPU parallel):**

```bash
bash scripts/run_all_models.sh
```

**Hyperparameter tuning with Optuna:**

```bash
bash scripts/run_optuna_all.sh
```

### Available Models

| Model | `--model_type` | Reference |
|-------|---------------|-----------|
| GRU-D | `grud` | [Che et al., 2018](https://doi.org/10.1038/s41598-018-24271-9) |
| TCN | `tcn` | [Bai et al., 2018](https://arxiv.org/abs/1803.01271) |
| STraTS | `strats` | [Tipirneni & Reddy, 2022](https://doi.org/10.1145/3534678.3539094) |
| SAND | `sand` | [Hyland et al., 2020](https://doi.org/10.1007/s10994-019-05862-7) |
| Raindrop | `raindrop` | [Zhang et al., ICLR 2022](https://openreview.net/forum?id=Kwm8I7dU-l5) |
| Warpformer | `warpformer` | [Zhang et al., KDD 2023](https://arxiv.org/abs/2306.09368) |
| HiPatch | `hipatch` | [Luo et al., ICML 2025](https://proceedings.mlr.press/v267/luo25r.html) |
| KEDGN | `kedgn` | [Luo et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7c04aea54c2a60a632a47bd451cd2849-Abstract-Conference.html) |
| MTM | `mtm` | [Zhang et al., KDD 2025](https://github.com/zshhans/MTM) |
| **DBVT (Ours)** | `dbvt` | *This paper* |

Model-specific hyperparameters are loaded from `configs/{model_type}.yaml` and can be overridden via command line.

## Project Structure

```
.
├── configs/                          # Model configs (YAML)
├── src/
│   ├── main.py                       # Training & evaluation entry point
│   ├── models/                       # Model implementations (baselines)
│   ├── dataset/                      # Data loading & model-specific adapters
│   └── utils/                        # Logger, evaluator, config loader
├── scripts/
│   ├── run_all_models.sh             # Batch training script
│   ├── run_optuna_all.sh             # Batch hyperparameter tuning
│   ├── optuna_tune.py                # Optuna tuning script
│   └── aggregate_test_metrics.py     # Aggregate test metrics across folds
├── data/
│   ├── process_scripts/              # Data preprocessing scripts
│   └── splits/                       # Pre-generated 5-fold CV splits
└── lab1/outputs/                     # Experiment logs (training + test results)
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{dbvt2026,
  title     = {Dual-Branch Variable-Temporal Network for Irregular Medical Time Series Classification},
  author    = {Anonymous},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

## Acknowledgement

We appreciate the following repositories for providing valuable code:

- [Warpformer](https://github.com/imJiawen/Warpformer) (KDD 2023)
- [KEDGN](https://github.com/easonLuo2001/KEDGN) (NeurIPS 2024)
- [STraTS](https://github.com/sindhura97/STraTS) (KDD 2022)


## License

This project is licensed under the [MIT License](LICENSE).