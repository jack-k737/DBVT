# DBVT

**DBVT: Dual-Branch variable temporal modeling with decoupled representations for irregular clinical time series**

This repository is a **partial public release** of the DBVT project.

The **full DBVT source code** will be released after the paper is accepted by **BIBM 2026**. The current repository only includes the parts needed for data preparation, split reproduction, baseline reference code, and inspection of reported experiment results.

## What Is Included

- dataset preprocessing scripts
- dataset split files
- MIMIC-III extraction / preprocessing code
- baseline model reference implementations under `src/models/`
- utility scripts under `scripts/`
- experiment-result aggregation code

## What Is Not Included Yet

The current public release does **not** include the full DBVT training pipeline or other paper-specific private components, including the proprietary wrapper / integration code used for the proposed model.

## Repository Layout

- `data/process_scripts/`: preprocessing scripts, including MIMIC-III extraction / preprocessing
- `data/splits/`: released split files
- `src/models/`: baseline model reference implementations
- `scripts/`: experiment scripts and result aggregation utilities
- `scripts/aggregate_test_metrics.py`: summarize experiment logs under an outputs directory

## View Experiment Results

Use the following command pattern to summarize reported experiment results:

```bash
python scripts/aggregate_test_metrics.py --outputs_dir outputs/lab-xxxx/outputs
```

Examples:

```bash
python scripts/aggregate_test_metrics.py --outputs_dir outputs
python scripts/aggregate_test_metrics.py --outputs_dir lab-ablation/outputs
python scripts/aggregate_test_metrics.py --outputs_dir lab-dropout/outputs
```

Historical result folders may still use the internal alias `transformer+`; the aggregation summary normalizes that name to `dbvt`.

## Notes

- `scripts/` is retained for archival and experiment-management reference.
- The public `src/main.py` is intentionally a placeholder in this partial release.
- Local experiment outputs, checkpoints, and processed data artifacts are excluded from Git tracking.
