"""
Generate stratified train/val/test splits for medical time series datasets.

This script keeps the same 5-fold JSON format consumed by the training code,
but makes the test fold class balance much more consistent by using stratified
splitting. With default arguments it overwrites the original split file name:
    data/splits/{dataset}_5fold.json

Usage:
    python data/process_scripts/generate_splits_stratified.py --dataset physionet_2012
    python data/process_scripts/generate_splits_stratified.py --dataset mimic_iii
    python data/process_scripts/generate_splits_stratified.py --dataset physionet_2019
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUTPUT_DIR = DATA_DIR / "splits"
N_FOLDS = 5


def load_processed_data(dataset_name: str, processed_dir: Path):
    """Load processed data and label metadata."""
    filepath = processed_dir / f"{dataset_name}.pkl"
    with filepath.open("rb") as handle:
        loaded = pickle.load(handle)

    if len(loaded) == 5:
        data, oc, _, _, _ = loaded
    elif len(loaded) == 2:
        data, oc = loaded
    else:
        raise ValueError(f"Unexpected pickle format with {len(loaded)} elements")

    subject_mapping = None
    if "SUBJECT_ID" in oc.columns:
        subject_mapping = oc.set_index("ts_id")["SUBJECT_ID"].to_dict()

    if "label" not in oc.columns:
        raise ValueError("Processed cohort table must contain a 'label' column.")
    label_mapping = oc.set_index("ts_id")["label"].astype(int).to_dict()

    return data, oc, subject_mapping, label_mapping


def filter_data_by_dataset(data, oc, dataset_name: str) -> np.ndarray:
    """Apply dataset-specific filtering and return valid sample ids."""
    if dataset_name == "mimic_iii":
        data_filtered = data.loc[(data.minute >= 0) & (data.minute <= 24 * 60)]
        valid_ts_ids = set(data_filtered.ts_id.unique())
    elif dataset_name == "physionet_2012":
        black_list = {
            140501, 150649, 140936, 143656, 141264, 145611,
            142998, 147514, 142731, 150309, 155655, 156254,
        }
        valid_ts_ids = set(data.ts_id.unique()) - black_list
    elif dataset_name == "physionet_2019":
        patient_stats = data.groupby("ts_id")["minute"].agg(["max", "nunique"])
        patient_stats["max_hours"] = patient_stats["max"] / 60
        valid_patients = patient_stats[
            (patient_stats["max_hours"] <= 60) &
            (patient_stats["nunique"] > 3)
        ].index
        valid_ts_ids = set(valid_patients)
    else:
        valid_ts_ids = set(data.ts_id.unique())

    labeled_ts_ids = set(oc.ts_id.unique())
    valid_ts_ids = valid_ts_ids & labeled_ts_ids
    return np.array(sorted(valid_ts_ids), dtype=int)


def build_labels(ts_ids: np.ndarray, label_mapping: dict[int, int]) -> np.ndarray:
    labels = np.array([int(label_mapping[int(ts_id)]) for ts_id in ts_ids], dtype=int)
    unique_labels = set(labels.tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, got: {sorted(unique_labels)}")
    if len(unique_labels) < 2:
        raise ValueError("Need at least two classes to build stratified splits.")
    return labels


def max_feasible_splits(labels: np.ndarray, groups: np.ndarray | None, desired_splits: int) -> int:
    class_counts = np.bincount(labels, minlength=2)
    non_zero_class_counts = class_counts[class_counts > 0]
    if len(non_zero_class_counts) == 0:
        return 0

    if groups is None:
        group_cap = len(labels)
    else:
        group_cap = len(np.unique(groups))

    return max(0, min(desired_splits, int(non_zero_class_counts.min()), int(group_cap)))


def choose_group_validation_split(
    ts_ids: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    desired_splits = max(int(round(1.0 / val_ratio)), 2)
    feasible_splits = max_feasible_splits(labels, groups, desired_splits)

    if feasible_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=feasible_splits,
            shuffle=True,
            random_state=seed,
        )
        best_pair = None
        best_score = None
        overall_pos_ratio = float(labels.mean())

        for train_idx, val_idx in splitter.split(ts_ids, labels, groups=groups):
            val_size_ratio = len(val_idx) / len(ts_ids)
            val_pos_ratio = float(labels[val_idx].mean())
            score = (
                abs(val_size_ratio - val_ratio),
                abs(val_pos_ratio - overall_pos_ratio),
                len(val_idx),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_pair = (train_idx, val_idx)

        if best_pair is not None:
            return best_pair

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    return next(splitter.split(ts_ids, labels, groups=groups))


def generate_stratified_5fold_splits(
    ts_ids: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    subject_mapping: dict[int, int] | None = None,
    seed: int = 2026,
) -> dict:
    """Generate 5-fold splits with stratified test folds."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    groups = None
    if subject_mapping is not None:
        groups = np.array([subject_mapping[int(ts_id)] for ts_id in ts_ids])
        feasible_splits = max_feasible_splits(labels, groups, N_FOLDS)
        if feasible_splits < N_FOLDS:
            raise ValueError(
                "Not enough grouped positive/negative examples to build 5 stratified folds. "
                f"Maximum feasible splits: {feasible_splits}"
            )
        test_splitter = StratifiedGroupKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed,
        )
        test_splits = test_splitter.split(ts_ids, labels, groups=groups)
        strategy = "stratified_group_kfold"
    else:
        feasible_splits = max_feasible_splits(labels, None, N_FOLDS)
        if feasible_splits < N_FOLDS:
            raise ValueError(
                "Not enough positive/negative examples to build 5 stratified folds. "
                f"Maximum feasible splits: {feasible_splits}"
            )
        test_splitter = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=seed,
        )
        test_splits = test_splitter.split(ts_ids, labels)
        strategy = "stratified_kfold"

    folds = []
    for fold_idx, (trainval_idx, test_idx) in enumerate(test_splits, start=1):
        trainval_ids = ts_ids[trainval_idx]
        trainval_labels = labels[trainval_idx]
        test_ids = ts_ids[test_idx]

        if groups is not None:
            trainval_groups = groups[trainval_idx]
            train_idx, val_idx = choose_group_validation_split(
                trainval_ids,
                trainval_labels,
                trainval_groups,
                val_ratio,
                seed + fold_idx,
            )
        else:
            val_splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_ratio,
                random_state=seed + fold_idx,
            )
            train_idx, val_idx = next(val_splitter.split(trainval_ids, trainval_labels))

        folds.append({
            "train_ids": trainval_ids[train_idx].astype(int).tolist(),
            "val_ids": trainval_ids[val_idx].astype(int).tolist(),
            "test_ids": test_ids.astype(int).tolist(),
        })

    return {
        "folds": folds,
        "n_folds": N_FOLDS,
        "strategy": strategy,
        "seed": seed,
        "val_ratio": val_ratio,
    }


def summarize_fold(name: str, ids: list[int], label_mapping: dict[int, int]) -> tuple[int, int, float]:
    total = len(ids)
    positives = sum(int(label_mapping[int(ts_id)]) for ts_id in ids)
    ratio = positives / total if total else 0.0
    print(f"    {name:<5} {total:>5} samples | pos {positives:>4} | pos_ratio {ratio:.4f}")
    return total, positives, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stratified 5-fold CV splits")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["physionet_2012", "physionet_2019", "mimic_iii"],
        help="Dataset name",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio from trainval data (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed (default: 2026)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed data directory (default: {DEFAULT_PROCESSED_DIR})",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {args.dataset}...")
    data, oc, subject_mapping, label_mapping = load_processed_data(args.dataset, args.processed_dir)

    print("Filtering data...")
    valid_ts_ids = filter_data_by_dataset(data, oc, args.dataset)
    labels = build_labels(valid_ts_ids, label_mapping)
    print(f"Valid samples after filtering: {len(valid_ts_ids)}")
    print(f"Overall positive ratio: {labels.mean():.4f} ({labels.sum()}/{len(labels)})")
    print(
        "Generating 5-fold stratified cross validation splits"
        + (" with grouped subjects..." if subject_mapping is not None else "...")
    )
    splits = generate_stratified_5fold_splits(
        valid_ts_ids,
        labels,
        args.val_ratio,
        subject_mapping=subject_mapping,
        seed=args.seed,
    )

    output_path = args.output_dir / f"{args.dataset}_5fold.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    print(f"Splits saved to {output_path}")

    print(f"{'=' * 72}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of folds: {N_FOLDS}")
    print(f"Validation ratio: {args.val_ratio:.2%} of trainval data")
    print(f"Split strategy: {splits['strategy']}")
    print(f"Output file: {output_path}")
    print()

    test_ratios = []
    for idx, fold in enumerate(splits["folds"], start=1):
        print(f"Fold {idx}:")
        summarize_fold("train", fold["train_ids"], label_mapping)
        summarize_fold("val", fold["val_ids"], label_mapping)
        _, _, test_ratio = summarize_fold("test", fold["test_ids"], label_mapping)
        test_ratios.append(test_ratio)

    print()
    print(
        "Test fold positive ratios: "
        + ", ".join(f"{ratio:.4f}" for ratio in test_ratios)
        + f" | range={max(test_ratios) - min(test_ratios):.4f}"
    )
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
