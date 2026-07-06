from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
DEFAULT_SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"


def load_labels(dataset: str, processed_dir: Path) -> dict[int, int]:
    with (processed_dir / f"{dataset}.pkl").open("rb") as handle:
        loaded = pickle.load(handle)
    if len(loaded) == 5:
        _, oc, _, _, _ = loaded
    elif len(loaded) == 2:
        _, oc = loaded
    else:
        raise ValueError(f"Unexpected pickle format with {len(loaded)} elements")
    return oc.set_index("ts_id")["label"].astype(int).to_dict()


def digest_ids(ids: list[int]) -> str:
    payload = ",".join(str(int(x)) for x in sorted(ids)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def summarize_split(name: str, ids: list[int], label_map: dict[int, int]) -> tuple[int, int, float, str]:
    total = len(ids)
    pos = sum(label_map[int(ts_id)] for ts_id in ids)
    ratio = pos / total if total else 0.0
    digest = digest_ids(ids)
    print(
        f"  {name:<5} n={total:<6} pos={pos:<6} ratio={ratio:.6%} digest={digest}"
    )
    return total, pos, ratio, digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Report label stats for an existing split JSON")
    parser.add_argument("--dataset", required=True, choices=["physionet_2012", "physionet_2019", "mimic_iii"])
    parser.add_argument("--splits_dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    split_path = args.splits_dir / f"{args.dataset}_5fold.json"
    with split_path.open("r", encoding="utf-8") as handle:
        splits = json.load(handle)

    label_map = load_labels(args.dataset, args.processed_dir)
    folds = splits.get("folds", [])
    if not folds:
        raise ValueError(f"No 'folds' field found in {split_path}")

    print(f"split_file: {split_path}")
    print(f"n_folds: {splits.get('n_folds', len(folds))}")

    test_sets = []
    val_sets = []
    for idx, fold in enumerate(folds, start=1):
        print(f"\nFold {idx}")
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]
        test_ids = fold["test_ids"]
        summarize_split("train", train_ids, label_map)
        summarize_split("val", val_ids, label_map)
        summarize_split("test", test_ids, label_map)
        test_sets.append(set(test_ids))
        val_sets.append(set(val_ids))

    print("\nPairwise equality checks")
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            same_test = test_sets[i] == test_sets[j]
            same_val = val_sets[i] == val_sets[j]
            overlap_test = len(test_sets[i] & test_sets[j])
            overlap_val = len(val_sets[i] & val_sets[j])
            print(
                f"  fold{i+1} vs fold{j+1}: "
                f"same_test={same_test} overlap_test={overlap_test}; "
                f"same_val={same_val} overlap_val={overlap_val}"
            )


if __name__ == "__main__":
    main()
