from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


_METRIC_RE = re.compile(
    r"AUROC:\s*([0-9]*\.?[0-9]+)\s*\|\s*AUPRC:\s*([0-9]*\.?[0-9]+)\s*\|\s*MinRP:\s*([0-9]*\.?[0-9]+)"
)


@dataclass(frozen=True)
class Record:
    dataset: str
    experiment: str
    auroc: float
    auprc: float
    minrp: float


def _safe_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=1))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.array(values, dtype=float)))


def _parse_final_test_metrics(log_path: Path) -> List[Tuple[float, float, float]]:
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return []

    lines = text.splitlines()
    out: List[Tuple[float, float, float]] = []
    for i, line in enumerate(lines):
        if "Final Test Results" not in line:
            continue

        # Usually metrics are printed on the next line, but be tolerant.
        for j in range(i, min(i + 6, len(lines))):
            m = _METRIC_RE.search(lines[j])
            if m is None:
                continue
            out.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
            break

    if out:
        return out

    # Fallback: if the marker is missing, take the last metrics-looking line.
    matches = list(_METRIC_RE.finditer(text))
    if matches:
        m = matches[-1]
        return [(float(m.group(1)), float(m.group(2)), float(m.group(3)))]

    return []


def _iter_log_files(outputs_dir: Path) -> Iterable[Path]:
    yield from outputs_dir.rglob("log.txt")


def _experiment_key(outputs_dir: Path, dataset: str, log_path: Path) -> str:
    rel = log_path.relative_to(outputs_dir / dataset)
    exp = rel.parent.as_posix()
    return exp


def _group_key(experiment: str) -> str:
    """Extract model name from experiment path.
    
    Examples:
        'mtm|fold:1' -> 'mtm'
        'warpformer|fold:2' -> 'warpformer'
        'medbivt|run:1o5' -> 'medbivt' (legacy format)
        'mtm_optuna' -> 'mtm_optuna'
    """
    first = experiment.split("/")[0]
    # Support new format: model_type|fold:X
    if "|fold:" in first:
        return first.split("|fold:")[0]
    # Support legacy format: model_type|run:XoY
    if "|run:" in first:
        return first.split("|run:")[0]
    return first


def _format_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.4f}"


def _format_mean_pm_std(mean: float, std: float) -> str:
    return f"{_format_float(mean)} ± {_format_float(std)}"


def _print_table(rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r_i, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if r_i == 0:
            print("  ".join("-" * w for w in widths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="../outputs")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise SystemExit(f"outputs_dir not found: {outputs_dir}")

    records: List[Record] = []
    for log_path in _iter_log_files(outputs_dir):
        try:
            dataset = log_path.relative_to(outputs_dir).parts[0]
        except Exception:
            continue

        metrics_list = _parse_final_test_metrics(log_path)
        if not metrics_list:
            continue

        experiment = _experiment_key(outputs_dir, dataset, log_path)
        
        # Skip optuna tuning experiments
        if '_optuna' in experiment:
            continue
        
        for auroc, auprc, minrp in metrics_list:
            records.append(
                Record(dataset=dataset, experiment=experiment, auroc=auroc, auprc=auprc, minrp=minrp)
            )

    if not records:
        raise SystemExit(f"No test metrics found under: {outputs_dir}")

    by_dataset_group: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
    for r in records:
        group = _group_key(r.experiment)
        by_dataset_group[(r.dataset, group)].append(r)

    print("# Test metrics (mean/std over folds)")
    datasets = sorted({d for d, _ in by_dataset_group.keys()})
    for d_i, dataset in enumerate(datasets):
        if d_i > 0:
            print()
        print("=" * 80)
        print(f"DATASET: {dataset}")
        print("=" * 80)

        rows = [["model", "n", "auroc", "auprc", "minrp"]]
        items = [(g, rs) for (ds, g), rs in by_dataset_group.items() if ds == dataset]
        for group, rs in sorted(items, key=lambda x: x[0]):
            aurocs = [x.auroc for x in rs]
            auprcs = [x.auprc for x in rs]
            minrps = [x.minrp for x in rs]

            auroc_mean, auroc_std = _safe_mean(aurocs), _safe_std(aurocs)
            auprc_mean, auprc_std = _safe_mean(auprcs), _safe_std(auprcs)
            minrp_mean, minrp_std = _safe_mean(minrps), _safe_std(minrps)
            rows.append(
                [
                    group,
                    str(len(rs)),
                    _format_mean_pm_std(auroc_mean, auroc_std),
                    _format_mean_pm_std(auprc_mean, auprc_std),
                    _format_mean_pm_std(minrp_mean, minrp_std),
                ]
            )
        _print_table(rows)


if __name__ == "__main__":
    main()
