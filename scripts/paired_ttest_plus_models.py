#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


_FINAL_METRIC_RE = re.compile(
    r"AUROC:\s*([0-9]*\.?[0-9]+)\s*\|\s*"
    r"AUPRC:\s*([0-9]*\.?[0-9]+)\s*\|\s*"
    r"MinRP:\s*([0-9]*\.?[0-9]+)"
    r"(?:\s*\|\s*Accuracy:\s*([0-9]*\.?[0-9]+))?"
)


@dataclass(frozen=True)
class FoldMetrics:
    dataset: str
    model: str
    fold: int
    auroc: float
    auprc: float
    minrp: float
    accuracy: float


@dataclass
class GroupResult:
    dataset: str
    base_model: str
    plus_model: str
    metric: str
    n_folds: int
    wins: int
    base_mean: float
    base_std: float
    plus_mean: float
    plus_std: float
    diff_mean: float
    diff_std: float
    t_stat: float
    p_value: float
    p_holm: float = float("nan")


def _format_float(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{x:.4f}"


def _format_p_value(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    if x == 0.0:
        return "0"
    if x < 1e-6:
        return f"{x:.2e}"
    if x < 1e-4:
        return f"{x:.6e}"
    return f"{x:.8f}"


def _format_mean_pm_std(mean: float, std: float, signed: bool = False) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "nan"
    mean_str = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    return f"{mean_str} ± {std:.4f}"


def _print_table(rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for idx, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def _iter_log_files(outputs_dir: Path) -> Iterable[Path]:
    yield from outputs_dir.rglob("log.txt")


def _parse_model_and_fold(log_path: Path) -> Tuple[str, int] | None:
    parent = log_path.parent.name
    if "|fold:" not in parent:
        return None
    model, fold_str = parent.split("|fold:", 1)
    try:
        return model, int(fold_str)
    except ValueError:
        return None


def _parse_final_metrics(log_path: Path) -> Tuple[float, float, float, float] | None:
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Final Test Results" not in line:
            continue
        for j in range(i, min(i + 6, len(lines))):
            m = _FINAL_METRIC_RE.search(lines[j])
            if m is None:
                continue
            accuracy = float(m.group(4)) if m.group(4) is not None else float("nan")
            return float(m.group(1)), float(m.group(2)), float(m.group(3)), accuracy

    matches = list(_FINAL_METRIC_RE.finditer(text))
    if matches:
        m = matches[-1]
        accuracy = float(m.group(4)) if m.group(4) is not None else float("nan")
        return float(m.group(1)), float(m.group(2)), float(m.group(3)), accuracy
    return None


def _load_records(outputs_dir: Path) -> Dict[Tuple[str, str, int], FoldMetrics]:
    records: Dict[Tuple[str, str, int], FoldMetrics] = {}
    for log_path in _iter_log_files(outputs_dir):
        try:
            dataset = log_path.relative_to(outputs_dir).parts[0]
        except Exception:
            continue

        parsed = _parse_model_and_fold(log_path)
        if parsed is None:
            continue
        model, fold = parsed
        metrics = _parse_final_metrics(log_path)
        if metrics is None:
            continue

        records[(dataset, model, fold)] = FoldMetrics(
            dataset=dataset,
            model=model,
            fold=fold,
            auroc=metrics[0],
            auprc=metrics[1],
            minrp=metrics[2],
            accuracy=metrics[3],
        )
    return records


def _discover_pairs(
    records: Dict[Tuple[str, str, int], FoldMetrics],
    datasets: set[str] | None = None,
    models: set[str] | None = None,
) -> Dict[Tuple[str, str], List[Tuple[FoldMetrics, FoldMetrics]]]:
    grouped: Dict[Tuple[str, str], List[Tuple[FoldMetrics, FoldMetrics]]] = defaultdict(list)
    keys = sorted(records.keys())
    for dataset, plus_model, fold in keys:
        if not plus_model.endswith("+") or plus_model.endswith("++"):
            continue
        base_model = plus_model[:-1]
        if datasets is not None and dataset not in datasets:
            continue
        if models is not None and base_model not in models and plus_model not in models:
            continue

        base_key = (dataset, base_model, fold)
        plus_key = (dataset, plus_model, fold)
        if base_key not in records or plus_key not in records:
            continue
        grouped[(dataset, base_model)].append((records[base_key], records[plus_key]))

    for key in grouped:
        grouped[key].sort(key=lambda pair: pair[0].fold)
    return grouped


def _paired_ttest(base: Sequence[float], plus: Sequence[float], alternative: str) -> Tuple[float, float]:
    base_arr = np.array(base, dtype=float)
    plus_arr = np.array(plus, dtype=float)
    if len(base_arr) != len(plus_arr):
        raise ValueError("base and plus arrays must have equal length")
    if len(base_arr) < 2:
        return float("nan"), float("nan")

    diff = plus_arr - base_arr
    if np.allclose(diff, diff[0]):
        if np.isclose(diff[0], 0.0):
            return 0.0, 1.0
        t_stat = math.inf if diff[0] > 0 else -math.inf
        if alternative == "greater":
            p_value = 0.0 if diff[0] > 0 else 1.0
        elif alternative == "less":
            p_value = 0.0 if diff[0] < 0 else 1.0
        else:
            p_value = 0.0
        return t_stat, p_value

    res = stats.ttest_rel(plus_arr, base_arr, alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def _metric_values(
    pairs: Sequence[Tuple[FoldMetrics, FoldMetrics]],
    metric: str,
) -> Tuple[List[float], List[float]]:
    base_vals: List[float] = []
    plus_vals: List[float] = []
    for base, plus in pairs:
        base_val = getattr(base, metric)
        plus_val = getattr(plus, metric)
        if not np.isfinite(base_val) or not np.isfinite(plus_val):
            continue
        base_vals.append(base_val)
        plus_vals.append(plus_val)
    return base_vals, plus_vals


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=1))


def _holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (n - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def _conclusion(diff_mean: float, p_value: float, alpha: float, alternative: str) -> str:
    if not np.isfinite(diff_mean) or not np.isfinite(p_value):
        return "insufficient folds"
    if alternative == "greater":
        if diff_mean > 0 and p_value < alpha:
            return "+ significantly better"
        if diff_mean > 0:
            return "+ better, not significant"
        return "+ not better"
    if alternative == "less":
        if diff_mean < 0 and p_value < alpha:
            return "+ significantly worse"
        if diff_mean < 0:
            return "+ worse, not significant"
        return "+ not worse"
    if p_value < alpha:
        return "significant difference"
    return "no significant difference"


def _build_group_results(
    grouped_pairs: Dict[Tuple[str, str], List[Tuple[FoldMetrics, FoldMetrics]]],
    metric: str,
    alternative: str,
) -> List[GroupResult]:
    results: List[GroupResult] = []
    for (dataset, base_model), pairs in sorted(grouped_pairs.items()):
        base_vals, plus_vals = _metric_values(pairs, metric)
        if not base_vals:
            continue

        diffs = np.array(plus_vals, dtype=float) - np.array(base_vals, dtype=float)
        t_stat, p_value = _paired_ttest(base_vals, plus_vals, alternative)
        results.append(
            GroupResult(
                dataset=dataset,
                base_model=base_model,
                plus_model=f"{base_model}+",
                metric=metric,
                n_folds=len(base_vals),
                wins=sum(1 for base_val, plus_val in zip(base_vals, plus_vals) if plus_val > base_val),
                base_mean=float(np.mean(base_vals)),
                base_std=_safe_std(base_vals),
                plus_mean=float(np.mean(plus_vals)),
                plus_std=_safe_std(plus_vals),
                diff_mean=float(np.mean(diffs)),
                diff_std=_safe_std(diffs.tolist()),
                t_stat=t_stat,
                p_value=p_value,
            )
        )
    return results


def _apply_holm_correction(results: List[GroupResult]) -> None:
    valid_indices = [idx for idx, result in enumerate(results) if np.isfinite(result.p_value)]
    adjusted = _holm_bonferroni([results[idx].p_value for idx in valid_indices])
    for idx, p_holm in zip(valid_indices, adjusted):
        results[idx].p_holm = p_holm


def _build_result_rows(
    results: Sequence[GroupResult],
    alpha: float,
    alternative: str,
) -> List[List[str]]:
    rows = [[
        "dataset",
        "pair",
        "metric",
        "n",
        "base",
        "plus",
        "delta",
        "wins",
        "t",
        "p",
        "p_holm",
        "conclusion",
    ]]
    for result in results:
        rows.append([
            result.dataset,
            f"{result.base_model} vs {result.plus_model}",
            result.metric.upper(),
            str(result.n_folds),
            _format_mean_pm_std(result.base_mean, result.base_std),
            _format_mean_pm_std(result.plus_mean, result.plus_std),
            _format_mean_pm_std(result.diff_mean, result.diff_std, signed=True),
            f"{result.wins}/{result.n_folds}",
            _format_float(result.t_stat),
            _format_p_value(result.p_value),
            _format_p_value(result.p_holm),
            _conclusion(result.diff_mean, result.p_holm, alpha, alternative),
        ])
    return rows


def _parse_csv_set(spec: str | None) -> set[str] | None:
    if spec is None:
        return None
    items = {token.strip() for token in spec.split(",") if token.strip()}
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired t-test table for base models vs corresponding + models"
    )
    parser.add_argument("--outputs_dir", type=str, default="../outputs")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument(
        "--metric",
        type=str,
        default="auprc",
        choices=["auprc", "auroc", "minrp", "accuracy"],
        help="Metric to report in the final paired table.",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="greater",
        choices=["greater", "less", "two-sided"],
        help="Alternative hypothesis for paired t-test. Default tests whether + > base.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise SystemExit(f"outputs_dir not found: {outputs_dir}")

    dataset_filter = _parse_csv_set(args.datasets)
    model_filter = _parse_csv_set(args.models)

    records = _load_records(outputs_dir)
    if not records:
        raise SystemExit(f"No final test metrics found under: {outputs_dir}")

    grouped = _discover_pairs(records, datasets=dataset_filter, models=model_filter)
    grouped = {key: value for key, value in grouped.items() if value}
    if not grouped:
        raise SystemExit("No matched base/+ fold pairs found.")

    results = _build_group_results(grouped, args.metric, args.alternative)
    if not results:
        raise SystemExit(f"No valid paired values found for metric: {args.metric}")

    _apply_holm_correction(results)
    _print_table(_build_result_rows(results, args.alpha, args.alternative))


if __name__ == "__main__":
    main()
