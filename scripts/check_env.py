#!/usr/bin/env python3
"""Check that the environment and required data files are ready."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ["torch", "numpy", "pandas", "sklearn", "scipy", "yaml", "optuna", "tqdm"]
DATASETS = ["physionet_2012", "physionet_2019", "mimic_iii"]


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    missing = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {name}: {version}")
        except Exception as exc:
            print(f"  {name}: MISSING ({exc})")
            missing.append(name)

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
    except Exception:
        pass

    ok = True
    for dataset in DATASETS:
        data_path = ROOT / "data" / "processed" / f"{dataset}.pkl"
        split_path = ROOT / "data" / "splits" / f"{dataset}_5fold.json"
        data_ok = data_path.exists()
        split_ok = split_path.exists()
        print(f"{dataset}: data={'OK' if data_ok else 'MISSING'} split={'OK' if split_ok else 'MISSING'}")
        if not split_ok:
            ok = False

    if missing:
        print("\nInstall dependencies with: pip install -r requirements.txt")
    if not ok:
        print("Prepare the missing datasets first; see README.md.")
    return 0 if not missing and ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
