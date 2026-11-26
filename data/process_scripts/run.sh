#!/bin/bash
# Build data/processed/*.pkl and data/splits/*_5fold.json.
#
# Raw data directories can be overridden with environment variables:
#   P12_RAW_DATA, P19_RAW_DATA, MIMIC_III_RAW_DATA
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" "$SCRIPT_DIR/preprocess_physionet_2012.py"
"$PYTHON_BIN" "$SCRIPT_DIR/preprocess_physionet_2019.py"

# MIMIC-III requires PhysioNet credentialing. Uncomment after placing the CSVs.
# "$PYTHON_BIN" "$SCRIPT_DIR/preprocess_mimic_iii.py"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_splits.py" --val_ratio 0.2 --dataset physionet_2012
"$PYTHON_BIN" "$SCRIPT_DIR/generate_splits.py" --val_ratio 0.2 --dataset physionet_2019
# "$PYTHON_BIN" "$SCRIPT_DIR/generate_splits.py" --val_ratio 0.2 --dataset mimic_iii
