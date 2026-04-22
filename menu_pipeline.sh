#!/usr/bin/env bash

###############################################################################
# Local Fine-Tuning - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the local fine-tuning pipelines (with data consistency + data quality + data drift):
#   - dataset preparation
#   - LoRA SFT training
#   - evaluation
#   - full pipeline
#   - data drift
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Local Fine-Tuning - Pipeline Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

## ---------------------------------------------------------------------------
## Helpers
## ---------------------------------------------------------------------------

pause() {
  read -rp "Press ENTER to continue..."
}

run_python() {
  echo ""
  echo ">>> $*"
  $PYTHON_BIN "$@"
}

## ---------------------------------------------------------------------------
## Menu
## ---------------------------------------------------------------------------

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Prepare dataset (with data consistency + data quality)"
  echo " 2) Train model (LoRA SFT) (with data consistency + data quality)"
  echo " 3) Evaluate model (with data consistency + data quality)"
  echo " 4) Run full pipeline (prepare + train + evaluate) (with data consistency + data quality)"
  echo " 5) Run data drift"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      run_python main.py prepare
      pause
      ;;
    2)
      run_python main.py train
      pause
      ;;
    3)
      read -rp "Path to run directory (e.g. artifacts/runs/run_YYYYMMDD_HHMMSS): " RUN_DIR
      run_python main.py evaluate --run-dir "$RUN_DIR"
      pause
      ;;
    4)
      run_python main.py full
      pause
      ;;
    5)
      ## DATA DRIFT
      read -rp "Reference dataset path [default: ./data/processed/train.jsonl]: " REF_PATH
      REF_PATH="${REF_PATH:-./data/processed/train.jsonl}"

      read -rp "Current dataset path [default: ./data/processed/new_data.jsonl]: " CUR_PATH
      CUR_PATH="${CUR_PATH:-./data/processed/new_data.jsonl}"

      run_python main.py \
        --mode drift \
        --ref "$REF_PATH" \
        --current "$CUR_PATH"

      pause
      ;;
    0)
      echo "Bye"
      exit 0
      ;;
    *)
      echo "Invalid choice."
      pause
      ;;
  esac
done