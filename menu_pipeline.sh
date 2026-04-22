#!/usr/bin/env bash

###############################################################################
# Local Quantization - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the local quantization pipelines (with data consistency + data quality + data drift):
#   - quantization
#   - export
#   - benchmarking
#   - full pipeline
#   - data drift
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Local Quantization - Pipeline Menu"
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

export_pipeline_mode() {
  export PIPELINE_MODE="$1"
  echo "PIPELINE_MODE=${PIPELINE_MODE}"
}

## ---------------------------------------------------------------------------
## Menu
## ---------------------------------------------------------------------------

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Quantize model (with data consistency + data quality)"
  echo " 2) Export quantized artifacts (with data consistency + data quality)"
  echo " 3) Benchmark quantized model (with data consistency + data quality)"
  echo " 4) Run full pipeline (quantize + export + benchmark) (with data consistency + data quality)"
  echo " 5) Run data drift"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      export_pipeline_mode "quantize"
      run_python main.py
      pause
      ;;
    2)
      export_pipeline_mode "export"
      run_python main.py
      pause
      ;;
    3)
      export_pipeline_mode "benchmark"
      run_python main.py
      pause
      ;;
    4)
      export_pipeline_mode "full"
      run_python main.py
      pause
      ;;
    5)
      ## DATA DRIFT
      read -rp "Reference weights path [default: ./artifacts/ref_weights.npy]: " REF_W
      REF_W="${REF_W:-./artifacts/ref_weights.npy}"

      read -rp "Current weights path [default: ./artifacts/quant_weights.npy]: " CUR_W
      CUR_W="${CUR_W:-./artifacts/quant_weights.npy}"

      read -rp "Reference model size [MB] [default: 100]: " REF_S
      REF_S="${REF_S:-100}"

      read -rp "Current model size [MB] [default: 25]: " CUR_S
      CUR_S="${CUR_S:-25}"

      run_python main.py \
        --mode drift \
        --weights-ref "$REF_W" \
        --weights-current "$CUR_W" \
        --size-ref "$REF_S" \
        --size-current "$CUR_S"

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