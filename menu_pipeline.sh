#!/usr/bin/env bash

###############################################################################
# Local Quantization - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run the local quantization pipelines:
#   - quantization
#   - export
#   - benchmarking
#   - full pipeline
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
  echo " 1) Quantize model"
  echo " 2) Export quantized artifacts"
  echo " 3) Benchmark quantized model"
  echo " 4) Run full pipeline (quantize + export + benchmark)"
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
