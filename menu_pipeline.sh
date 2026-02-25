#!/usr/bin/env bash

###############################################################################
# LLM Proxy Gateway - CLI Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   Interactive CLI to run:
#   - Cost simulation (chat or embeddings)
#   - Direct chat completion
#   - Embeddings generation
#   - Evaluation
#   - FastAPI service
#   - Unit tests
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " LLM Proxy Gateway - CLI Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

###############################################################################
# Helpers
###############################################################################

pause() {
  read -rp "Press ENTER to continue..."
}

run_python() {
  echo ""
  echo ">>> $*"
  $PYTHON_BIN "$@"
}

###############################################################################
# Menu
###############################################################################

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Simulate cost (chat)"
  echo " 2) Simulate cost (embeddings)"
  echo " 3) Run API (uvicorn)"
  echo " 4) Run tests (pytest)"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "Provider (openai|google|xai) [default: openai]: " PROVIDER
      read -rp "Model [default: gpt-4o-mini]: " MODEL
      read -rp "Text prompt: " TEXT
      read -rp "Expected output tokens [default: 1024]: " OUTTOK

      PROVIDER="${PROVIDER:-openai}"
      MODEL="${MODEL:-gpt-4o-mini}"
      OUTTOK="${OUTTOK:-1024}"

      run_python main.py \
        --simulate-cost \
        --mode chat \
        --providers "${PROVIDER}" \
        --model "${MODEL}" \
        --text "${TEXT}" \
        --expected-output-tokens "${OUTTOK}"

      pause
      ;;
    2)
      read -rp "Provider (openai|google|xai) [default: openai]: " PROVIDER
      read -rp "Model [default: text-embedding-3-small]: " MODEL
      read -rp "Text (leave empty to use folder scan): " TEXT
      read -rp "Folder path (if no text) [default: ./data/raw]: " FOLDER

      PROVIDER="${PROVIDER:-openai}"
      MODEL="${MODEL:-text-embedding-3-small}"
      FOLDER="${FOLDER:-./data/raw}"

      if [[ -n "${TEXT}" ]]; then
        run_python main.py \
          --simulate-cost \
          --mode embeddings \
          --providers "${PROVIDER}" \
          --model "${MODEL}" \
          --text "${TEXT}"
      else
        run_python main.py \
          --simulate-cost \
          --mode embeddings \
          --providers "${PROVIDER}" \
          --model "${MODEL}" \
          --path "${FOLDER}" \
          --recursive
      fi

      pause
      ;;
    3)
      read -rp "Host [default: 0.0.0.0]: " HOST
      read -rp "Port [default: 8000]: " PORT
      read -rp "Reload? (y/n) [default: n]: " RELOAD

      HOST="${HOST:-0.0.0.0}"
      PORT="${PORT:-8000}"
      RELOAD="${RELOAD:-n}"

      if [[ "$RELOAD" == "y" || "$RELOAD" == "Y" ]]; then
        run_python main.py --run-api --host "$HOST" --port "$PORT" --reload
      else
        run_python main.py --run-api --host "$HOST" --port "$PORT"
      fi

      pause
      ;;
    4)
      echo ""
      echo ">>> Running pytest"
      cd "${PROJECT_ROOT}"
      $PYTHON_BIN -m pytest -q
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