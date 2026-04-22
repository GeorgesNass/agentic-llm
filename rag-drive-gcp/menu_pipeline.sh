#!/usr/bin/env bash

###############################################################################
# RAG-Drive-GCP - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.1.0
# Description:
#   CLI menu to run the main rag-drive-gcp workflows:
#   - validate configuration (with data consistency + data quality)
#   - run ingestion pipeline from Google Drive (with optional OCR) (with data consistency + data quality)
#   - run a RAG query from CLI (with data consistency + data quality)
#   - launch Streamlit UI (separate file)
#   - run data drift (Evidently)
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " RAG-Drive-GCP - Pipeline Menu (data consistency + data quality)"
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
  echo " 1) Validate config (with data consistency + data quality)"
  echo " 2) Dry-run"
  echo " 3) Run ingestion (Drive → OCR → GCS → embeddings) (with data consistency + data quality)"
  echo " 4) Run RAG query (CLI) (with data consistency + data quality)"
  echo " 5) Launch Streamlit UI"
  echo " 6) Show help"
  echo " 7) Show version"
  echo " 8) Run data quality check only"
  echo " 9) Run data drift "
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      run_python main.py --validate-config
      pause
      ;;
    2)
      run_python main.py --dry-run
      pause
      ;;
    3)
      read -rp "Drive folder ID [default: from config]: " FOLDER_ID
      read -rp "Run OCR? (y/n) [default: y]: " RUN_OCR
      read -rp "Keep local files? (y/n) [default: n]: " KEEP_LOCAL
      read -rp "Chunk size [default: from config]: " CHUNK_SIZE
      read -rp "Chunk overlap [default: from config]: " CHUNK_OVERLAP

      RUN_OCR="${RUN_OCR:-y}"
      KEEP_LOCAL="${KEEP_LOCAL:-n}"

      CMD="main.py --ingest"

      if [[ -n "$FOLDER_ID" ]]; then
        CMD="$CMD --folder-id \"$FOLDER_ID\""
      fi

      if [[ "$RUN_OCR" == "y" || "$RUN_OCR" == "Y" ]]; then
        CMD="$CMD --run-ocr"
      fi

      if [[ "$KEEP_LOCAL" == "y" || "$KEEP_LOCAL" == "Y" ]]; then
        CMD="$CMD --keep-local"
      fi

      if [[ -n "$CHUNK_SIZE" ]]; then
        CMD="$CMD --chunk-size \"$CHUNK_SIZE\""
      fi

      if [[ -n "$CHUNK_OVERLAP" ]]; then
        CMD="$CMD --chunk-overlap \"$CHUNK_OVERLAP\""
      fi

      eval run_python $CMD
      pause
      ;;
    4)
      read -rp "Question: " QUESTION
      read -rp "Top-K [default: from config]: " TOP_K

      CMD="main.py --query --question \"$QUESTION\""

      if [[ -n "$TOP_K" ]]; then
        CMD="$CMD --top-k \"$TOP_K\""
      fi

      eval run_python $CMD
      pause
      ;;
    5)
      read -rp "Path to Streamlit file [default: app.py]: " UI_FILE
      UI_FILE="${UI_FILE:-app.py}"

      echo ""
      echo ">>> streamlit run $UI_FILE"
      streamlit run "$UI_FILE"

      pause
      ;;
    6)
      run_python main.py --help
      pause
      ;;
    7)
      run_python main.py --version
      pause
      ;;
    8)
      echo ""
      echo "Running data quality check..."
      echo ""

      run_python main.py --validate-config

      pause
      ;;
    9)
      ## DATA DRIFT (RAG + EVIDENTLY)
      read -rp "Reference dataset CSV: " REF
      read -rp "Current dataset CSV: " CUR

      run_python main.py \
        --mode drift \
        --ref "$REF" \
        --current "$CUR"

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