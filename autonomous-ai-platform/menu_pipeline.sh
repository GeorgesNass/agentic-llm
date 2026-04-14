#!/usr/bin/env bash

###############################################################################
# Autonomous-AI-Platform - Pipeline Menu
# Author: Georges Nassopoulos
# Version: 1.0.0
# Description:
#   CLI menu to run autonomous-ai-platform workflows (with data consistency):
#   - ingest documents (folder scan -> chunk -> embeddings -> vector store)
#   - run chat (auto/local/api) with optional RAG + Text-to-SQL
#   - run autonomous loop (agentic planning/execution/self-correction)
#   - run evaluation (offline metrics + optional LLM judge)
#   - run monitoring stack helpers (Prometheus metrics endpoint + traces export)
#   - run MCP server (FastAPI)
#   - run Streamlit UI
###############################################################################

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo " Autonomous-AI-Platform - Pipeline Menu"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

pause() {
  read -rp "Press ENTER to continue..."
}

run_python() {
  echo ""
  echo ">>> $*"
  $PYTHON_BIN "$@"
}

run_streamlit() {
  echo ""
  echo ">>> streamlit run streamlit_app.py"
  streamlit run streamlit_app.py
}

while true; do
  echo ""
  echo "Select an action:"
  echo " 1) Ingest docs -> Vector Store (RAG build) (with data consistency)"
  echo " 2) Chat (auto/local/api) + optional RAG + optional Text-to-SQL (with data consistency)"
  echo " 3) Autonomous loop (agentic) + optional exports (with data consistency)"
  echo " 4) Evaluate (offline metrics + optional LLM judge) (with data consistency)"
  echo " 5) Run MCP server (FastAPI) (with data consistency)"
  echo " 6) Run Streamlit UI (with data consistency)"
  echo " 0) Exit"
  echo ""

  read -rp "Your choice: " choice

  case "$choice" in
    1)
      read -rp "Folder to ingest [default: ./data/raw]: " FOLDER
      read -rp "Prefer local embeddings? (y/n) [default: y]: " PREFER_LOCAL
      read -rp "Use GPU? (y/n) [default: n]: " USE_GPU
      read -rp "Overwrite existing index? (y/n) [default: n]: " OVERWRITE

      FOLDER="${FOLDER:-./data/raw}"
      PREFER_LOCAL="${PREFER_LOCAL:-y}"
      USE_GPU="${USE_GPU:-n}"
      OVERWRITE="${OVERWRITE:-n}"

      CMD=(main.py --ingest --folder "$FOLDER")

      if [[ "$PREFER_LOCAL" == "y" || "$PREFER_LOCAL" == "Y" ]]; then
        CMD+=(--prefer-local)
      else
        CMD+=(--prefer-api)
      fi

      if [[ "$USE_GPU" == "y" || "$USE_GPU" == "Y" ]]; then
        CMD+=(--use-gpu)
      fi

      if [[ "$OVERWRITE" == "y" || "$OVERWRITE" == "Y" ]]; then
        CMD+=(--overwrite)
      fi

      run_python "${CMD[@]}"
      pause
      ;;
    2)
      read -rp "Your prompt: " PROMPT
      read -rp "Enable RAG? (y/n) [default: y]: " ENABLE_RAG
      read -rp "Enable Text-to-SQL? (y/n) [default: n]: " ENABLE_SQL
      read -rp "Prefer local LLM? (y/n) [default: n]: " PREFER_LOCAL
      read -rp "Use GPU? (y/n) [default: n]: " USE_GPU
      read -rp "Top-K (RAG) [default: 5]: " TOPK
      read -rp "Max tokens [default: 512]: " MAXTOK

      ENABLE_RAG="${ENABLE_RAG:-y}"
      ENABLE_SQL="${ENABLE_SQL:-n}"
      PREFER_LOCAL="${PREFER_LOCAL:-n}"
      USE_GPU="${USE_GPU:-n}"
      TOPK="${TOPK:-5}"
      MAXTOK="${MAXTOK:-512}"

      CMD=(main.py --chat --prompt "$PROMPT" --top-k "$TOPK" --max-tokens "$MAXTOK")

      if [[ "$ENABLE_RAG" == "y" || "$ENABLE_RAG" == "Y" ]]; then
        CMD+=(--rag)
      fi

      if [[ "$ENABLE_SQL" == "y" || "$ENABLE_SQL" == "Y" ]]; then
        CMD+=(--text-to-sql)
      fi

      if [[ "$PREFER_LOCAL" == "y" || "$PREFER_LOCAL" == "Y" ]]; then
        CMD+=(--prefer-local)
      else
        CMD+=(--prefer-api)
      fi

      if [[ "$USE_GPU" == "y" || "$USE_GPU" == "Y" ]]; then
        CMD+=(--use-gpu)
      fi

      run_python "${CMD[@]}"
      pause
      ;;
    3)
      read -rp "Your objective: " OBJECTIVE
      read -rp "Max steps [default: 5]: " MAXSTEPS
      read -rp "Enable RAG tools? (y/n) [default: y]: " ENABLE_RAG
      read -rp "Enable SQL tools? (y/n) [default: y]: " ENABLE_SQL
      read -rp "Prefer local LLM? (y/n) [default: n]: " PREFER_LOCAL
      read -rp "Use GPU? (y/n) [default: n]: " USE_GPU
      read -rp "Export JSON report? (y/n) [default: y]: " EXPORT

      MAXSTEPS="${MAXSTEPS:-5}"
      ENABLE_RAG="${ENABLE_RAG:-y}"
      ENABLE_SQL="${ENABLE_SQL:-y}"
      PREFER_LOCAL="${PREFER_LOCAL:-n}"
      USE_GPU="${USE_GPU:-n}"
      EXPORT="${EXPORT:-y}"

      CMD=(main.py --loop --prompt "$OBJECTIVE" --max-steps "$MAXSTEPS")

      if [[ "$ENABLE_RAG" == "y" || "$ENABLE_RAG" == "Y" ]]; then
        CMD+=(--enable-rag-tools)
      fi

      if [[ "$ENABLE_SQL" == "y" || "$ENABLE_SQL" == "Y" ]]; then
        CMD+=(--enable-sql-tools)
      fi

      if [[ "$PREFER_LOCAL" == "y" || "$PREFER_LOCAL" == "Y" ]]; then
        CMD+=(--prefer-local)
      else
        CMD+=(--prefer-api)
      fi

      if [[ "$USE_GPU" == "y" || "$USE_GPU" == "Y" ]]; then
        CMD+=(--use-gpu)
      fi

      if [[ "$EXPORT" == "y" || "$EXPORT" == "Y" ]]; then
        CMD+=(--export)
      fi

      run_python "${CMD[@]}"
      pause
      ;;
    4)
      read -rp "Evaluation input JSON path [default: ./artifacts/evaluations/eval_input.json]: " EVAL_IN
      read -rp "Enable LLM judge? (y/n) [default: n]: " LLMJ
      read -rp "Prefer local judge? (y/n) [default: n]: " PREFER_LOCAL
      read -rp "Use GPU? (y/n) [default: n]: " USE_GPU

      EVAL_IN="${EVAL_IN:-./artifacts/evaluations/eval_input.json}"
      LLMJ="${LLMJ:-n}"
      PREFER_LOCAL="${PREFER_LOCAL:-n}"
      USE_GPU="${USE_GPU:-n}"

      CMD=(main.py --evaluate --eval-input "$EVAL_IN")

      if [[ "$LLMJ" == "y" || "$LLMJ" == "Y" ]]; then
        CMD+=(--llm-judge)
      fi

      if [[ "$PREFER_LOCAL" == "y" || "$PREFER_LOCAL" == "Y" ]]; then
        CMD+=(--prefer-local)
      else
        CMD+=(--prefer-api)
      fi

      if [[ "$USE_GPU" == "y" || "$USE_GPU" == "Y" ]]; then
        CMD+=(--use-gpu)
      fi

      run_python "${CMD[@]}"
      pause
      ;;
    5)
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
    6)
      run_streamlit
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