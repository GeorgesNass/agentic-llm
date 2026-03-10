# 🤖 Autonomous AI Platform

This project implements a complete **agentic LLM platform** for local and API-based inference.

## 🎯 Project Overview

The platform supports:

* Run an agentic loop (plan → execute tools → self-check → finalize)
* Support **local quantized models (GGUF)** via llama-cpp-python (CPU or CUDA GPU)
* Support **GPU server inference** via **vLLM** (OpenAI-compatible)
* Support external LLM APIs: **ChatGPT (OpenAI), Grok (xAI), Gemini (Google)**
* Provide **RAG ingestion + retrieval** on local corpora
* Provide **Text-to-SQL** on a local **SQLite** database
* Orchestrate the full stack with **Airflow**
* Evaluate outputs (LLM-as-a-judge + metrics)
* Export telemetry with **Prometheus** and visualize via **Grafana**

The system is designed to be **local-first, modular, and production-ready**.

---

## ⚙️ Tech Stack

Core technologies used in the platform:

* Python
* Docker & Docker Compose
* Airflow
* Streamlit
* llama.cpp / GGUF models
* vLLM
* OpenAI / xAI / Google APIs
* FAISS / Chroma vector stores
* SQLite
* Prometheus
* Grafana

---

## 📂 Project Structure

```
autonomous-ai-platform/
├── main.py                              ## Streamlit UI (chat + control center)
├── menu_pipeline.sh                     ## CLI launcher (ingest / run / eval / airflow / monitor)
├── requirements.txt                     ## Python dependencies
├── README.md                            ## Full architecture documentation
├── .env                                 ## Environment configuration (models, GPU, API keys, DB, etc.)
├── .gitignore
├── .dockerignore
│
├── docker/                              
│   ├── Dockerfile                       ## Main app container
│   └── docker-compose.yml               ## Orchestrator + Airflow + DB + Prometheus + Grafana + Vector DB
│
├── airflow/                             
│   ├── dags/
│   │   ├── ingestion_dag.py             ## Drive → OCR → Embeddings → Vector DB
│   │   ├── evaluation_dag.py            ## Automatic RAG evaluation
│   │   └── monitoring_dag.py            ## Periodic health checks
│   └── airflow.cfg
│
├── monitoring/
│   ├── prometheus.yml                   ## Metrics scraping config
│   └── grafana/                         ## Dashboards JSON
│
├── logs/                                ## Centralized logs
│
├── data/
│   ├── raw/                             ## Local ingestion temp
│   ├── processed/                       ## Clean text chunks
│   └── sqlite/                          ## Local SQLite DB (Text-to-SQL)
│
├── artifacts/
│   ├── models/                          ## Local quantized models (GGUF, etc.)
│   ├── embeddings/                      ## Saved embedding artifacts
│   ├── vector_store/                    ## FAISS / Chroma persistence
│   ├── reports/                         ## Evaluation reports
│   └── evaluations/                     ## Evaluation outputs
│
├── tests/
│   └──test_unit.py                      ## Unit tests for agents/rag/sql/orchestrator (no real HTTP calls)                 
│
└── src/
	├── __init__.py                      
	├── pipeline.py                      ## End-to-end orchestration (manual mode)
	│
	├── core/
	│   ├── __init__.py                  ## Core package marker
	│   ├── mcp_server.py                ## MCP server
	│   ├── streamlit_app.py             ## Streamlit chatbot-like app
	│   ├── schema.py                    ## Pydantic request/response models (API contract)
	│   ├── config.py                    ## Settings loader (env parsing, paths, environment=dev/prod)
	│   └── errors.py                    ## Custom exceptions + helpers (log_and_raise_*)
	│
	├── llm/
	│   ├── __init__.py                  	
	│   ├── local_runtime.py             ## GGUF / quantized local models
	│   ├── api_clients.py               ## ChatGPT / Grok / others
	│   └── embeddings.py                ## Local or API embeddings
	│	
	├── agents/                          
	│   ├── __init__.py                  
	│   ├── reasoning.py                 ## Task planning & Self-evaluation agent
	│   ├── executor.py                  ## Tool execution agent
	│   ├── text_to_sql.py               ## LLM-driven SQL generation	
	│   └── aggregator.py                ## Final answer synthesis
	│
	├── orchestrator/
	│   ├── __init__.py                  	
	│   ├── routing.py                   ## Tool routing logic & GPU/CPU/API model switching
	│   ├── retrieval.py                 ## chunking + ingestion + rag_search
	│   ├── vector_store.py              ## FAISS + Chroma + factory	
	│   ├── tools.py                     ## rag_search, sql_query, etc.	
	│   └── loop.py                      ## Self-correction loop	              	
	│
	├── monitoring/
	│   ├── __init__.py          
	│   ├── evaluation.py                ## LLM-as-a-judge & Scoring + metrics	
	│   ├── metrics.py                   ## Prometheus exporters
	│   └── tracing.py                   ## Execution traces
	│
	└── utils/
		├── __init__.py
		├── logging_utils.py      		 ## Logging + execution decorator
		├── sqlite_manager.py     	     ## SQLite init & queries
		├── env_utils.py                 ## .env readers & config resolvers
		├── validation_utils.py          ## Input validation & coercion
		├── safe_utils.py                ## Safe JSON/string helpers
		├── request_utils.py             ## HTTP headers & response parsing
		├── text_utils.py                ## Text normalization & detection
		├── io_utils.py                  ## Filesystem & file discovery
		└── llm_utils.py                 ## Chunk + embeddings helpers
```
---

## ❓ Problem Statement

Modern LLM systems introduce multiple operational challenges:

* Multiple inference providers (local models, API providers)
* Hardware constraints (CPU vs GPU)
* Tool orchestration (RAG, SQL, custom tools)
* Difficult evaluation and observability
* Local deployment complexity

Key problems addressed:

* Runtime routing between **local models, GPU inference, and APIs**
* Reliable **RAG ingestion and retrieval**
* Safe **Text-to-SQL execution**
* Reproducible pipelines with **Airflow**
* Structured error handling
* Monitoring with **Prometheus and Grafana**

---
## 🧠 Approach / Methodology / Strategy

The platform follows an **agentic LLM architecture** combining tool orchestration, automated evaluation, and monitoring to produce reliable answers.

Core principles:

* **Tool-augmented reasoning** using RAG retrieval and Text-to-SQL execution
* **Multi-backend inference routing** (local models, GPU servers, external APIs)
* **Evaluation-driven reliability** using LLM-as-a-judge scoring and metrics
* **Operational monitoring** with Prometheus telemetry and Grafana dashboards

### Tool Ecosystem

| Component         | Role                                                     |
| ----------------- | -------------------------------------------------------- |
| RAG Retrieval     | Document ingestion, chunking, embeddings, vector search  |
| Text-to-SQL       | Natural language → SQL queries executed safely on SQLite |
| Local LLM Runtime | Quantized GGUF models via llama-cpp                      |
| API LLM Providers | OpenAI (ChatGPT), xAI (Grok), Google (Gemini)            |
| GPU Inference     | High-throughput inference via vLLM                       |
| Evaluation        | LLM-as-a-judge scoring and response validation           |
| Monitoring        | Prometheus metrics and Grafana dashboards                |

---

## 🏗 Pipeline Architecture

```
User Query
    ↓
Routing (local / API / GPU)
    ↓
Reasoning Agent
    ↓
Executor (RAG / SQL tools)
    ↓
Self-Correction Loop
    ↓
Aggregator
    ↓
Evaluation
    ↓
Prometheus Metrics
    ↓
Grafana Dashboards
```

---

## 📊 Exploratory Data Analysis

When using RAG pipelines, the platform performs preprocessing steps such as:

* document ingestion
* text normalization
* chunk distribution analysis
* embedding generation
* vector index inspection

Processed artifacts are stored in:

```
data/processed/
artifacts/vector_store/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python **3.11+**
* Docker & Docker Compose (recommended)
* Optional GPU with **CUDA**

### 2. OS prerequists

Verify that you have the necessairy packages installed.

#### Windows / WSL2 (recommended)

```bash
# PowerShell
wsl --status
wsl --install
wsl --list --online
wsl --install -d Ubuntu
wsl -d Ubuntu

docker --version
docker compose version
```

#### Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential curl git
python3 --version
```

### 3. Python environment

```bash
python -m venv .llm_platf
source .llm_platf/bin/activate          ## Windows: .llm_platf\Scripts\activate.bat
pip install --upgrade pip               ## for windows : .llm_platf\Scripts\python.exe -m pip install --upgrade pip 
pip install -r requirements.txt
python -m pip check
```

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ▶️ Usage & End-to-End Testing

```bash
## Check raw input data folders
ls data/raw

## Run ingestion pipeline (documents → chunks → embeddings → vector store)
python main.py --ingest

## Run a single chat interaction through the CLI (no UI)
python main.py --chat "hello"

## Run the autonomous agent loop for a complex task
python main.py --loop "complex task" --export

## Run evaluation for a query/answer pair
python main.py --evaluate --query "..." --answer "..."

## Run evaluation (LLM-as-a-judge)
## NOTE: current CLI flag is --evaluate (there is NO --eval in main.py)
python main.py --evaluate --query "..." --answer "..." --export

## Test CLI parser: prefer API routing
python main.py --chat "hello" --prefer-api

## Test CLI parser: prefer local routing (explicit)
python main.py --chat "hello" --prefer-local

## Test CLI parser: GPU flag auto/true/false (parsing + propagation)
python main.py --chat "hello" --use-gpu auto
python main.py --chat "hello" --use-gpu true
python main.py --chat "hello" --use-gpu false

## Test CLI parser: export artifacts on loop/eval
python main.py --loop "complex task" --export
python main.py --evaluate --query "..." --answer "..." --export

## Run full pipeline
python main.py --run-all

## Run API server
python main.py --run-api

## Run Streamlit UI
streamlit run main.py

## Test API health endpoint
curl http://localhost:8000/health

## Test metrics endpoint (Prometheus)
curl http://localhost:8000/metrics

## Inspect evaluation artifacts
ls artifacts/evaluations
ls artifacts/reports

## Run Airflow pipeline (if docker compose configured)
docker compose up -d airflow

## Inspect Airflow logs
docker compose logs -f airflow

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                | Cause                    | Solution                              |
| -------------------- | ------------------------ | ------------------------------------- |
| ModuleNotFoundError  | Missing dependencies     | Run `pip install -r requirements.txt` |
| Docker build failure | Incorrect Docker version | Update Docker                         |
| API 401 error        | Missing API key          | Check `.env` configuration            |
| CUDA not detected    | GPU drivers missing      | Install CUDA drivers                  |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Research / Professional NLP project
