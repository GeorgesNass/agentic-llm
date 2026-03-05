# 🤖 Autonomous AI Platform

## 1. Project Overview

This project implements a complete **agentic LLM platform** for local and API-based inference.

The objective is to:

- Run an agentic loop (plan → execute tools → self-check → finalize)
- Support **local quantized models (GGUF)** via llama-cpp-python (CPU or CUDA GPU)
- Support **GPU server inference** via **vLLM** (OpenAI-compatible)
- Support external LLM APIs: **ChatGPT (OpenAI), Grok (xAI), Gemini (Google)**
- Provide **RAG ingestion + retrieval** on local corpora
- Provide **Text-to-SQL** on a local **SQLite** database
- Orchestrate the full stack with **Airflow**
- Evaluate outputs (LLM-as-a-judge + metrics)
- Export telemetry with **Prometheus** and visualize via **Grafana**

The platform is designed to be **local-first**, Docker-friendly, and production-clean.

---

## 2. Problem Statement

Modern LLM systems are:

- Multi-provider (local, OpenAI, xAI, Google)
- Cost-sensitive and hardware-dependent
- Tool-augmented (RAG, SQL, web, internal tools)
- Hard to evaluate reliably
- Hard to operate locally with observability

Challenges:

- Switching between CPU / GPU / API at runtime
- Local model provisioning (HuggingFace download, GGUF files)
- RAG ingestion and persistence (FAISS / Chroma)
- Reproducible orchestration (Airflow)
- Structured errors (no raw stack traces)
- Monitoring (Prometheus metrics + Grafana dashboards)

This project addresses these constraints through:

- Backend routing (local llama.cpp / vLLM / API)
- HuggingFace model download support
- Unified RAG ingestion + retrieval layer
- Text-to-SQL execution with safe SQLite management
- Airflow DAGs for ingestion, evaluation, monitoring
- Structured exceptions + standardized API responses
- Prometheus exporters and local Grafana dashboards

---

## 3. Agentic Strategy

### Objective

Produce reliable answers by combining:

- A planning step
- Tool execution (RAG search, SQL queries)
- Self-evaluation (critic)
- Final synthesis

### Supported Tools

- RAG retrieval (chunking → embeddings → vector search)
- SQL query execution (Text-to-SQL → SQLite)
- Optional evaluation scoring (LLM-as-a-judge)

### Execution Flow

- Reasoning agent builds a step plan
- Executor runs tools safely
- Critic validates results and can request another loop
- Aggregator composes the final response

---

## 4. Pipeline Architecture

```
User Query
    ↓
Routing (auto/local/api + CPU/GPU)
    ↓
Reasoning Agent (plan)
    ↓
Executor (tools: RAG, SQL)
    ↓
Optional Self-correction Loop
    ↓
Aggregator (final answer)
    ↓
Evaluation (optional)
    ↓
Prometheus metrics + Grafana dashboards
```

---

## 5. Evaluation & Monitoring

The monitoring layer provides:

- Request counts, latency, failures
- Tool-level metrics (RAG/SQL)
- LLM provider usage
- Evaluation scoring exports

Outputs are exported in:

```
artifacts/reports/
artifacts/evaluations/
```

Prometheus config and Grafana dashboards are stored in:

```
monitoring/prometheus.yml
monitoring/grafana/
```

---

## 6. Project Structure

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

## 7. Prerequisites

- Python 3.11+
- Docker & Docker Compose (recommended)
- GPU optional (CUDA required for vLLM / llama.cpp GPU offload)

### Ubuntu Example

```bash
sudo apt update
sudo apt install python3 python3-pip
python3 --version
```

---

## 8. Setup

### Python

```bash
python -m venv .llm_platf
source .llm_platf/bin/activate                              ## Windows: .llm_platf\Scripts\activate.bat
pip install --upgrade pip                     				## for windows : .llm_platf\Scripts\python.exe -m pip install --upgrade pip 
pip install -r requirements.txt
python -m pip check
```

### Docker

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ✅ Full System Verification (End-to-End)

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

## Authors

**Georges Nassopoulos**  
Email: georges.nassopoulos@gmail.com
