# 🚀 LLM Proxy Gateway – Multi-Provider LLM Orchestration Platform

The gateway provides a **unified interface to orchestrate multiple LLM providers**, enabling cost simulation, model evaluation, and provider-agnostic execution.

---

## 🎯 Project Overview

Main capabilities:

* Unified **chat completion interface**
* Unified **embeddings interface**
* **Pre-execution cost simulation**
* **Text evaluation metrics** for model outputs
* **Multi-provider routing** (OpenAI, Gemini, xAI)
* **CLI + FastAPI API interface**
* Clean modular architecture and reproducible execution

The gateway abstracts provider differences and exposes a **stable orchestration layer for LLM applications**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* FastAPI
* Docker & Docker Compose
* Pydantic
* REST APIs (OpenAI / Gemini / xAI)
* Token estimation utilities
* JSON provider & pricing catalogs

---

## 📂 Project Structure

```
llm-proxy-gateway/
├── main.py                            ## CLI entry point (cost, run, evaluate, run-api) + uvicorn bootstrap
├── menu_pipeline.sh                   ## Interactive CLI menu to run cost/pipeline/eval or API service
├── requirements.txt                   ## Python dependencies
├── README.md                          ## Project documentation
├── .env                               ## Environment configuration (API keys, base urls, environment)
├── .gitignore                         ## Git ignored files
├── .dockerignore                      ## Docker build exclusions
│
├── docker/                            ## Container configuration and service orchestration
│   ├── Dockerfile                     ## Application container definition
│   └── docker-compose.yml             ## Local orchestration (API + volumes + environment)
│
├── logs/                              ## Centralized runtime logs (application.log, etc.)
│
├── secrets/                           ## Service account credentials (excluded from version control)
│
├── data/
│   ├── raw/                           ## Raw .txt files for folder scan / evaluation corpora
│   └── processed/                     ## Optional CSV exports (per-file scan, results, etc.)
│
├── artifacts/
│   ├── resources/
│   │   ├── models_catalog.json        ## Provider mapping + model names + defaults + context limits
│   │   └── pricing_catalog.json       ## Pricing per provider/model ($/1K input, output, embeddings)
│   │
│   ├── config/
│   │   └── swagger.yaml               ## OpenAPI spec (optional override / stable contract)
│   │
│   └── exports/                       ## Optional exports (CSV outputs, cost reports, evaluation reports)
│
├── tests/
│   └── test_unit.py                   ## Unit tests for utils/costing/evaluation (no real HTTP calls)
│
└── src/
    ├── pipeline.py                    ## Orchestration: cost → (embeddings | chat) → optional eval/export
    │
    ├── core/
    │   ├── auth.py                    ## JWT auth: tokens, login, refresh, dependencies
    │   ├── security.py                ## RBAC, middleware, permissions, request security		
    │   ├── service.py                 ## FastAPI app factory + routes (/healthcheck, /cost, /chat, /embeddings, /evaluation)
    │   ├── schema.py                  ## Pydantic request/response models (API contract)
    │   ├── config.py                  ## Settings loader (env parsing, paths, environment=dev/prod)
    │   └── errors.py                  ## Custom exceptions + helpers (log_and_raise_*)
    │
    ├── llm/
    │   ├── completion.py              ## Provider chat completion clients + dispatch (OpenAI/Gemini/xAI)
    │   ├── embeddings.py              ## Provider embeddings clients + dispatch (OpenAI/Gemini/xAI)
    │   ├── evaluation.py              ## Completion evaluation orchestration (calls metrics_utils)
    │   └── costing.py                 ## Cost simulation orchestration (catalog load, pricing resolution, calls scan helpers)
    │
    └── utils/
        ├── logging_utils.py           ## Centralized logging + decorator (execution time + path on error)
        ├── utils.py                   ## Generic helpers: env, paths, safe IO, CLI input helpers, scan helpers, CSV export, basic stats
        ├── http_utils.py              ## Shared HTTP helpers (headers, payload builders, safe JSON logging)
        ├── metrics_utils.py           ## Text metrics (exact match, contains, F1 token, jaccard, cosine, ROUGE/BLEU/BERTScore optional)
        ├── tokeniser_utils.py         ## Tokenization + estimation helpers (approx + future provider tokenizers)
        └── costing_utils.py           ## Pure costing helpers (pricing rows, chunking, token estimation for embeddings, cost math)
```

---

## ❓ Problem Statement

Modern LLM systems introduce several operational challenges:

* Multiple providers with **different APIs**
* Pricing differences across **models and token types**
* Token estimation inconsistencies
* Limited **cost visibility before execution**
* Lack of standardized evaluation tools
* Hardcoded provider logic in applications

This project addresses these issues through:

* Provider-agnostic dispatch abstraction
* JSON-based model and pricing catalogs
* Token estimation helpers
* Pre-execution cost simulation
* Built-in evaluation metrics
* Modular architecture supporting easy extension

---

## 🧠 Approach / Methodology / Strategy

The gateway provides a **provider-agnostic orchestration layer** combining cost estimation, provider routing, and output evaluation.

Core principles:

* **Multi-provider abstraction** for chat and embeddings
* **Pre-execution cost simulation** using token estimation
* **Evaluation-driven analysis** of LLM outputs
* **Provider-agnostic interface** for future model integration

### LLM Orchestration Ecosystem

| Component            | Role                                     |
| -------------------- | ---------------------------------------- |
| Provider Dispatch    | Route requests to OpenAI, Gemini, xAI    |
| Cost Simulation      | Estimate cost before execution           |
| Token Estimation     | Approximate token counts for prompts     |
| Pricing Catalog      | JSON mapping of models and token pricing |
| Evaluation Metrics   | Text similarity and quality metrics      |
| Embeddings Interface | Unified embedding generation             |

### Evaluation Metrics

| Metric                   | Purpose                      |
| ------------------------ | ---------------------------- |
| Exact Match              | Strict equality comparison   |
| Contains                 | Substring presence           |
| F1 Token                 | Token precision/recall       |
| Jaccard                  | Set similarity               |
| Cosine Similarity        | Vector similarity            |
| ROUGE / BLEU / BERTScore | Optional advanced evaluation |

### Cost Simulation Logic

| Step | Purpose |
|------|----------|
| Approx token estimation | Fast, offline calculation |
| Folder scan (.txt) | Batch estimation |
| Chunking | Embedding window logic |
| Pricing lookup | JSON-based provider pricing |
| Cost math | $/1K tokens calculation |


---

## 🏗 Pipeline Architecture

```
	 User CLI / HTTP Request
            ↓
        Validation Layer (Pydantic)
            ↓
        Pipeline Orchestrator
            ↓
   Cost Simulation (optional)
            ↓
  Provider Dispatch (chat / embeddings)
            ↓
   Optional Evaluation Metrics
            ↓
      Structured JSON Response
```

---

## 📊 Exploratory Data Analysis

Although this project does not process structured datasets, it includes **analysis utilities for model outputs and cost estimation**.

Examples:

* cost comparison between providers
* evaluation metrics for model responses
* token estimation diagnostics

Generated outputs can be exported in:

```
artifacts/exports/
```

---

## 🔧 Setup & Installation
In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python 3.10+
* Docker & Docker Compose (optional)
* API keys for desired LLM providers

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
python -m venv .llm_env
source .llm_env/bin/activate   							    ## for windows : .llm_env\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel		## for windows : .llm_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ▶️ Usage & End-to-End Testing

```bash
## Run API
python main.py --run-api

## Simulate chat cost
python main.py --simulate-cost --mode chat --providers openai --text "Hello"

## Simulate embeddings cost (folder)
python main.py --cost --mode embeddings --providers openai --path ./data/raw --recursive

## Run evaluation
python main.py --evaluate --predictions "Paris" --references "Paris"

## Run evaluation from file 
python main.py --evaluate --predictions-path ./data/raw/pred.txt --references-path ./data/raw/ref.txt

## Run test suite
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                      | Cause                          | Solution                   |
| -------------------------- | ------------------------------ | -------------------------- |
| API authentication failure | Missing provider API key       | Check `.env` configuration |
| Token estimation mismatch  | Approximation method used      | Adjust estimation logic    |
| Provider request failure   | Invalid provider configuration | Verify provider settings   |
| Docker container failure   | Misconfigured environment      | Rebuild container          |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** AI Engineering / LLM Infrastructure Project