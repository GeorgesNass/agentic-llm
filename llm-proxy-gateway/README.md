# 🚀 LLM Proxy Gateway – Multi‑Provider LLM Orchestration Platform

## 1. Project Overview

This project implements a complete **LLM proxy gateway** designed to orchestrate multiple Large Language Model providers (OpenAI, Gemini, xAI, etc.) through a unified interface.

The objective is to:

- Provide a unified chat completion interface
- Provide a unified embeddings interface
- Simulate cost before execution (chat or embeddings)
- Evaluate model outputs with text metrics
- Offer both CLI and FastAPI usage
- Ensure reproducibility, logging, and clean architecture

The gateway abstracts provider differences and exposes a stable, extensible API layer.

---

## 2. Problem Statement

Modern LLM systems face several challenges:

- Multiple providers with different APIs
- Pricing differences per model and token type
- Token estimation inconsistencies
- Lack of cost visibility before execution
- No standardized evaluation layer
- Hardcoded provider logic in applications

This project addresses these constraints through:

- Provider dispatch abstraction
- JSON-based model & pricing catalogs
- Token estimation helpers (approximate + extensible)
- Pre-execution cost simulation
- Text metric evaluation layer
- Modular architecture (core / llm / utils)
- CLI + FastAPI interface

---

## 3. LLM Strategy

### Core Functional Dimensions

| Dimension | Description | Example |
|------------|------------|----------|
| provider | LLM provider backend | openai |
| model | Model name | gpt-4o-mini |
| input_tokens | Estimated prompt tokens | 1250 |
| output_tokens | Estimated completion tokens | 800 |
| total_cost_usd | Simulated or real cost | 0.0234 |
| evaluation_metric | Text similarity metric | f1_token |
| execution_mode | chat / embeddings | chat |

### Key Operational Objectives

| Objective | Why It Matters | Example Insight |
|------------|----------------|----------------|
| Cost transparency | Prevent unexpected billing | Compare providers before execution |
| Multi-provider fallback | Avoid vendor lock-in | Switch OpenAI → Gemini |
| Embedding consistency | Unified vector generation | Same interface for all providers |
| Evaluation capability | Quantify model quality | F1 = 0.83 |
| Modular extensibility | Add providers safely | Future Claude / Mistral support |

---

## 4. Pipeline Architecture

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

## 5. Analytics & Evaluation Layer

The project provides built-in text evaluation metrics.

### Text Metrics Techniques

| Technique | Purpose | Example |
|------------|---------|----------|
| Exact Match | Strict equality | prediction == reference |
| Contains | Substring presence | "Paris" in response |
| F1 Token | Token-level precision/recall | F1 = 0.84 |
| Jaccard | Set similarity | 0.72 |
| Cosine Similarity | Vector similarity | 0.91 |
| ROUGE (optional) | Summarization quality | ROUGE-L |
| BLEU (optional) | N-gram overlap | BLEU-4 |
| BERTScore (optional) | Semantic similarity | 0.89 |

### Cost Simulation Logic

| Step | Purpose |
|------|----------|
| Approx token estimation | Fast, offline calculation |
| Folder scan (.txt) | Batch estimation |
| Chunking | Embedding window logic |
| Pricing lookup | JSON-based provider pricing |
| Cost math | $/1K tokens calculation |

---

## 6. Project Structure

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
    ├── __init__.py                    ## Package marker
    ├── pipeline.py                    ## Orchestration: cost → (embeddings | chat) → optional eval/export
    │
    ├── core/
    │   ├── __init__.py                ## Core package marker
    │   ├── service.py                 ## FastAPI app factory + routes (/healthcheck, /cost, /chat, /embeddings, /evaluation)
    │   ├── schema.py                  ## Pydantic request/response models (API contract)
    │   ├── config.py                  ## Settings loader (env parsing, paths, environment=dev/prod)
    │   └── errors.py                  ## Custom exceptions + helpers (log_and_raise_*)
    │
    ├── llm/
    │   ├── __init__.py                ## LLM package marker
    │   ├── completion.py              ## Provider chat completion clients + dispatch (OpenAI/Gemini/xAI)
    │   ├── embeddings.py              ## Provider embeddings clients + dispatch (OpenAI/Gemini/xAI)
    │   ├── evaluation.py              ## Completion evaluation orchestration (calls metrics_utils)
    │   └── costing.py                 ## Cost simulation orchestration (catalog load, pricing resolution, calls scan helpers)
    │
    └── utils/
        ├── __init__.py                ## Utils package marker
        ├── logging_utils.py           ## Centralized logging + decorator (execution time + path on error)
        ├── utils.py                   ## Generic helpers: env, paths, safe IO, CLI input helpers, scan helpers, CSV export, basic stats
        ├── http_utils.py              ## Shared HTTP helpers (headers, payload builders, safe JSON logging)
        ├── metrics_utils.py           ## Text metrics (exact match, contains, F1 token, jaccard, cosine, ROUGE/BLEU/BERTScore optional)
        ├── tokeniser_utils.py         ## Tokenization + estimation helpers (approx + future provider tokenizers)
        └── costing_utils.py           ## Pure costing helpers (pricing rows, chunking, token estimation for embeddings, cost math)
```

---

## 7. Prerequisites

- Python 3.10+
- Docker & Docker Compose
- API keys for desired LLM providers

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
python -m venv .llm_env
source .llm_env/bin/activate   							    ## for windows : .llm_env\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel		## for windows : .llm_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Docker

```bash
docker compose build
docker compose up
```

---

## 9. Full System Verification

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

## 10. Author

**Georges Nassopoulos**  
Email: georges.nassopoulos@gmail.com

