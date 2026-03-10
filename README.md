# 🛠️ Local LLM Fine-Tuning Pipeline (LoRA / QLoRA)

The project implements an **end-to-end local LLM fine-tuning pipeline** focused on **domain adaptation and label normalization** for medical symptom classification (CISP).

---

## 🎯 Project Overview

Main capabilities:

* Dataset preparation and normalization
* Supervised Fine-Tuning (SFT) using **LoRA / QLoRA**
* Deterministic inference to reduce hallucinations
* Strict evaluation metrics for label consistency
* CLI-driven pipeline execution
* Docker-based execution

The system transforms raw symptom descriptions into **domain-adapted LLM models capable of consistent medical label normalization**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* PyTorch
* HuggingFace Transformers
* LoRA / QLoRA
* Parameter-efficient fine-tuning
* Docker & Docker Compose
* GPU / CUDA acceleration (optional)

---

## 📂 Project Structure

```text
local-finetuning/
├── main.py                         ## CLI entry point
├── entrypoint.sh                   ## Container startup script
├── requirements.txt                ## Python dependencies
├── README.md                       ## Project documentation
├── .env                            ## Environment configuration
│
├── docker/
│   ├── Dockerfile                  ## Docker image definition
│   └── docker-compose.yml          ## Docker Compose configuration
│
├── data/
│   ├── raw/                        ## Raw symptom descriptions
│   ├── interim/                    ## Intermediate datasets
│   └── processed/                  ## Final training datasets
│
├── artifacts/
│   ├── runs/                       ## Training runs and checkpoints
│   └── exports/                    ## Exported adapters and metrics
│
├── tests/
│   └── test_unit.py                ## Unit tests
│
└── src/
    ├── pipeline.py                 ## Pipeline orchestration
    │
    ├── core/
    │   ├── model.py                ## Model loading utilities
    │   ├── prepare_dataset.py      ## Dataset preparation logic
    │   ├── train.py                ## Fine-tuning procedure
    │   ├── evaluate.py             ## Evaluation pipeline
    │   └── metrics.py              ## Evaluation metrics
    │
    ├── config/
    │   └── settings.py             ## Environment configuration
    │
    └── utils/
        ├── io_utils.py             ## Data loading helpers
        ├── logging_utils.py        ## Logging utilities
        └── utils.py                ## Shared helpers
```

---

## ❓ Problem Statement

Medical symptom descriptions are often:

* unstructured
* inconsistent in terminology
* expressed using multiple synonyms
* prone to labeling ambiguity

These characteristics make **reliable label normalization difficult**.

This project addresses the problem through:

* dataset cleaning and normalization
* supervised LLM fine-tuning
* deterministic inference constraints
* strict evaluation metrics for label consistency

---

## 🧠 Approach / Methodology / Strategy

The system follows a structured fine-tuning workflow composed of three stages.

### Dataset Preparation

* Load raw symptom descriptions
* Normalize labels
* Deduplicate samples
* Split into **train / validation / test datasets**

---

### Model Training

* Load instruction-tuned base LLM
* Apply **LoRA / QLoRA adapters**
* Perform **Supervised Fine-Tuning (SFT)**
* Export trained adapters and metadata

---

### Evaluation

Evaluation focuses on **label reliability and hallucination control**.

Metrics include:

* exact match accuracy
* hallucination rate
* confusion analysis
* label coverage diagnostics

---

## 🏗 Pipeline Architecture

```text
Raw Dataset
      ↓
Dataset Normalization
      ↓
Train / Validation / Test Split
      ↓
LLM Fine-Tuning (LoRA / QLoRA)
      ↓
Adapter Export
      ↓
Evaluation
      ↓
Prediction Export
```

---

## 📊 Exploratory Data Analysis

EDA helps analyze dataset quality before training:

* label distribution
* dataset size diagnostics
* duplicate detection
* vocabulary analysis

Outputs can be stored in:

```
artifacts/exports/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python **3.10+**
* Docker & Docker Compose
* Optional GPU with CUDA support

---

### 2. OS prerequisites

Verify that required packages are installed.

#### Windows / WSL2 (recommended)

```powershell
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

---

### 3. Python environment

```bash
python -m venv .lf_env
source .lf_env/bin/activate		     ## for windows .lf_env\Scripts\activate.bat
pip install --upgrade pip            ## for windows : .lf_env\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt

```

---

### 4. Docker setup

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up
```

---

## ▶️ Usage & End-to-End Testing

```bash
## Prepare dataset
python main.py prepare

## Train model
python main.py train

## Evaluate model
python main.py evaluate --run-dir artifacts/runs/run_YYYYMMDD_HHMMSS

## Run full pipeline
python main.py full

## Inspect training runs
ls artifacts/runs

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                     | Cause                          | Solution                           |
| ------------------------- | ------------------------------ | ---------------------------------- |
| Training failure          | Missing GPU dependencies       | Install correct CUDA / torch build |
| Dataset preparation error | Invalid label list             | Verify `LABEL_LIST_FILE` contents  |
| Evaluation failure        | Missing training run directory | Provide valid `--run-dir`          |
| Docker container failure  | Environment misconfiguration   | Rebuild containers                 |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Medical AI / LLM Fine-Tuning Project
