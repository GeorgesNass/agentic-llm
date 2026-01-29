# local-finetuning

## Overview

`local-finetuning` is an end-to-end **local LLM fine-tuning project**
focused on **domain adaptation and label normalization**.

The application provides:
- Dataset preparation and normalization
- Supervised Fine-Tuning (SFT) with **LoRA / QLoRA**
- Deterministic inference to reduce hallucinations
- Strict evaluation metrics for label consistency
- CLI-based pipeline execution
- Docker-based execution

This project is designed as a **technical / portfolio / production-ready POC**
for **medical symptom normalization (CISP)**.

---

## Project Structure

```text
local-finetuning/
├── main.py
├── entrypoint.sh
├── requirements.txt
├── README.md
├── .env
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── runs/
│   └── exports/
├── tests/
│   └── test_unit.py
└── src/
    ├── __init__.py
    ├── pipeline.py
    ├── core/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── prepare_dataset.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── metrics.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    └── utils/
        ├── __init__.py
        ├── io_utils.py
        ├── logging_utils.py
        └── utils.py
```

## Global Variables

The following variables are required to configure **data paths, training, and evaluation**.

| Variable name | Description | Placeholder |
|--------------|------------|-------------|
| BASE_MODEL_NAME | HuggingFace base model identifier | `<MODEL_NAME>` |
| USE_GPU | Enable GPU usage | `true / false` |
| DATA_DIR | Root data directory | `./data` |
| OUTPUT_DIR | Training output directory | `./artifacts/runs` |
| LABEL_LIST_FILE | Allowed CISP label list | `./data/labels.txt` |

---

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Optional GPU with CUDA support

---

## Windows & WSL2 Prerequisites

### PowerShell
```powershell
wsl --status
wsl --install
wsl --list --online
wsl --install -d Ubuntu
wsl -d Ubuntu
docker --version
docker compose version
```

### Ubuntu
```bash
    sudo apt update
    sudo apt install -y git
    git --version
```
### Python
```bash
    python3 --version
    sudo apt install -y python3-pip python3-venv
```
---

## Setup

### Manual installation
```bash
    python -m venv .lf_env
    source .lf_env/bin/activate ## .lf_env\Scripts\activate.bat for windows
    pip install --upgrade pip
    pip install -r requirements.txt
```
---

## Docker Usage

### Build and start the pipeline
```bash
    docker compose build
    docker compose up
```

---

## Application Workflow

1. **Dataset Preparation**
   - Load raw symptom descriptions
   - Normalize labels
   - Deduplicate samples
   - Split into train / validation / test sets

2. **Training**
   - Load instruction-tuned base LLM
   - Apply LoRA / QLoRA adapters
   - Supervised Fine-Tuning (SFT)
   - Export LoRA adapters and metadata

3. **Evaluation**
   - Deterministic inference
   - Exact match accuracy
   - Hallucination rate
   - Confusion and coverage diagnostics

---

## CLI Usage

### Prepare dataset
```bash
    python main.py prepare
```
### Train model
```bash
    python main.py train
```

### Evaluate model
```bash
    python main.py evaluate --run-dir artifacts/runs/run_YYYYMMDD_HHMMSS
```

### Run full pipeline
```bash
    python main.py full
```
---

## Tests
```bash
    pytest
```

Tests cover:
- Dataset preparation
- Evaluation metrics
- Label normalization consistency

---

## Author

Georges Nassopoulos  
Email: georges.nassopoulos@gmail.com  
Status: Technical / Portfolio project
