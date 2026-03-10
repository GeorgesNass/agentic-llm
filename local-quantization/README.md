# ⚡ Local LLM Quantization Pipeline

The project provides an **end-to-end pipeline for local LLM quantization**, enabling efficient  **CPU and GPU inference** while maintaining a strong **speed ↔ quality trade-off**.

---

## 🎯 Project Overview

Main capabilities:

* Multiple quantization backends (**GGUF / AWQ / GPTQ / bitsandbytes / ONNX**)
* CPU and GPU inference targets
* Calibration-aware post-training quantization
* Deterministic benchmarking for fair comparison
* CLI-driven pipeline execution
* Docker-based execution

The system converts base LLM checkpoints into **optimized quantized artifacts suitable for local inference and edge deployment**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* PyTorch
* HuggingFace Transformers
* bitsandbytes
* GPTQ
* AWQ
* GGUF / llama.cpp
* ONNX / ONNX Runtime
* Docker & Docker Compose

---

## 📂 Project Structure

```text
local-quantization/
├── main.py                         ## Pipeline entry point
├── menu_pipeline.sh                ## Interactive CLI pipeline launcher
├── requirements.txt                ## Python dependencies
├── README.md                       ## Project documentation
├── .env                            ## Environment configuration
├── pytest.ini                      ## Pytest configuration
│
├── tests/
│   └── test_unit.py                ## Unit tests
│
├── docker/
│   ├── Dockerfile                  ## Container definition
│   └── docker-compose.yml          ## Docker orchestration
│
├── models/
│   ├── base/                       ## Base model checkpoints
│   └── adapters/                   ## Optional LoRA adapters
│
├── artifacts/
│   ├── runs/                       ## Pipeline run metadata
│   ├── exports/                    ## Quantized model artifacts
│   └── benchmarks/                 ## Benchmark results
│
└── src/
    ├── pipeline.py                 ## Pipeline orchestration
    │
    ├── core/
    │   ├── quantize.py             ## Quantization logic
    │   ├── calibration.py          ## Calibration dataset loading
    │   ├── model_loader.py         ## Model loading utilities
    │   ├── backends.py             ## Backend abstraction
    │   ├── export.py               ## Artifact export utilities
    │   └── errors.py               ## Custom exceptions
    │
    ├── config/
    │   ├── schemas.py              ## Configuration schemas
    │   └── settings.py             ## Environment configuration
    │
    ├── inference/
    │   ├── runners.py              ## Inference backends
    │   └── decoding.py             ## Token decoding strategies
    │
    └── utils/
        ├── logging_utils.py        ## Logging utilities
        └── utils.py                ## Shared helper functions
```

---

## ❓ Problem Statement

Running large language models locally presents several challenges:

* large model sizes
* high memory requirements
* slow inference on CPU
* fragmented quantization tooling
* difficulty benchmarking quantization strategies fairly

This project addresses these issues by providing a **unified pipeline for quantization, export, and benchmarking across multiple backends**.

---

## 🧠 Approach / Methodology / Strategy

The system provides a reproducible quantization workflow composed of three main stages.

### Quantization

* Load base model from HuggingFace or local checkpoint
* Optionally load LoRA adapters
* Run calibration using a representative dataset
* Execute backend-specific quantization

Supported backends:

| Backend      | Description                       |
| ------------ | --------------------------------- |
| GGUF         | llama.cpp optimized CPU inference |
| AWQ          | Activation-aware quantization     |
| GPTQ         | Post-training weight quantization |
| bitsandbytes | 4-bit / 8-bit quantization        |
| ONNX         | Graph-optimized inference         |

### Export

* Persist quantized artifacts
* Store metadata and configuration snapshot
* Export compatible runtime formats

### Benchmark

* Deterministic decoding
* Latency and throughput measurement
* CPU vs. GPU comparison

---

## 🏗 Pipeline Architecture

```text
Base Model (HF / local)
        ↓
Calibration Dataset
        ↓
Quantization Backend
(GGUF / AWQ / GPTQ / BNB / ONNX)
        ↓
Quantized Model
        ↓
Artifact Export
        ↓
Benchmarking
        ↓
Performance Reports
```

---

## 📊 Exploratory Data Analysis

The project includes benchmarking diagnostics such as:

* inference latency
* tokens per second
* memory consumption
* backend comparison metrics

Benchmark outputs are stored in:

```
artifacts/benchmarks/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python 3.10+
* Docker & Docker Compose
* Optional Nvidia GPU

---

### 2. OS prerequisites

Verify that required packages are installed.

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
python --version
```

---

### 3. Python environment

```bash
## Create the virtual environment
python -m venv .lq_env

## Activate the virtual environment
source .lq_env/bin/activate ## for windows : .lq_env\Scripts\activate.bat

## Upgrade build tooling (important for pyproject builds like autoawq)
python -m pip install --upgrade pip setuptools wheel ## for windows : .lq_env\Scripts\python.exe -m pip install --upgrade pip setuptools wheel

## Pin NumPy to <2 (bitsandbytes compatibility)
python -m pip install -U "numpy<2"

## Install PyTorch first (CPU)
python -m pip install "torch==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu

## Install a compatible Transformers stack
python -m pip install -U "transformers==4.38.2" "tokenizers==0.15.2"

## Install project deps (WITHOUT duplicates with the lines above)
python -m pip install -r requirements.txt

## Install AWQ WITHOUT letting pip change torch/transformers
python -m pip install --no-build-isolation --no-deps "autoawq==0.2.4"

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
## Verify environment dependencies
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__)"

## Verify optional quantization libraries
python -c "import bitsandbytes as bnb; print('bnb', bnb.__version__)"
python -c "import awq; print('awq import OK')"
python -c "import auto_gptq; print('auto_gptq OK')"

## Check ONNX runtime
python -c "import onnx; print('onnx', onnx.__version__)"
python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"

## Inspect project structure
ls src
ls artifacts

## Print pipeline configuration
python main.py --print-config

## Run quantization pipeline
python main.py

## Inspect exported artifacts
ls artifacts/exports

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                       | Cause                                   | Solution                            |
| --------------------------- | --------------------------------------- | ----------------------------------- |
| Torch installation mismatch | Incorrect CPU/GPU wheel                 | Install correct torch build         |
| AWQ build failure           | Dependency conflicts                    | Install with `--no-build-isolation` |
| NumPy compatibility issue   | NumPy ≥2 incompatible with bitsandbytes | Install `numpy<2`                   |
| ONNX runtime error          | Missing runtime backend                 | Install `onnxruntime`               |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Local AI / Quantization Engineering Project
