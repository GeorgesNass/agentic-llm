# local-quantization

## Overview

`local-quantization` is an end-to-end **local LLM quantization project** focused on **efficient inference on CPU and GPU**, while preserving a strong **speed ↔ quality trade-off**.

The application provides:
- Multiple quantization backends (**GGUF / AWQ / GPTQ / bitsandbytes / ONNX**)
- CPU and GPU inference targets (Windows / Linux / cloud VM)
- Calibration-aware post-training quantization (PTQ)
- Deterministic benchmarking for fair comparison
- CLI-based pipeline execution
- Docker-based execution

This project is designed as a **technical / portfolio / production-ready POC** for **local and edge-friendly LLM inference**.

---

## Project Structure

```text
local-quantization/
├── main.py
├── menu_pipeline.sh
├── requirements.txt
├── README.md
├── .env
├── pytest.ini
├── tests/
│   └── test_unit.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── base/
│   └── adapters/
├── artifacts/
│   ├── runs/
│   ├── exports/
│   └── benchmarks/
└── src/
    ├── __init__.py
    ├── pipeline.py
    ├── core/
    │   ├── __init__.py
    │   ├── quantize.py
    │   ├── calibration.py
    │   ├── model_loader.py
    │   ├── backends.py
    │   ├── export.py
    │   └── errors.py
    ├── config/
    │   ├── __init__.py
    │   ├── schemas.py
    │   └── settings.py
    ├── inference/
    │   ├── __init__.py
    │   ├── runners.py
    │   └── decoding.py
    └── utils/
        ├── __init__.py
        ├── logging_utils.py
        └── utils.py
```

---

## Global Variables

The following variables are required to configure **model loading, quantization, export, and benchmarking**.

| Variable name | Description | Placeholder |
|--------------|------------|-------------|
| PIPELINE_MODE | quantize \| export \| benchmark \| full | `full` |
| MODEL_NAME_OR_PATH | HuggingFace model id or local path | `<MODEL_NAME>` |
| QUANT_BACKEND | Quantization backend | `gguf / awq / gptq / bnb_nf4 / onnx` |
| QUANT_BITS | Target bit-width | `4 / 8` |
| CALIBRATION_DATASET | Calibration text dataset | `./data/calibration.txt` |
| EXPORT_OUTPUT_DIR | Exported artifacts directory | `./artifacts/exports` |
| BENCHMARK_PROMPTS | Prompt file for benchmarking | `./data/bench_prompts.txt` |

---

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Optional Nvidia GPU with CUDA support

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

## ✅ Full System Verification (End-to-End)

Run the following commands in order:

```bash
# Sanity check: Python environment and core dependencies
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
python -c "import bitsandbytes as bnb; print('bnb', bnb.__version__)"
python -c "import awq; print('awq import OK')"
python -c "import auto_gptq; print('auto_gptq OK')"
python -c "import onnx; print('onnx', onnx.__version__)"
python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"

# Check project structure
ls src
ls artifacts

# Print resolved pipeline configuration
python main.py --print-config

# Run quantization pipeline
python main.py

# Inspect outputs (if export enabled)
ls artifacts/exports

# Run unit tests
pytest -q

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

1. **Quantization**
   - Load base model from Hugging Face or local path
   - Optional LoRA adapter validation
   - Load calibration dataset (AWQ / GPTQ / ONNX)
   - Run backend-specific quantization

2. **Export**
   - Persist quantized artifacts (GGUF / INT4 / ONNX)
   - Save metadata and configuration snapshot

3. **Benchmark**
   - Deterministic decoding
   - Latency and throughput measurement
   - CPU / GPU comparison

---

## CLI Usage

### Quantize model
```bash
PIPELINE_MODE=quantize python main.py
```

### Export quantized artifacts
```bash
PIPELINE_MODE=export python main.py
```

### Benchmark model
```bash
PIPELINE_MODE=benchmark python main.py
```

### Run full pipeline
```bash
PIPELINE_MODE=full python main.py
```

Or via interactive menu:
```bash
bash menu_pipeline.sh
```

---

## Tests

```bash
pytest
```

Tests cover:
- Environment parsing utilities
- Configuration validation
- Error helpers

---

## Author

Georges Nassopoulos  
Email: georges.nassopoulos@gmail.com  
Status: Technical / Portfolio project

