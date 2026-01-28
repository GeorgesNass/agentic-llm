# rag-drive-gcp

## Overview

`rag-drive-gcp` is an end-to-end **Retrieval-Augmented Generation (RAG)** project
built on **Google Cloud Platform**.

The application provides:
- Google Drive document ingestion
- Local or remote OCR processing
- Text chunking and embedding generation with Vertex AI
- Persistent storage of text and embeddings in Google Cloud Storage
- Interactive RAG chat interface using Streamlit
- CLI pipeline for ingestion and querying
- Docker-based execution

This project is designed as a **technical / portfolio / production-ready POC**.

---

## Project Structure

```
    rag-drive-gcp/
    ├── main.py                     # Streamlit application entrypoint
    ├── launch_pipeline.py          # CLI pipeline launcher
    ├── requirements.txt
    ├── README.md
    ├── .env
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── data/                       # Local data workspace
    ├── logs/                       # Application logs
    ├── tests/
    │   └── test_core.py            # Minimal unit tests
    └── src/
        ├── __init__.py
        ├── core/
        │   ├── __init__.py
        │   ├── ocr.py
        │   ├── rag.py
        │   ├── vertex.py
        │   └── persistence.py
        ├── io/
        │   ├── __init__.py
        │   ├── drive.py
        │   └── gcs.py
        ├── model/
        │   ├── __init__.py
        │   └── settings.py
        └── utils/
            ├── __init__.py
            ├── logging_utils.py
            └── utils.py

```
---

## Global GCS Variables

The following variables are required to enable **persistent storage**
of text and embeddings artifacts in Google Cloud Storage.

| Variable name | Description | Placeholder |
|--------------|------------|-------------|
| `GCS_BUCKET_TEXT` | GCS bucket for extracted text files | `<YOUR_GCS_TEXT_BUCKET>` |
| `GCS_PREFIX_TEXT` | Prefix path for text artifacts | `<YOUR_GCS_TEXT_PREFIX>` |
| `GCS_BUCKET_EMB` | GCS bucket for embeddings artifacts | `<YOUR_GCS_EMBEDDINGS_BUCKET>` |
| `GCS_PREFIX_EMB` | Prefix path for embeddings artifacts | `<YOUR_GCS_EMBEDDINGS_PREFIX>` |

---

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Google Cloud Platform account
- Service Account with:
  - Google Drive API access
  - Vertex AI permissions
  - Google Cloud Storage read/write permissions

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
cd ~/Desktop/git_projects/
python -m venv .rag_env
source .rag_env/bin/activate ## .rag_env\Scripts\activate.bat for windows
pip install --upgrade pip
pip install -r requirements.txt
```
---

## Docker Usage

### Build and start the application
```bash
    docker compose build
    docker compose up
```
The Streamlit application will be available at:
```
    http://localhost:8501
```

---

## Application Workflow

1. **Ingestion**
   - Documents are listed and downloaded from Google Drive
   - OCR is applied if needed (local Docker or remote microservice)
   - Text is chunked and embedded using Vertex AI
   - Artifacts are uploaded to Google Cloud Storage

2. **Persistence**
   - Embeddings and metadata are stored in GCS
   - Index can be reloaded on application restart

3. **RAG Chat**
   - User questions are embedded
   - Relevant chunks are retrieved
   - The LLM generates contextual answers

---

## CLI Usage

### Run ingestion
```bash
    python launch_pipeline.py --mode ingest --drive-folder-id <FOLDER_ID>
```

### Run query
```bash
    python launch_pipeline.py --mode query --question "What is this document about?"
```
---

## Tests

Minimal unit tests are provided.
```bash
    pytest
```

Tests cover:
- Settings loading
- Text chunking
- Basic RAG retrieval logic

---

## Author

Georges Nassopoulos  
Email: georges.nassopoulos@gmail.com  
Status: Technical / Portfolio project
