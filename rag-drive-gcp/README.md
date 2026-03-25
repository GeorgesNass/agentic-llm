# ☁️ RAG Pipeline with Google Drive & Vertex AI (GCP)

The project implements an **end-to-end Retrieval-Augmented Generation (RAG) pipeline**
built on **Google Cloud Platform**.

---

## 🎯 Project Overview

Main capabilities:

* Google Drive document ingestion
* Local or remote OCR processing
* Text chunking and embedding generation using **Vertex AI**
* Persistent storage of text and embeddings in **Google Cloud Storage**
* Interactive **RAG chat interface with Streamlit**
* CLI pipeline for ingestion and querying
* Docker-based execution

The system transforms documents stored in **Google Drive** into **searchable knowledge bases enabling contextual question answering**.

---

## ⚙️ Tech Stack

Core technologies used in the project:

* Python
* Streamlit
* Google Cloud Platform
* Google Drive API
* Google Cloud Storage (GCS)
* Vertex AI embeddings
* Retrieval-Augmented Generation (RAG)
* Docker & Docker Compose

---

## 📂 Project Structure

```text
rag-drive-gcp/
├── main.py                         ## Streamlit application entrypoint
├── launch_pipeline.py              ## CLI pipeline launcher
├── requirements.txt                ## Python dependencies
├── README.md                       ## Project documentation
├── .env                            ## Environment configuration
│
├── docker/
│   ├── Dockerfile                  ## Docker image definition
│   └── docker-compose.yml          ## Docker Compose configuration
│
├── data/                           ## Local data workspace
├── logs/                           ## Application logs
│
├── tests/
│   └── test_unit.py                ## Minimal unit tests
│
└── src/
    ├── pipeline.py                 ## End-to-end orchestration
    │
    ├── core/
    │   ├── auth.py                 ## JWT auth: tokens, login, refresh, dependencies
    │   ├── security.py             ## RBAC, middleware, permissions, request security	
    │   ├── ocr.py                  ## OCR processing
    │   ├── rag.py                  ## RAG retrieval pipeline
    │   ├── vertex.py               ## Vertex AI embeddings
    │   └── persistence.py          ## GCS persistence utilities
    │
    ├── io/
    │   ├── drive.py                ## Google Drive ingestion
    │   └── gcs.py                  ## Google Cloud Storage operations
    │
    ├── model/
    │   └── settings.py             ## Application configuration
    │
    └── utils/
        ├── logging_utils.py        ## Logging utilities
        └── utils.py                ## Shared helpers
```

---

## ❓ Problem Statement

Many organizations store large amounts of documents in **Google Drive**.

Challenges include:

* difficulty searching large document collections
* lack of semantic search capabilities
* manual document exploration
* absence of contextual question answering

This project solves these issues using a **cloud-native RAG pipeline** combining:

* document ingestion
* OCR extraction
* embedding-based retrieval
* contextual LLM responses

---

## 🧠 Approach / Methodology / Strategy

The system implements a **three-stage RAG architecture**.

### Document Ingestion

* Documents are listed and downloaded from Google Drive
* OCR is applied if required
* Text is extracted and normalized

---

### Embedding & Storage

* Text is split into chunks
* Chunks are embedded using **Vertex AI**
* Text and embeddings are stored in **Google Cloud Storage**

---

### RAG Query

* User questions are embedded
* Relevant chunks are retrieved
* The LLM generates contextual answers

---

## 🏗 Pipeline Architecture

```text id="7xti2o"
Google Drive
      ↓
Document Download
      ↓
OCR (optional)
      ↓
Text Chunking
      ↓
Vertex AI Embeddings
      ↓
Storage in GCS
      ↓
Vector Retrieval
      ↓
LLM Response
      ↓
Streamlit RAG Chat
```

---

## 📊 Exploratory Data Analysis

The project allows diagnostics on the ingestion process:

* number of documents processed
* chunk statistics
* embedding generation diagnostics
* ingestion logs

Artifacts can be stored in:

```
logs/
data/
```

---

## 🔧 Setup & Installation

In this section we explain the minimum OS verification, python usage and docker setup.

### 1. Requirements

* Python **3.10+**
* Docker & Docker Compose
* Google Cloud Platform account
* Service Account with:

  * Google Drive API access
  * Vertex AI permissions
  * Google Cloud Storage read/write permissions

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
python -m venv .rag_env
source .rag_env/bin/activate		     ## for windows .rag_env\Scripts\activate.bat
pip install --upgrade pip            ## for windows : .rag_env\Scripts\python.exe -m pip install --upgrade pip
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
## Run document ingestion
python launch_pipeline.py --mode ingest --drive-folder-id <FOLDER_ID>

## Run query
python launch_pipeline.py --mode query --question "What is this document about?"

## Launch Streamlit interface
streamlit run main.py

## Verify interface
curl -X GET http://localhost:8501

## Run tests
pytest -q
```

---

## 📛 Common Errors & Troubleshooting

| Error                         | Cause                               | Solution                     |
| ----------------------------- | ----------------------------------- | ---------------------------- |
| Google Drive API error        | Invalid service account permissions | Verify Drive API access      |
| Vertex AI embedding failure   | Incorrect GCP configuration         | Verify Vertex AI credentials |
| GCS upload failure            | Bucket permission error             | Check GCS IAM roles          |
| Streamlit application failure | Dependency mismatch                 | Reinstall requirements       |

---

## 👤 Author

**Georges Nassopoulos**
[georges.nassopoulos@gmail.com](mailto:georges.nassopoulos@gmail.com)

**Status:** Cloud RAG / LLM Engineering Project
