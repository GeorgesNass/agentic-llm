'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Airflow DAG for ingestion: folder scan -> chunking -> embeddings -> vector store persistence (calls src/ directly)."
'''

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.orchestrator.retrieval import ingest_folder
from src.utils.env_utils import _get_env_bool, _get_env_str
from src.utils.logging_utils import get_logger

## ============================================================
## LOGGER
## ============================================================
logger = get_logger("airflow.ingestion_dag")

## ============================================================
## DAG DEFAULTS
## ============================================================
DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "georges_nassopoulos",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

## ============================================================
## TASKS
## ============================================================
def run_ingestion_task() -> None:
    """
        Run RAG ingestion by calling src/ modules directly

        High-level workflow:
            1) Resolve ingestion folder (RAW_DIR)
            2) Resolve vector store persistence options via env
            3) Call src.orchestrator.retrieval.ingest_folder
            4) Persist index to artifacts/vector_store

        Env variables:
            RAW_DIR: Folder containing source documents
            USE_GPU: Whether GPU is allowed
            PREFER_LOCAL_EMBEDDINGS: Prefer local embeddings (SentenceTransformers)
            VECTOR_STORE_BACKEND: faiss or chroma
            VECTOR_STORE_DIR: Persistence directory
            VECTOR_INDEX_NAME: Index/collection name
            INGEST_OVERWRITE: Whether to overwrite index/collection
            INGEST_MAX_FILES: Max number of files to ingest (0 means no limit)

        Returns:
            None
    """

    ## Resolve ingestion folder
    raw_dir = _get_env_str("RAW_DIR", "./data/raw")
    folder = Path(raw_dir).expanduser().resolve()

    ## Read ingestion flags
    use_gpu = _get_env_bool("USE_GPU", False)
    prefer_local = _get_env_bool("PREFER_LOCAL_EMBEDDINGS", True)

    ## Read vector store parameters
    backend = _get_env_str("VECTOR_STORE_BACKEND", "faiss").strip().lower()
    store_dir = _get_env_str("VECTOR_STORE_DIR", "./artifacts/vector_store")
    index_name = _get_env_str("VECTOR_INDEX_NAME", "default").strip() or "default"

    ## Optional ingestion controls
    overwrite = _get_env_bool("INGEST_OVERWRITE", False)
    max_files_raw = _get_env_str("INGEST_MAX_FILES", "0").strip()
    try:
        max_files = int(max_files_raw)
    except ValueError:
        max_files = 0

    logger.info(
        "Ingestion start | folder=%s | backend=%s | index=%s | prefer_local=%s | use_gpu=%s | overwrite=%s | max_files=%s",
        str(folder),
        backend,
        index_name,
        prefer_local,
        use_gpu,
        overwrite,
        max_files,
    )

    ## Validate folder existence
    if not folder.exists():
        raise FileNotFoundError(f"RAW_DIR folder not found: {folder}")

    ## Call src pipeline (business logic stays in src/)
    ingest_folder(
        folder=str(folder),
        prefer_local=prefer_local,
        use_gpu=use_gpu,
        vector_store_backend=backend,
        vector_store_dir=store_dir,
        index_name=index_name,
        overwrite=overwrite,
        max_files=max_files if max_files > 0 else None,
    )

    logger.info("Ingestion success | folder=%s | index=%s", str(folder), index_name)

## ============================================================
## DAG
## ============================================================
with DAG(
    dag_id="autonomous_ai_platform_ingestion",
    description="Ingestion pipeline (RAG): folder scan -> chunk -> embeddings -> vector store",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=["autonomous-ai-platform", "rag", "ingestion"],
) as dag:
    ingest_documents = PythonOperator(
        task_id="ingest_documents",
        python_callable=run_ingestion_task,
    )

    ingest_documents