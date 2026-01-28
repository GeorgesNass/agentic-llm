'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Project settings and Pydantic models for rag-drive-gcp (Drive → OCR → GCS → RAG)."
'''

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.utils.utils import ensure_directories, get_project_root, load_dotenv_if_present, to_bool
from src.core.errors import log_and_raise_missing_env

## ============================================================
## SETTINGS MODEL
## ============================================================
class Settings(BaseModel):
    """
        Central application settings loaded from environment variables

        Notes:
            - This project expects a `.env` file at the project root
            - Environment variables are loaded using a lightweight parser
    """

    ## Core
    environment: str = Field(default="dev", description="Execution environment: dev|prod.")
    app_version: str = Field(default="1.0.0", description="Application version.")
    log_level: str = Field(default="INFO", description="Log level: DEBUG|INFO|WARNING|ERROR.")

    ## Google Drive
    drive_folder_id: Optional[str] = Field(default=None, description="Default Google Drive folder ID.")

    ## OCR
    ocr_mode: str = Field(
        default="local_docker",
        description="OCR mode: local_docker|remote_service.",
    )
    ocr_docker_image: Optional[str] = Field(
        default=None,
        description="Docker image name for OCR Universal when using local_docker.",
    )
    ocr_service_url: Optional[str] = Field(
        default=None,
        description="Base URL for OCR microservice when using remote_service.",
    )

    ## GCP / Vertex
    gcp_project_id: Optional[str] = Field(default=None, description="GCP project ID.")
    gcp_region: Optional[str] = Field(default=None, description="GCP region for Vertex AI.")
    vertex_llm_model: Optional[str] = Field(default=None, description="Vertex AI LLM model name.")
    vertex_embed_model: Optional[str] = Field(default=None, description="Vertex AI embedding model name.")

    ## GCS (separate locations for text and embeddings)
    gcs_bucket_text: Optional[str] = Field(default=None, description="GCS bucket for text artifacts.")
    gcs_prefix_text: str = Field(default="texts/", description="GCS prefix for text artifacts.")
    gcs_bucket_embeddings: Optional[str] = Field(default=None, description="GCS bucket for embeddings artifacts.")
    gcs_prefix_embeddings: str = Field(default="embeddings/", description="GCS prefix for embeddings artifacts.")

    ## Chunking / Retrieval
    chunk_size: int = Field(default=1024, description="Chunk size for document splitting.")
    chunk_overlap: int = Field(default=128, description="Overlap size for document splitting.")
    top_k: int = Field(default=8, description="Top-K chunks to retrieve.")

    ## Local storage and cleanup
    keep_local: bool = Field(default=False, description="Keep local traces after successful pipeline execution.")

    ## Local directories
    data_dir: Path = Field(default_factory=lambda: get_project_root() / "data")
    logs_dir: Path = Field(default_factory=lambda: get_project_root() / "logs")

    raw_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "raw")
    tmp_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "tmp")
    traces_dir: Path = Field(default_factory=lambda: get_project_root() / "data" / "traces")

## ============================================================
## PYDANTIC MODELS (PIPELINE + UI)
## ============================================================
class DriveFileMeta(BaseModel):
    """
        Metadata for a Google Drive file

        Attributes:
            file_id (str): Drive file ID
            name (str): File name
            mime_type (str): MIME type
            modified_time (Optional[str]): Last modification time
    """

    file_id: str
    name: str
    mime_type: str
    modified_time: Optional[str] = None

class IngestionStatus(BaseModel):
    """
        Pipeline status output after an ingestion run

        Attributes:
            drive_folder_id (str): Ingested folder ID
            downloaded_files (int): Number of downloaded files
            ocr_processed_files (int): Number of files that went through OCR
            uploaded_text_files (int): Number of text artifacts uploaded to GCS
            uploaded_embedding_files (int): Number of embedding artifacts uploaded to GCS
            message (str): Human-readable summary
            details (Dict[str, Any]): Optional structured details (counts, lists, errors)
    """

    drive_folder_id: str
    downloaded_files: int = 0
    ocr_processed_files: int = 0
    uploaded_text_files: int = 0
    uploaded_embedding_files: int = 0
    message: str = "ok"
    details: Dict[str, Any] = Field(default_factory=dict)

class RagAnswer(BaseModel):
    """
        RAG answer container

        Attributes:
            question (str): Input question
            answer (str): Generated answer
            sources (List[Dict[str, Any]]): Optional sources metadata
    """

    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)

## ============================================================
## PLACEHOLDER VALIDATION
## ============================================================
def _validate_no_placeholders(settings: Settings) -> None:
    """
        Validate that no environment variable still contains a placeholder value

        This project uses placeholders like `<YOUR_VALUE_HERE>` in `.env`
        If placeholders are detected, the application fails fast with a clear error

        Args:
            settings (Settings): Loaded settings instance

        Raises:
            ValueError: If one or more settings values contain `<YOUR_`.
    """
    
    invalid = []

    for key, value in settings.model_dump().items():
        if isinstance(value, str) and "<YOUR_" in value:
            invalid.append(key)

    if invalid:
        log_and_raise_missing_env(invalid)

## ============================================================
## SETTINGS FACTORY (CACHED)
## ============================================================
@lru_cache(maxsize=1)
def get_settings(env_path: Optional[Path] = None) -> Settings:
    """
        Build and cache the application settings

        Args:
            env_path (Optional[Path]): Optional .env path override

        Returns:
            Settings: Loaded settings instance
    """
    
    ## Load .env first (if any)
    load_dotenv_if_present(env_path)

    ## Build settings from environment variables
    settings = Settings(
        environment=os.getenv("ENVIRONMENT", "dev"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        drive_folder_id=os.getenv("DRIVE_FOLDER_ID"),
        ocr_mode=os.getenv("OCR_MODE", "local_docker"),
        ocr_docker_image=os.getenv("OCR_DOCKER_IMAGE"),
        ocr_service_url=os.getenv("OCR_SERVICE_URL"),
        gcp_project_id=os.getenv("GCP_PROJECT_ID"),
        gcp_region=os.getenv("GCP_REGION"),
        vertex_llm_model=os.getenv("VERTEX_LLM_MODEL"),
        vertex_embed_model=os.getenv("VERTEX_EMBED_MODEL"),
        gcs_bucket_text=os.getenv("GCS_BUCKET_TEXT"),
        gcs_prefix_text=os.getenv("GCS_PREFIX_TEXT", "texts/"),
        gcs_bucket_embeddings=os.getenv("GCS_BUCKET_EMB"),
        gcs_prefix_embeddings=os.getenv("GCS_PREFIX_EMB", "embeddings/"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "128")),
        top_k=int(os.getenv("TOP_K", "8")),
        keep_local=to_bool(os.getenv("KEEP_LOCAL"), default=False),
    )

    ## Ensure local directories exist
    ensure_directories(
        settings.data_dir,
        settings.logs_dir,
        settings.raw_dir,
        settings.tmp_dir,
        settings.traces_dir,
    )
    
    _validate_no_placeholders(settings)
    
    return settings